"""FFN and the fine-tuned SegNeuron on one unseen crop, one V100 (docs/unseen_design.md).

  run_unseen_wahab.py prep      --crop K --out DIR   # DIR/raw.npy (z, y, x), SegNeuron's input
  run_unseen_wahab.py ffn       --crop K --out DIR   # DIR/ffn.json (LICONN_BACKEND=gpu)
  run_unseen_wahab.py segneuron --crop K --out DIR --infer-wall-seconds S   # DIR/segneuron.json

As the crop-12 head-to-head (examples/run_headtohead.py, docs/headtohead_design.md), on crop K of
41-50: FFN in fp16_head32 with the fast tracer, one seed at a time; between ffn and segneuron the
job (cluster/wahab/unseen.sbatch) times the authors' inference.py with the fine-tuned best.ckpt on
DIR/raw.npy, writing DIR/infer/. Our pipeline then runs at the fixed threshold 0.6. --crop 11 with
--smoke N runs the same steps on the first N objects of the validation crop instead.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import headtohead as hh  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import segneuron  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position
from run_segneuron import load_outputs  # pylint: disable=g-import-not-at-top,wrong-import-position

SMOKE_CROP = 11


def load_crop(k: int, smoke: int | None):
  """(raw x, y, z; labels z, y, x; objects). Refuses any crop but 41-50, or 11 with --smoke."""
  if smoke is None:
    unseen.check_crop(k)
    raw, gt = unseen.load_raw_xyz(k), unseen.load_proofread_zyx(k)
  elif k == SMOKE_CROP:
    raw, gt = manyseed.load_raw_xyz(k), manyseed.load_proofread_zyx(k)
  else:
    raise ValueError(f'smoke runs use crop {SMOKE_CROP} only')
  return raw, gt, vitseg.tuning_objects(gt)[:smoke]


def header(k: int, smoke: int | None, objects: list) -> dict:
  sha = unseen._entry(k)['sha256'] if smoke is None else manyseed.load_manifest()['crops'][k]['sha256']  # pylint: disable=protected-access
  return {'design': 'docs/unseen_design.md', 'design_sha256': unseen.design_sha256(), 'crop': k,
          'smoke_objects': smoke, 'crop_sha256': sha, 'object_count': len(objects),
          'object_ids': [o['object_id'] for o in objects]}


def write(path: Path, r: dict) -> None:
  """Written whole or not at all: a job stopped while writing leaves no partial result file."""
  tmp = path.with_name(f'.{path.name}.tmp')
  tmp.write_text(json.dumps(r, indent=1) + '\n')
  tmp.replace(path)


def prep(k: int, smoke: int | None, out: Path):
  raw, _, _ = load_crop(k, smoke)
  out.mkdir(parents=True, exist_ok=True)
  np.save(out / 'raw.npy', np.ascontiguousarray(np.transpose(raw, [2, 1, 0])), allow_pickle=False)


def ffn(k: int, smoke: int | None, out: Path):
  from liconn_v2 import earlystop  # pylint: disable=g-import-not-at-top
  from liconn_v2 import fasttrace  # pylint: disable=g-import-not-at-top
  from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
  raw, gt, objects = load_crop(k, smoke)
  raw = reference.normalize(raw)
  rows = []
  claims = np.zeros(gt.shape, np.uint16)  # FFN masks per voxel (traces are independent)
  t = time.perf_counter()
  with reference.Tracer(data.checkpoint_path()) as tracer:
    earlystop.install(tracer, unseen.V100_FFN_VARIANT)
    fast = fasttrace.FastTracer(tracer)
    fast.trace(raw, tuple(objects[0]['seed_zyx']))  # warm-up: compiles the tracer's GPU functions
    setup = time.perf_counter() - t
    for o in objects:
      t = time.perf_counter()
      tr = fast.trace(raw, tuple(o['seed_zyx']))
      mask = benchmark.object_mask(np.asarray(tr.seed))
      wall = time.perf_counter() - t
      claims += mask
      r = {'name': o['name'], 'seed_zyx': o['seed_zyx'], **benchmark.score(mask, gt, o['seed_zyx'])}
      if r['object_id'] != o['object_id']:
        raise ValueError(f"seed of {o['object_id']} is on {r['object_id']}")
      r.update({'f1': hh.f1(r['precision'], r['recall']), 'steps': tr.iterations,
                'trace_seconds': tr.trace_seconds, 'wall_seconds': wall,
                'canvas_sha256': reference.canvas_sha256(tr.seed),
                'history_sha256': reference.history_sha256(tr.history)})
      rows.append(r)
      print(json.dumps({key: r[key] for key in ('name', 'steps', 'recall', 'precision', 'wall_seconds')}), flush=True)
    compile_seconds = tracer.compile_seconds
  ckpt = {f: data.sha256_file(data.cache_dir() / f) for f in data.CHECKPOINT_FILES}
  if ckpt != {f: data.PINNED_SHA256[f] for f in data.CHECKPOINT_FILES}:
    raise ValueError(f'FFN checkpoint {ckpt} is not the pinned one')
  r = {**header(k, smoke, objects), 'variant': unseen.V100_FFN_VARIANT, 'checkpoint_sha256': ckpt,
       'setup_seconds': setup,
       'compile_seconds': compile_seconds,
       'overlap': {'voxels_in_any_mask': int((claims > 0).sum()), 'voxels_in_two_or_more': int((claims > 1).sum()),
                   'max_masks_on_a_voxel': int(claims.max())},
       'records': rows, 'platform': provenance()}
  write(out / 'ffn.json', r)


def sn(k: int, smoke: int | None, out: Path, infer_wall_seconds: float, checkpoint: Path):
  _, gt, objects = load_crop(k, smoke)
  ckpt = data.sha256_file(checkpoint)
  if ckpt != unseen.SEGNEURON_CKPT_SHA256:
    raise ValueError(f'{checkpoint}: sha256 {ckpt}, expected {unseen.SEGNEURON_CKPT_SHA256}')
  aff, fg = load_outputs(out / 'infer')
  nonfinite = int((~np.isfinite(aff)).sum() + (~np.isfinite(fg)).sum())
  if fg.shape != gt.shape:
    raise ValueError(f'outputs {fg.shape} are not on the crop grid {gt.shape}')
  times = {}
  t = time.perf_counter()
  a = segneuron.vitseg_layout(aff, fg)
  frag = vitseg.fragments(a)
  times['watershed_seconds'] = time.perf_counter() - t
  t = time.perf_counter()
  merges = vitseg.agglomerate(*vitseg.fragment_edges(frag, a))
  times['agglomeration_seconds'] = time.perf_counter() - t
  t = time.perf_counter()
  seg = vitseg.segmentation_at(frag, merges, unseen.SEGNEURON_THRESHOLD)
  times['segmentation_seconds'] = time.perf_counter() - t
  post = sum(times.values())
  acc = segneuron.object_accounting(seg, gt, objects)
  recs = {r['object_id']: r for r in vitseg.seed_records(seg, gt, objects)}
  for row in acc['objects']:
    row['merges'] = recs[row['object_id']]['merges']
    row['merged_voxels'] = recs[row['object_id']]['merged_voxels']
  r = {**header(k, smoke, objects), 'threshold': unseen.SEGNEURON_THRESHOLD, 'checkpoint_sha256': ckpt,
       'nonfinite_outputs': nonfinite,
       'fragments': int(frag.max()),
       'inference_wall_seconds': infer_wall_seconds,
       'inference': json.loads((out / 'infer' / 'inference.json').read_text()),
       **times, 'post_seconds': post, **acc, 'platform': provenance()}
  write(out / 'segneuron.json', r)
  print(json.dumps({**segneuron.summary(acc), 'post_seconds': post}, default=str), flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('step', choices=('prep', 'ffn', 'segneuron'))
  p.add_argument('--crop', type=int, required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--smoke', type=int, default=None)
  p.add_argument('--infer-wall-seconds', type=float)
  p.add_argument('--sn-checkpoint', help='the fine-tuned best.ckpt that inference.py used')
  a = p.parse_args()
  out = Path(a.out)
  if a.step == 'prep':
    prep(a.crop, a.smoke, out)
  elif a.step == 'ffn':
    ffn(a.crop, a.smoke, out)
  else:
    sn(a.crop, a.smoke, out, a.infer_wall_seconds, Path(a.sn_checkpoint))


if __name__ == '__main__':
  main()
