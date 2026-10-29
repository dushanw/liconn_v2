"""FFN against fine-tuned SegNeuron on crop 12 (docs/headtohead_design.md).

  run_headtohead.py ffn       --out DIR   # DIR/ffn_crop12.json (LICONN_BACKEND=gpu)
  run_headtohead.py segneuron --out DIR --infer-wall-seconds S   # DIR/segneuron_crop12.json
  run_headtohead.py compare   --out DIR --ffn-recorded PATH --sn-recorded PATH --sn-recorded-infer DIR
                              # DIR/headtohead_crop12.json

Between ffn and segneuron (cluster/wahab/headtohead.sbatch), the authors' inference.py with the
fine-tuned best.ckpt writes DIR/infer/{affinities.npy,boundaries.tif,inference.json}; the job
times that process. No test crop is read.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
from liconn_v2 import headtohead as hh  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import segneuron  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position
from run_segneuron import load_outputs  # pylint: disable=g-import-not-at-top,wrong-import-position


def ffn(out: Path):
  from liconn_v2 import data  # pylint: disable=g-import-not-at-top
  from liconn_v2 import earlystop  # pylint: disable=g-import-not-at-top
  from liconn_v2 import fasttrace  # pylint: disable=g-import-not-at-top
  from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
  gt = manyseed.load_proofread_zyx(hh.CROP)
  objects = vitseg.tuning_objects(gt)
  raw = reference.normalize(manyseed.load_raw_xyz(hh.CROP))
  rows = []
  t = time.perf_counter()
  with reference.Tracer(data.checkpoint_path()) as tracer:
    earlystop.install(tracer, hh.FFN_VARIANT)
    fast = fasttrace.FastTracer(tracer)
    fast.trace(raw, tuple(objects[0]['seed_zyx']))  # warm-up: compiles the tracer's GPU functions
    setup = time.perf_counter() - t
    claims = np.zeros(raw.shape, np.uint16)  # FFN masks per voxel (traces are independent)
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
      print(json.dumps({k: r[k] for k in ('name', 'steps', 'recall', 'precision', 'wall_seconds')}), flush=True)
    compile_seconds = tracer.compile_seconds
  claimed = int((claims > 0).sum())
  r = {'design': 'docs/headtohead_design.md', 'crop': hh.CROP, 'variant': hh.FFN_VARIANT,
       'objects': len(rows), 'setup_seconds': setup, 'compile_seconds': compile_seconds,
       'overlap': {'voxels_in_any_mask': claimed, 'voxels_in_two_or_more': int((claims > 1).sum()),
                   'sum_of_mask_voxels': int(sum(x['mask_voxels'] for x in rows)), 'max_masks_on_a_voxel': int(claims.max())},
       'records': rows, 'platform': provenance()}
  (out / 'ffn_crop12.json').write_text(json.dumps(r, indent=1) + '\n')


def sn(out: Path, infer_wall_seconds: float):
  gt = manyseed.load_proofread_zyx(hh.CROP)
  objects = vitseg.tuning_objects(gt)
  aff, fg = load_outputs(out / 'infer')
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
  seg = vitseg.segmentation_at(frag, merges, hh.THRESHOLD)
  times['segmentation_seconds'] = time.perf_counter() - t
  post = sum(times.values())
  acc = segneuron.object_accounting(seg, gt, objects)
  recs = {r['object_id']: r for r in vitseg.seed_records(seg, gt, objects)}
  for row in acc['objects']:
    row['merges'] = recs[row['object_id']]['merges']
    row['merged_voxels'] = recs[row['object_id']]['merged_voxels']
  r = {'design': 'docs/headtohead_design.md', 'crop': hh.CROP, 'threshold': hh.THRESHOLD,
       'objects': len(objects), 'fragments': int(frag.max()), 'inference_wall_seconds': infer_wall_seconds,
       'inference': json.loads((out / 'infer' / 'inference.json').read_text()),
       **times, 'post_seconds': post, **acc, 'platform': provenance()}
  (out / 'segneuron_crop12.json').write_text(json.dumps(r, indent=1) + '\n')
  print(json.dumps({**segneuron.summary(acc), 'post_seconds': post}, default=str), flush=True)


def compare(out: Path, ffn_recorded: Path, sn_recorded: Path, sn_recorded_infer: Path):
  f = json.loads((out / 'ffn_crop12.json').read_text())
  s = json.loads((out / 'segneuron_crop12.json').read_text())
  rec = [r for r in json.loads(ffn_recorded.read_text())['records'] if r['object_id'] == hh.CHECK_OBJECT][0]
  mine = [r for r in f['records'] if r['object_id'] == hh.CHECK_OBJECT][0]
  old = {r['object_id']: r for r in json.loads(sn_recorded.read_text())['ours']['objects']}
  aff, fg = load_outputs(out / 'infer')
  aff0, fg0 = load_outputs(sn_recorded_infer)
  r = {'design': 'docs/headtohead_design.md', 'crop': hh.CROP, 'ffn_variant': hh.FFN_VARIANT,
       'segneuron_threshold': hh.THRESHOLD, 'accuracy': hh.paired(f['records'], s['objects']),
       'ffn_overlap': f['overlap'],
       'speed': hh.speed(f['records'], f['setup_seconds'], s['inference_wall_seconds'],
                         s['inference']['inference_seconds'], s['post_seconds']),
       'segneuron_post_breakdown': {k: s[k] for k in ('watershed_seconds', 'agglomeration_seconds',
                                                      'segmentation_seconds')},
       'checks': {'ffn_trace_matches_recorded': (mine['steps'], mine['canvas_sha256'], mine['history_sha256'])
                  == (rec['steps'], rec['canvas_sha256'], rec['history_sha256']),
                  'segneuron_max_abs_affinity_change_vs_recorded': float(np.abs(aff - aff0).max()),
                  'segneuron_max_abs_foreground_change_vs_recorded': float(np.abs(fg - fg0).max()),
                  'segneuron_max_abs_f1_change_vs_recorded':
                  float(max(abs(x['f1'] - old[x['object_id']]['f1']) for x in s['objects'])),
                  'segneuron_mean_f1_recorded': float(np.mean([x['f1'] for x in old.values()]))}}
  (out / 'headtohead_crop12.json').write_text(json.dumps(r, indent=1) + '\n')
  print(json.dumps(r, indent=1), flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('step', choices=('ffn', 'segneuron', 'compare'))
  p.add_argument('--out', required=True)
  p.add_argument('--infer-wall-seconds', type=float)
  p.add_argument('--ffn-recorded')
  p.add_argument('--sn-recorded')
  p.add_argument('--sn-recorded-infer')
  a = p.parse_args()
  out = Path(a.out)
  if a.step == 'ffn':
    ffn(out)
  elif a.step == 'segneuron':
    sn(out, a.infer_wall_seconds)
  else:
    compare(out, Path(a.ffn_recorded), Path(a.sn_recorded), Path(a.sn_recorded_infer))


if __name__ == '__main__':
  main()
