"""Fresh-seed verdict, ground-truth metrics and ERL (docs/fresh_design.md).

  judge_fresh.py --recorded [--out DIR]

Judges fp16_head32 on the 300 fresh seeds with the owner-confirmed many-seed rule
(decisive), and reports absolute metrics and the adapted ERL for the fresh seeds
(float32, fp16_head32) and for the 300 many-seed seeds (float32, float16, fp16_head32),
from the runs recorded in the data folder. ERL is withheld for a seed set whose
skeletons fail the pre-registered sanity gate.

Output: DIR/fresh_report.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top
from scipy import ndimage  # pylint: disable=g-import-not-at-top

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import erl  # pylint: disable=g-import-not-at-top
from liconn_v2 import fresh  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top

FRESH_RUNS = {'fp32': 'fresh-fp32-20261023', 'fp16_head32': 'fresh-fp16_head32-20261023'}
MANYSEED_RUNS = {'fp32': 'manyseed-fp32-20261021', 'fp16': 'manyseed-fp16-20261021',
                 'fp16_head32': 'manyseed-fp16_head32-20261022'}


def load(run_dir: Path) -> dict:
  return json.loads((run_dir / 'benchmark.json').read_text())


def metrics(records: list) -> dict:
  """Absolute accuracy against the proofread segmentation, over all seeds."""
  tp = np.array([r['correct_voxels'] for r in records], float)
  m = np.array([r['mask_voxels'] for r in records], float)
  g = np.array([r['object_voxels'] for r in records], float)
  return {'seeds': len(records),
          'mean_recall': float(np.mean([r['recall'] for r in records])),
          'mean_precision': float(np.mean([r['precision'] for r in records])),
          'mean_iou': float(np.mean(tp / (m + g - tp))), 'mean_dice': float(np.mean(2 * tp / (m + g))),
          'large_splits': sum(r['recall'] < 0.5 for r in records),
          'any_merge': sum(1 for r in records if erl.merged(r)),
          'large_merge': sum(1 for r in records if r['mask_voxels'] > 0 and r['merged_voxels'] >= 0.1 * r['mask_voxels']),
          'merged_voxels': int(sum(r['merged_voxels'] for r in records)),
          'steps': int(sum(r['steps'] for r in records)),
          'ms_per_step': 1000 * sum(r['trace_seconds'] for r in records) / sum(r['steps'] for r in records)}


def full_mask(npz, name, shape) -> np.ndarray:
  m = np.zeros(shape, bool)
  if 'mask_' + name in npz.files:
    o, a = npz['offset_zyx_' + name], npz['mask_' + name]
    m[o[0]:o[0] + a.shape[0], o[1]:o[1] + a.shape[1], o[2]:o[2] + a.shape[2]] = a
  return m


PAD = 2  # voxels around an object's box, so that kimimaro's fix_borders sees only true crop borders


def padded_box(obj_xyz: np.ndarray) -> tuple:
  sl = ndimage.find_objects(obj_xyz.astype(np.uint8))[0]
  return tuple(slice(max(0, s.start - PAD), min(n, s.stop + PAD)) for s, n in zip(sl, obj_xyz.shape))


def manyseed_labels_xyz(k: int) -> np.ndarray:
  manyseed.verify_crop(k)
  return np.load(manyseed.crop_dir(k) / manyseed.PROOFREAD_FILE)


def fresh_verdict(ref: list, cand: list, seeds: list, pinned: list, crops: list) -> dict:
  """manyseed.verdict for runs paired by position over the fresh crops (numbered from 13):
  each seed's crop is mapped to the position of its crop's run pair."""
  position = {k: i for i, k in enumerate(crops)}
  return manyseed.verdict(ref, cand, [{**s, 'crop': position[s['crop']]} for s in seeds], pinned)


def seed_set(seeds: list, labels_xyz, run_names: dict) -> dict:
  """Metrics and ERL for one seed set (a list of seeds with crops) and its runs."""
  runs = data.cache_dir() / 'runs'
  terms = {v: [] for v in run_names}
  records = {v: [] for v in run_names}
  skeletons = []
  for k in sorted({s['crop'] for s in seeds}):
    lab = labels_xyz(k)
    per_run = {}
    for v, d in run_names.items():
      crop = runs / d / f'crop-{k:02d}'
      per_run[v] = (np.load(crop / 'benchmark_masks.npz'), {r['name']: r for r in load(crop)['records']})
    for s in (x for x in seeds if x['crop'] == k):
      obj = lab == s['object_id']  # x, y, z
      sl = padded_box(obj)
      skel = erl.skeleton(obj[sl], [x.start for x in sl])
      skeletons.append(skel)
      for v, (npz, recs) in per_run.items():
        rec = recs[s['name']]
        records[v].append(rec)
        mask = full_mask(npz, s['name'], lab.shape[::-1])  # z, y, x
        terms[v].append({**erl.seed_terms(skel, mask, erl.merged(rec)), 'class': s['class']})
  gate = erl.gate(skeletons)
  out = {'gate': gate, 'metrics': {v: metrics(r) for v, r in records.items()}}
  out['erl'] = ({v: erl.summarize(t) for v, t in terms.items()} if gate['passes']
                else 'withheld: skeletons failed the sanity gate')
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--recorded', action='store_true', required=True)
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  runs = data.cache_dir() / 'runs'
  crops = [c['index'] for c in fresh.load_manifest()['crops']]
  ref = [load(runs / FRESH_RUNS['fp32'] / f'crop-{k:02d}') for k in crops]
  cand = [load(runs / FRESH_RUNS['fp16_head32'] / f'crop-{k:02d}') for k in crops]
  ckpt = {k: v for k, v in data.PINNED_SHA256.items() if k.startswith('ckpt/')}
  pinned = [{**c['sha256'], **ckpt} for c in fresh.load_manifest()['crops']]
  verdict = fresh_verdict(ref, cand, fresh.recorded_seeds(), pinned, crops)
  report = {
      'design': 'docs/fresh_design.md',
      'verdict': verdict,
      'fresh': seed_set(fresh.recorded_seeds(), fresh.load_proofread_xyz, FRESH_RUNS),
      'manyseed': seed_set(manyseed.recorded_seeds(), manyseed_labels_xyz, MANYSEED_RUNS),
  }
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / 'fresh_report.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps({'passes': verdict['passes'], 'counts': verdict['counts'], 'rules': verdict['rules'],
                    'fresh_erl': report['fresh']['erl'], 'fresh_gate': report['fresh']['gate'],
                    'manyseed_erl': report['manyseed']['erl'], 'manyseed_gate': report['manyseed']['gate']},
                   indent=2))


if __name__ == '__main__':
  main()
