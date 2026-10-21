"""The FFN benchmark with the network in lower precision (docs/precision_design.md, step 2).

  run_precision_benchmark.py --variant fp16 [--seed NAME ...] [--out DIR]

Writes benchmark.json and benchmark_masks.npz in the same format as
examples/run_benchmark.py (number_format per record), for merge_benchmark.py
and compare_runs.py. Set LICONN_BACKEND=gpu on the cluster.
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
from liconn_v2 import precision  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--variant', required=True, choices=sorted(precision.VARIANTS))
  p.add_argument('--seed', action='append', default=[])
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)

  raw = reference.normalize(data.load_raw_xyz())
  gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
  all_seeds = benchmark.select_seeds(gt)
  seeds = benchmark.pick(all_seeds, a.seed)
  started = time.time()
  records, masks = precision.run_variant(
      seeds, raw, gt, data.checkpoint_path(), a.variant,
      on_record=lambda r: print(f"{r['name']}: steps={r['steps']} recall={r['recall']:.3f} "
                                f"precision={r['precision']:.3f} {r['trace_seconds']:.0f}s", flush=True))

  arrays = {'shape_zyx': np.array(gt.shape)}
  for name, mask in masks.items():
    if not mask.any():
      continue
    zz, yy, xx = np.nonzero(mask)
    lo = np.array([zz.min(), yy.min(), xx.min()])
    hi = np.array([zz.max(), yy.max(), xx.max()]) + 1
    arrays['mask_' + name] = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    arrays['offset_zyx_' + name] = lo
  np.savez_compressed(out / 'benchmark_masks.npz', **arrays)
  result = {
      'design': 'docs/precision_design.md',
      'subset': 'all seeds' if len(seeds) == len(all_seeds) else [s['name'] for s in seeds],
      'number_format': a.variant,
      'seeds': [{**s, 'seed_zyx': list(s['seed_zyx'])} for s in seeds],
      'records': records,
      'summary': benchmark.summarize(records),
      'inputs_sha256': data.PINNED_SHA256,
      'wall_seconds_total': time.time() - started,
      'platform': provenance(),
  }
  (out / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result['summary']['all'], indent=2))


if __name__ == '__main__':
  main()
