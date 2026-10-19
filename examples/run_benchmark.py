"""Runs the FFN benchmark (docs/benchmark_design.md) and writes outputs/.

  run_benchmark.py                        all 12 seeds (CPU: tens of minutes or more)
  run_benchmark.py --only-notebook-seed   the authors' notebook seed only (~2.5 min)

Outputs:
  outputs/benchmark.json        seeds, per-seed accuracy and cost, summary, provenance
  outputs/benchmark_masks.npz   each seed's FFN object mask, cropped to its bounding box
                                (mask_<name>, offset_zyx_<name>; full shape in shape_zyx)
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--only-notebook-seed', action='store_true')
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(exist_ok=True)

  raw = reference.normalize(data.load_raw_xyz())
  gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
  seeds = benchmark.select_seeds(gt)
  if a.only_notebook_seed:
    seeds = [s for s in seeds if s['name'] == benchmark.NOTEBOOK_SEED_NAME]

  started = time.time()
  def progress(rec):
    print(f"{rec['name']}: steps={rec['steps']} recall={rec['recall']:.3f} "
          f"precision={rec['precision']:.3f} merges={len(rec['merges'])} "
          f"{rec['trace_seconds']:.0f}s", flush=True)
  records, masks = benchmark.run(seeds, raw, gt, data.checkpoint_path(), on_record=progress)

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

  versions = {n: importlib.metadata.version(n) for n in ('ffn', 'connectomics', 'jax', 'jaxlib', 'numpy', 'scipy')}
  result = {
      'design': 'docs/benchmark_design.md',
      'subset': 'notebook seed only' if a.only_notebook_seed else 'all seeds',
      'seeds': [{**s, 'seed_zyx': list(s['seed_zyx'])} for s in seeds],
      'records': records,
      'summary': benchmark.summarize(records),
      'inputs_sha256': data.PINNED_SHA256,
      'wall_seconds_total': time.time() - started,
      'platform': {'machine': platform.machine(), 'cpu_count': os.cpu_count(),
                   'python': sys.version.split()[0]},
      'versions': versions,
  }
  (out / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
  main()
