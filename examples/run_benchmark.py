"""Runs the FFN benchmark (docs/benchmark_design.md) and writes outputs/.

  run_benchmark.py                        all 12 seeds (CPU: tens of minutes or more)
  run_benchmark.py --only-notebook-seed   the authors' notebook seed only (~2.5 min)
  run_benchmark.py --seed NAME [--seed NAME ...]   selected seeds (e.g. one per cluster job)

Set LICONN_BACKEND=gpu to run on an NVIDIA GPU (see cluster/wahab/README.md).

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


def provenance() -> dict:
  """Where and with what this run executed (GPU runs need more than CPU runs)."""
  versions = {}
  for d in importlib.metadata.distributions():
    name = d.metadata.get('Name', '')
    if name in ('ffn', 'connectomics', 'jax', 'jaxlib', 'numpy', 'scipy') or \
        name.startswith(('jax-cuda', 'nvidia-')):
      versions[name] = d.version
  return {
      'backend': reference.EXPECTED_BACKEND,
      'machine': platform.machine(), 'node': platform.node(), 'cpu_count': os.cpu_count(),
      'python': sys.version.split()[0],
      'xla_flags': os.environ.get('XLA_FLAGS', ''),
      'slurm_job': os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID'),
      'slurm_task': os.environ.get('SLURM_ARRAY_TASK_ID'),
      'versions': dict(sorted(versions.items())),
  }


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--only-notebook-seed', action='store_true')
  p.add_argument('--seed', action='append', default=[], help='seed name; repeatable')
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)

  raw = reference.normalize(data.load_raw_xyz())
  gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
  all_seeds = benchmark.select_seeds(gt)
  names = list(a.seed) + ([benchmark.NOTEBOOK_SEED_NAME] if a.only_notebook_seed else [])
  seeds = benchmark.pick(all_seeds, names)

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

  result = {
      'design': 'docs/benchmark_design.md',
      'subset': 'all seeds' if len(seeds) == len(all_seeds) else [s['name'] for s in seeds],
      'seeds': [{**s, 'seed_zyx': list(s['seed_zyx'])} for s in seeds],
      'records': records,
      'summary': benchmark.summarize(records),
      'inputs_sha256': data.PINNED_SHA256,
      'wall_seconds_total': time.time() - started,
      'platform': provenance(),
  }
  (out / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
  main()
