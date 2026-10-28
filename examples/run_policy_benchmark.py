"""The FFN benchmark with a modified movement policy (docs/postpone_design.md).

  run_policy_benchmark.py --policy NAME [--seed NAME ...] [--out DIR]

Policies: default (FFN's own), skip_enclosed, postpone, postpone_drop_enclosed.
Writes benchmark.json and benchmark_masks.npz in the same format as
examples/run_benchmark.py, so examples/merge_benchmark.py and
examples/compare_runs.py work on the result. Set LICONN_BACKEND=gpu for the GPU.
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
from liconn_v2 import policies  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top


def provenance() -> dict:
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
  p.add_argument('--policy', required=True, choices=policies.POLICIES)
  p.add_argument('--seed', action='append', default=[], help='seed name; repeatable')
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)

  raw = reference.normalize(data.load_raw_xyz())
  gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
  all_seeds = benchmark.select_seeds(gt)
  seeds = benchmark.pick(all_seeds, a.seed)

  started = time.time()
  def progress(rec):
    extra = ''
    if 'policy_stats' in rec:
      st = rec['policy_stats']
      extra = (f" postponed={st['postponed']} skipped={st['main_skipped_enclosed']}"
               f" dropped_enclosed={st['postponed_dropped_enclosed']}")
    print(f"{rec['name']}: steps={rec['steps']} recall={rec['recall']:.3f} "
          f"precision={rec['precision']:.3f}{extra} {rec['trace_seconds']:.0f}s", flush=True)
  records, masks = policies.run_policy(seeds, raw, gt, data.checkpoint_path(), a.policy,
                                       on_record=progress)

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
      'design': 'docs/postpone_design.md',
      'subset': 'all seeds' if len(seeds) == len(all_seeds) else [s['name'] for s in seeds],
      'policy': a.policy,
      'seeds': [{**s, 'seed_zyx': list(s['seed_zyx'])} for s in seeds],
      'records': records,
      'summary': benchmark.summarize(records),
      'inputs_sha256': data.PINNED_SHA256,
      'wall_seconds_total': time.time() - started,
      'platform': provenance(),
  }
  (out / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result['summary']['all'], indent=2))
  problems = {r['name']: r['policy_stats_problems'] for r in records if r.get('policy_stats_problems')}
  if problems:
    raise SystemExit('Policy bookkeeping failed: ' + json.dumps(problems))


if __name__ == '__main__':
  main()
