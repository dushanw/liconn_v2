"""Residual-block drop screen (docs/block_drop_design.md, step 1).

  run_block_screen.py [--seed NAME ...] [--every N] [--reference RUN_DIR] [--out DIR]

Traces the benchmark seeds with the authors' network and records the network input
of every N-th step. Then it measures, on those inputs, every single-block drop and
the greedy multi-block sequence, and applies the pre-registered selection rule.
With --reference, each seed's canvas and visit-order sha256 must equal that run's.
Set LICONN_BACKEND=gpu on the cluster.

Output: DIR/block_screen.json
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
from liconn_v2 import predict  # pylint: disable=g-import-not-at-top
from liconn_v2 import pruned  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--seed', action='append', default=[])
  p.add_argument('--every', type=int, default=8)
  p.add_argument('--max-blocks', type=int, default=pruned.DEPTH, help='greedy search depth')
  p.add_argument('--reference')
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)

  raw = reference.normalize(data.load_raw_xyz())
  gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
  seeds = benchmark.pick(benchmark.select_seeds(gt), a.seed)
  ref = {}
  if a.reference:
    ref = {r['name']: r for r in json.loads((Path(a.reference) / 'benchmark.json').read_text())['records']}

  started = time.time()
  inputs, checks, counts = [], {}, {}
  with reference.Tracer(data.checkpoint_path()) as tracer:
    params = tracer.params
    for s in seeds:
      rec = pruned.InputRecorder(every=a.every)
      trace = tracer.trace(raw, s['seed_zyx'], instrument=rec.attach)
      c = {'recorded_every_nth_step': rec.steps == list(range(0, trace.iterations, a.every))}
      if a.reference:
        c.update(predict.reference_checks(s['name'], reference.canvas_sha256(trace.seed),
                                          reference.history_sha256(trace.history), ref))
      checks[s['name']] = c
      counts[s['name']] = len(rec.inputs)
      inputs.extend(rec.inputs)
      print(f"{s['name']}: steps={trace.iterations} recorded={len(rec.inputs)} checks={c}", flush=True)
  if not all(all(c.values()) for c in checks.values()):
    raise SystemExit('A trace check failed: ' + json.dumps(checks))

  scr = pruned.Screen(params, inputs)
  singles = [scr.measure(f'drop_{i}', (i,)) for i in range(pruned.DEPTH)]
  greedy = pruned.sequential_greedy(scr.measure, max_blocks=a.max_blocks)
  selected = pruned.select_variants(singles, greedy)
  result = {
      'design': 'docs/block_drop_design.md',
      'backend': reference.EXPECTED_BACKEND,
      'seeds': [s['name'] for s in seeds],
      'every': a.every,
      'greedy': 'sequential (lowest combined flip rate at each step)',
      'inputs_recorded': counts,
      'inputs_total': len(inputs),
      'reference': a.reference,
      'checks': checks,
      'full_gflop_per_eval': pruned.gflop_per_eval(params, ()),
      'single_drops': singles,
      'greedy_path': greedy,
      'selected_for_benchmark': [{'label': l, 'skip': list(s)} for l, s in selected],
      'wall_seconds_total': time.time() - started,
      'inputs_sha256': data.PINNED_SHA256,
  }
  (out / 'block_screen.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
  for r in singles:
    print(f"{r['label']}: flip={r['flip_rate']:.5f} lost={r['lost_rate']:.5f} "
          f"shift={r['mean_logit_shift']:+.4f}")
  for r in greedy:
    print(f"{r['label']} {r['skip']}: flip={r['flip_rate']:.5f} gflop={r['gflop_per_eval']:.1f}")
  print('selected:', result['selected_for_benchmark'])


if __name__ == '__main__':
  main()
