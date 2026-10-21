"""Lower-precision screen (docs/precision_design.md, step 1).

  run_precision_screen.py [--seed NAME ...] [--every N] [--reference RUN_DIR] [--out DIR]

Traces the benchmark seeds with the authors' network (float32) and records the
network input of every N-th step. Then it compares each lower-precision variant
with float32 on those inputs and applies the selection rule fixed in advance. With
--reference, each seed's canvas and visit-order sha256 must equal that run's. Set
LICONN_BACKEND=gpu on the cluster; CPU timings are not meaningful.

Output: DIR/precision_screen.json
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
from liconn_v2 import predict  # pylint: disable=g-import-not-at-top
from liconn_v2 import pruned  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--seed', action='append', default=[])
  p.add_argument('--every', type=int, default=8)
  p.add_argument('--batch', type=int, default=1, help='inputs per network call (tracing uses 1)')
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
  inputs, checks = [], {}
  with reference.Tracer(data.checkpoint_path()) as tracer:
    params, device = tracer.params, tracer.device_kind
    for s in seeds:
      rec = pruned.InputRecorder(every=a.every)
      trace = tracer.trace(raw, s['seed_zyx'], instrument=rec.attach)
      c = {'recorded_every_nth_step': rec.steps == list(range(0, trace.iterations, a.every))}
      if a.reference:
        c.update(predict.reference_checks(s['name'], reference.canvas_sha256(trace.seed),
                                          reference.history_sha256(trace.history), ref))
      checks[s['name']] = c
      inputs.extend(rec.inputs)
      print(f"{s['name']}: steps={trace.iterations} recorded={len(rec.inputs)} checks={c}", flush=True)
  if not all(all(c.values()) for c in checks.values()):
    raise SystemExit('A trace check failed: ' + json.dumps(checks))

  variants = [v for v in precision.VARIANTS if v != 'fp32']
  scr = precision.screen(params, inputs, variants, batch=a.batch)
  result = {
      'design': 'docs/precision_design.md',
      'backend': reference.EXPECTED_BACKEND,
      'device_kind': device,
      'seeds': [s['name'] for s in seeds],
      'every': a.every,
      'inputs_total': len(inputs),
      'reference': a.reference,
      'checks': checks,
      **scr,
      'selected_for_benchmark': precision.select_for_benchmark(scr['variants']),
      'wall_seconds_total': time.time() - started,
      'inputs_sha256': data.PINNED_SHA256,
  }
  (out / 'precision_screen.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
  print(f"fp32: {scr['fp32_seconds_per_eval'] * 1000:.2f} ms/eval on {device}")
  for r in scr['variants']:
    print(f"{r['variant']}: flip={r['flip_rate']:.6f} lost={r['lost_rate']:.6f} "
          f"non_finite={r['non_finite_rate']:.6f} ({r['inputs_with_non_finite']} inputs) "
          f"max|d|={r['max_abs_logit_diff']} {r['seconds_per_eval'] * 1000:.2f} ms "
          f"(x{r['speedup_vs_fp32']:.2f})")
  print('selected:', result['selected_for_benchmark'])


if __name__ == '__main__':
  main()
