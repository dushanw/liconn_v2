"""Predictability of low-yield FFN steps (docs/predictability_design.md).

  run_predictability.py                        waste-analysis seeds (11; no dendrite_1525)
  run_predictability.py --only-notebook-seed   the notebook seed only (~2.5 min on CPU)
  run_predictability.py --reference RUN_DIR    also require every seed's canvas and
                                               visit-order sha256 to equal RUN_DIR's
  run_predictability.py --from-steps STEPS.npz recompute the analysis from a recorded
                                               predictability_steps.npz (no tracing)

Set LICONN_BACKEND=gpu for the GPU reference backend (see cluster/wahab/README.md).

Outputs:
  outputs/predictability.json       checks, feature AUCs, leave-one-seed-out results
  outputs/predictability_steps.npz  per-step features and targets, keyed <field>_<seed>
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
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
from liconn_v2 import waste  # pylint: disable=g-import-not-at-top

EXCLUDED = ('dendrite_1525',)  # owner decision, 2026-10-19 (as in the waste analysis)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--only-notebook-seed', action='store_true')
  p.add_argument('--reference', help='benchmark run whose per-seed hashes must match')
  p.add_argument('--out', default='outputs')
  p.add_argument('--from-steps')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  if a.from_steps:
    reanalyse(Path(a.from_steps), out)
    return

  raw = reference.normalize(data.load_raw_xyz())
  gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
  seeds = [s for s in benchmark.select_seeds(gt) if s['name'] not in EXCLUDED]
  if a.only_notebook_seed:
    seeds = [s for s in seeds if s['name'] == benchmark.NOTEBOOK_SEED_NAME]
  ref = {}
  if a.reference:
    ref = {r['name']: r for r in json.loads((Path(a.reference) / 'benchmark.json').read_text())['records']}

  started = time.time()
  per_seed, arrays, checks = {}, {}, {}
  with reference.Tracer(data.checkpoint_path()) as tracer:
    for s in seeds:
      rec = waste.StepRecorder()
      trace = tracer.trace(raw, s['seed_zyx'], instrument=rec.attach)
      steps = rec.arrays()
      canvas = reference.canvas_sha256(trace.seed)
      history = reference.history_sha256(trace.history)
      c = {'steps_equal_iterations': len(steps['new_fg']) == trace.iterations,
           'positions_equal_history': [tuple(q) for q in steps['positions'].tolist()] == trace.history}
      if a.reference:
        c.update(predict.reference_checks(s['name'], canvas, history, ref))
      if s['name'] == benchmark.NOTEBOOK_SEED_NAME and trace.backend == 'cpu':
        c['matches_milestone1_reference'] = reference.fingerprint(trace) == reference.load_expected()
      checks[s['name']] = c
      per_seed[s['name']] = steps
      for k, v in steps.items():
        arrays[f"{k}_{s['name']}"] = v
      print(f"{s['name']}: steps={len(steps['new_fg'])} checks={'ok' if all(c.values()) else c} "
            f"{trace.trace_seconds:.0f}s", flush=True)

  np.savez_compressed(out / 'predictability_steps.npz', **arrays)
  table = predict.step_table(per_seed)
  result = {
      'design': 'docs/predictability_design.md',
      'backend': reference.EXPECTED_BACKEND,
      'subset': [s['name'] for s in seeds],
      'excluded': list(EXCLUDED),
      'reference': a.reference,
      'checks': checks,
      'all_checks_pass': all(all(c.values()) for c in checks.values()),
      'analysis': predict.analyse(table),
      'wall_seconds_total': time.time() - started,
      'inputs_sha256': data.PINNED_SHA256,
  }
  (out / 'predictability.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
  print(json.dumps(result['analysis'], indent=2))
  if not result['all_checks_pass']:
    raise SystemExit('A trace check failed: ' + json.dumps(checks))


def reanalyse(steps_path: Path, out: Path):
  """Analysis only, from per-step arrays recorded by an earlier run."""
  with np.load(steps_path) as f:
    names = sorted(k[len('new_fg_'):] for k in f.files if k.startswith('new_fg_'))
    per_seed = {n: {k[:-len(n) - 1]: f[k] for k in f.files if k.endswith('_' + n)} for n in names}
  result = {'design': 'docs/predictability_design.md', 'recomputed_from': str(steps_path),
            'analysis': predict.analyse(predict.step_table(per_seed))}
  (out / 'predictability_reanalysed.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
  print(json.dumps(result['analysis'], indent=2))


if __name__ == '__main__':
  main()
