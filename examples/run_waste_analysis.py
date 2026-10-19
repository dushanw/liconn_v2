"""Step-waste analysis of the authors' FFN traces (docs/waste_analysis_design.md).

  run_waste_analysis.py                        benchmark seeds except dendrite_1525 (11)
  run_waste_analysis.py --only-notebook-seed   the notebook seed only (~2.5 min)
  run_waste_analysis.py --resummarize STEPS    recompute summaries from a recorded
                                               waste_steps.npz (no tracing)

Outputs:
  outputs/waste_analysis.json   per-seed and pooled step classes and distributions
  outputs/waste_steps.npz       per-step arrays, keyed <field>_<seed name>
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
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
from liconn_v2 import waste  # pylint: disable=g-import-not-at-top

EXCLUDED = ('dendrite_1525',)  # owner decision, 2026-10-19


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--only-notebook-seed', action='store_true')
  p.add_argument('--out', default='outputs')
  p.add_argument('--resummarize')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(exist_ok=True)
  if a.resummarize:
    resummarize(Path(a.resummarize), out)
    return

  raw = reference.normalize(data.load_raw_xyz())
  gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
  seeds = [s for s in benchmark.select_seeds(gt) if s['name'] not in EXCLUDED]
  if a.only_notebook_seed:
    seeds = [s for s in seeds if s['name'] == benchmark.NOTEBOOK_SEED_NAME]

  started = time.time()
  per_seed, arrays, checks = [], {}, {}
  with reference.Tracer(data.checkpoint_path()) as tracer:
    for s in seeds:
      rec = waste.StepRecorder()
      trace = tracer.trace(raw, s['seed_zyx'], instrument=rec.attach)
      steps = rec.arrays()
      summary = waste.seed_summary(steps)
      summary.update(name=s['name'], cls=s['class'], trace_seconds=trace.trace_seconds)
      predicted = int((~np.isnan(trace.seed)).sum())
      checks[s['name']] = {
          'steps_equal_iterations': len(steps['new_fg']) == trace.iterations,
          'new_voxels_sum_equals_predicted': int((steps['new_fg'] + steps['new_bg']).sum()) == predicted,
          'established_owners_sum_equals_predicted': int(steps['established_owner_voxels'].sum()) == predicted,
          'positions_equal_history': [tuple(p) for p in steps['positions'].tolist()] == trace.history,
      }
      if s['name'] == benchmark.NOTEBOOK_SEED_NAME:
        fp = reference.fingerprint(trace)
        checks[s['name']]['matches_milestone1_reference'] = fp == reference.load_expected()
      per_seed.append(summary)
      for k, v in steps.items():
        arrays[f"{k}_{s['name']}"] = v
      print(f"{s['name']}: steps={summary['steps']} idle={summary['idle_share']:.3f} "
            f"low_yield={summary['low_yield_share']:.3f} mask_idle={summary['mask_idle_share']:.3f} "
            f"mask_low_yield={summary['mask_low_yield_share']:.3f} enclosed={summary['enclosed_share']:.3f} "
            f"{trace.trace_seconds:.0f}s", flush=True)

  np.savez_compressed(out / 'waste_steps.npz', **arrays)
  groups = group_seeds(per_seed)
  result = {
      'design': 'docs/waste_analysis_design.md',
      'subset': 'notebook seed only' if a.only_notebook_seed else 'benchmark seeds except dendrite_1525',
      'excluded': list(EXCLUDED),
      'per_seed': per_seed,
      'pooled': {k: waste.pooled(v) for k, v in groups.items()},
      'checks': checks,
      'all_checks_pass': all(all(c.values()) for c in checks.values()),
      'wall_seconds_total': time.time() - started,
      'inputs_sha256': data.PINNED_SHA256,
  }
  (out / 'waste_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result['pooled'], indent=2))
  if not result['all_checks_pass']:
    raise SystemExit('Instrumentation invariant failed: ' + json.dumps(checks))


def group_seeds(per_seed):
  return {
      'all': per_seed,
      'axon': [s for s in per_seed if s['cls'] == 'axon'],
      'dendrite': [s for s in per_seed if s['cls'] == 'dendrite'],
      'all_without_axon_218': [s for s in per_seed if s['name'] != 'axon_218'],
      'axon_without_axon_218': [s for s in per_seed if s['cls'] == 'axon' and s['name'] != 'axon_218'],
  }


def resummarize(steps_path: Path, out: Path):
  """Recomputes per-seed and pooled summaries from recorded per-step arrays."""
  classes = {'notebook': 'unclassified'}
  with np.load(steps_path) as f:
    names = sorted(k[len('new_fg_'):] for k in f.files if k.startswith('new_fg_'))
    per_seed = []
    for name in names:
      a = {k: f[f'{k}_{name}'] for k in (*waste.FIELDS, 'final_owner_voxels', 'established_owner_voxels')}
      summary = waste.seed_summary(a)
      summary.update(name=name, cls=classes.get(name, name.split('_')[0]))
      per_seed.append(summary)
  result = {
      'design': 'docs/waste_analysis_design.md',
      'recomputed_from': str(steps_path),
      'per_seed': per_seed,
      'pooled': {k: waste.pooled(v) for k, v in group_seeds(per_seed).items()},
  }
  (out / 'waste_analysis_resummarized.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result['pooled'], indent=2))


if __name__ == '__main__':
  main()
