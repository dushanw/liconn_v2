"""Combine and compare benchmark runs (docs/gpu_design.md).

A run is a directory with benchmark.json and benchmark_masks.npz as written by
examples/run_benchmark.py.
"""

import json
from pathlib import Path

import numpy as np

from liconn_v2 import benchmark

# Accuracy tolerance for speed variants, confirmed by the owner on 2026-10-19
# (docs/benchmark_design.md): every rule at 1%.
TOL_MEAN_RECALL_DROP = 0.01
TOL_MEAN_PRECISION_DROP = 0.01
TOL_MEAN_UNLABELED_RISE = 0.01
TOL_MERGED_VOXELS_GROWTH = 0.01
TOL_SEED_RECALL_DROP = 0.01
TOLERANCE_STATUS = 'confirmed by the owner on 2026-10-19 (all rules at 1%)'
# Reference run for speed variants, adopted by the owner on 2026-10-19
# (docs/gpu_baseline.md), relative to the data folder's runs/ directory.
REFERENCE_RUN = 'gpu-baseline-a-20261019/merged'

# Same method on another backend: flag seeds for owner review beyond these.
# Slack for floating-point rounding, so an exact 1-point change passes.
EPS = 1e-9
EQUIV_MIN_IOU = 0.99
EQUIV_MAX_STEP_CHANGE = 0.01
SENSITIVE_SEEDS = ('dendrite_1525', 'axon_218')


def load_run(run_dir) -> dict:
  run_dir = Path(run_dir)
  return json.loads((run_dir / 'benchmark.json').read_text())


def load_masks(run_dir) -> dict:
  """Returns ({seed name: full-size boolean mask}, volume shape); empty masks are absent."""
  run_dir = Path(run_dir)
  out = {}
  with np.load(run_dir / 'benchmark_masks.npz') as f:
    shape = tuple(int(v) for v in f['shape_zyx'])
    names = [k[len('mask_'):] for k in f.files if k.startswith('mask_')]
    for name in names:
      full = np.zeros(shape, bool)
      lo = f['offset_zyx_' + name]
      crop = f['mask_' + name]
      full[lo[0]:lo[0] + crop.shape[0], lo[1]:lo[1] + crop.shape[1], lo[2]:lo[2] + crop.shape[2]] = crop
      out[name] = full
  return out, shape


def mask_for(masks: dict, shape, name) -> np.ndarray:
  return masks.get(name, np.zeros(shape, bool))


def iou(a: np.ndarray, b: np.ndarray) -> float:
  union = int((a | b).sum())
  return int((a & b).sum()) / union if union else 1.0


def merge_runs(part_dirs, out_dir, canonical_names) -> dict:
  """Combines per-seed runs into one run in canonical seed order."""
  records, seeds, arrays, platforms = {}, {}, {}, []
  shape = None
  inputs = load_run(part_dirs[0]).get('inputs_sha256') if part_dirs else None
  for d in part_dirs:
    run = load_run(d)
    if run.get('inputs_sha256') != inputs:
      raise ValueError(f'{d} used different inputs (inputs_sha256) from the first part')
    platforms.append(run.get('platform'))
    for s in run['seeds']:
      seeds[s['name']] = s
    for r in run['records']:
      if r['name'] in records:
        raise ValueError(f"seed {r['name']} appears in more than one part")
      records[r['name']] = r
    with np.load(Path(d) / 'benchmark_masks.npz') as f:
      if shape is None:
        shape = f['shape_zyx']
      elif tuple(f['shape_zyx']) != tuple(shape):
        raise ValueError(f'{d} has mask volume {tuple(f["shape_zyx"])}, expected {tuple(shape)}')
      for k in f.files:
        if k != 'shape_zyx':
          arrays[k] = f[k]
  missing = [n for n in canonical_names if n not in records]
  extra = sorted(set(records) - set(canonical_names))
  if missing or extra:
    raise ValueError(f'incomplete or unexpected seeds: missing {missing}, extra {extra}')
  ordered = [records[n] for n in canonical_names]
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out_dir / 'benchmark_masks.npz', shape_zyx=shape, **arrays)
  result = {
      'design': 'docs/benchmark_design.md',
      'subset': 'all seeds',
      'merged_from': [str(d) for d in part_dirs],
      'seeds': [seeds[n] for n in canonical_names],
      'records': ordered,
      'summary': benchmark.summarize(ordered),
      'inputs_sha256': inputs,
      'platforms': platforms,
  }
  (out_dir / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
  return result


def _tolerance(base_recs, cand_recs) -> dict | None:
  if not base_recs:
    return None
  b, c = benchmark.summarize(base_recs)['all'], benchmark.summarize(cand_recs)['all']
  bm = {r['name']: {m[0] for m in r['merges']} for r in base_recs}
  new_merges = {r['name']: sorted({m[0] for m in r['merges']} - bm[r['name']])
                for r in cand_recs if {m[0] for m in r['merges']} - bm[r['name']]}
  br = {r['name']: r['recall'] for r in base_recs}
  big_drops = {r['name']: br[r['name']] - r['recall'] for r in cand_recs
               if br[r['name']] - r['recall'] > TOL_SEED_RECALL_DROP + EPS}
  # Relative growth is undefined when the base has no merged voxels; report None and
  # count any new merged voxels as outside the tolerance.
  growth = ((c['total_merged_voxels'] - b['total_merged_voxels']) / b['total_merged_voxels']
            if b['total_merged_voxels'] else (0.0 if c['total_merged_voxels'] == 0 else None))
  checks = {
      'mean_recall_drop': b['mean_recall'] - c['mean_recall'],
      'mean_precision_drop': b['mean_precision'] - c['mean_precision'],
      'mean_unlabeled_rise': c['mean_unlabeled_fraction'] - b['mean_unlabeled_fraction'],
      'merged_voxels_growth': growth,
      'merged_voxels_base_and_candidate': [b['total_merged_voxels'], c['total_merged_voxels']],
      'seeds_with_new_merges': new_merges,
      'seeds_with_recall_drop_over_limit': big_drops,
  }
  checks['within_tolerance'] = (
      checks['mean_recall_drop'] <= TOL_MEAN_RECALL_DROP + EPS
      and checks['mean_precision_drop'] <= TOL_MEAN_PRECISION_DROP + EPS
      and checks['mean_unlabeled_rise'] <= TOL_MEAN_UNLABELED_RISE + EPS
      and growth is not None and growth <= TOL_MERGED_VOXELS_GROWTH + EPS
      and not new_merges and not big_drops)
  return checks


def _check_run(run, masks, label):
  """Raises if a run's records, seeds and masks do not belong together."""
  seeds = {s['name']: s for s in run['seeds']}
  names = [r['name'] for r in run['records']]
  if sorted(names) != sorted(seeds) or len(set(names)) != len(names):
    raise ValueError(f'{label}: records {sorted(names)} do not match declared seeds {sorted(seeds)}')
  for r in run['records']:
    s = seeds[r['name']]
    if s.get('object_id', r['object_id']) != r['object_id'] or \
        list(r.get('seed_zyx', s['seed_zyx'])) != list(s['seed_zyx']):
      raise ValueError(f"{label}: record {r['name']} does not match its seed entry")
    if r['mask_voxels'] > 0 and r['name'] not in masks:
      raise ValueError(f"{label}: mask for {r['name']} is missing from benchmark_masks.npz")


def _check_same_inputs(base, cand):
  """Raises if the two runs used different inputs or different seeds under one name."""
  bi, ci = base.get('inputs_sha256'), cand.get('inputs_sha256')
  if bi is not None and ci is not None and bi != ci:
    raise ValueError('runs used different inputs (inputs_sha256 differ)')
  bs = {s['name']: s for s in base['seeds']}
  for s in cand['seeds']:
    b = bs.get(s['name'])
    if b is not None and (list(b['seed_zyx']) != list(s['seed_zyx'])
                          or b.get('object_id') != s.get('object_id')):
      raise ValueError(f"seed {s['name']} differs between runs: {b} vs {s}")
  return bi is not None and ci is not None


def compare_runs(base_dir, cand_dir) -> dict:
  """Per-seed and overall comparison of a candidate run with a base run."""
  base, cand = load_run(base_dir), load_run(cand_dir)
  inputs_checked = _check_same_inputs(base, cand)
  bmasks, shape = load_masks(base_dir)
  cmasks, cshape = load_masks(cand_dir)
  if tuple(shape) != tuple(cshape):
    raise ValueError(f'mask volumes differ: {shape} vs {cshape}')
  _check_run(base, bmasks, 'base')
  _check_run(cand, cmasks, 'candidate')
  brec = {r['name']: r for r in base['records']}
  crec = {r['name']: r for r in cand['records']}
  names = [n for n in brec if n in crec]
  per_seed = []
  for n in names:
    b, c = brec[n], crec[n]
    row = {'name': n, 'class': b['class'],
           'mask_iou': iou(mask_for(bmasks, shape, n), mask_for(cmasks, shape, n)),
           'base_steps': b['steps'], 'cand_steps': c['steps'],
           'same_history': (b.get('history_sha256') == c.get('history_sha256')
                            if b.get('history_sha256') and c.get('history_sha256') else None),
           'same_canvas': (b.get('canvas_sha256') == c.get('canvas_sha256')
                           if b.get('canvas_sha256') and c.get('canvas_sha256') else None)}
    for k in ('recall', 'precision', 'unlabeled_fraction', 'merged_voxels',
              'evals_per_1000_correct_voxels'):
      # The cost ratio is None for a trace with no correct voxels.
      row['delta_' + k] = None if b[k] is None or c[k] is None else c[k] - b[k]
    row['base_merges'] = b['merges']  # [object ID, overlapping voxels]
    row['cand_merges'] = c['merges']
    new_merges = sorted({m[0] for m in c['merges']} - {m[0] for m in b['merges']})
    recall_drop = b['recall'] - c['recall']
    # Per-seed parts of the tolerance (docs/benchmark_design.md). The mean
    # rules (recall, precision, unlabeled, merged-voxel growth) apply to groups only;
    # the per-seed deltas above show each seed's contribution.
    row['seed_tolerance'] = {
        'recall_drop': recall_drop,
        'recall_drop_within_limit': recall_drop <= TOL_SEED_RECALL_DROP + EPS,
        'new_merges': new_merges,
        'within_seed_rules': recall_drop <= TOL_SEED_RECALL_DROP + EPS and not new_merges,
    }
    step_change = abs(c['steps'] - b['steps']) / b['steps'] if b['steps'] else 0.0
    row['flag_for_review'] = row['mask_iou'] < EQUIV_MIN_IOU or step_change > EQUIV_MAX_STEP_CHANGE
    per_seed.append(row)
  groups = {'all': names,
            'without_sensitive_seeds': [n for n in names if n not in SENSITIVE_SEEDS]}
  overall = {}
  for g, ns in groups.items():
    # summarize() gives None for an empty group (e.g. a run of sensitive seeds only).
    overall[g] = {
        'seeds': len(ns),
        'base_summary': benchmark.summarize([brec[n] for n in ns])['all'],
        'cand_summary': benchmark.summarize([crec[n] for n in ns])['all'],
        'tolerance': _tolerance([brec[n] for n in ns], [crec[n] for n in ns]),
    }
  return {
      'base': str(base_dir), 'candidate': str(cand_dir),
      'missing_in_candidate': [n for n in brec if n not in crec],
      'missing_in_base': [n for n in crec if n not in brec],
      'complete': set(brec) == set(crec),
      # Seeds (coordinates, object IDs) are always checked; input hashes when both runs
      # record them.
      'inputs_sha256_checked': inputs_checked,
      # The verdict: tolerance met on all seeds, with every seed present and the
      # inputs verified. A partial or unverifiable comparison never passes.
      'passes_tolerance': (set(brec) == set(crec) and inputs_checked
                           and bool(overall['all']['tolerance'])
                           and overall['all']['tolerance']['within_tolerance']),
      'per_seed': per_seed,
      'seeds_flagged_for_review': [r['name'] for r in per_seed if r['flag_for_review']],
      'seeds_outside_seed_rules': [r['name'] for r in per_seed
                                   if not r['seed_tolerance']['within_seed_rules']],
      'overall': overall,
      'thresholds': {'equivalence_min_iou': EQUIV_MIN_IOU,
                     'equivalence_max_step_change': EQUIV_MAX_STEP_CHANGE,
                     'tolerance_status': TOLERANCE_STATUS},
  }
