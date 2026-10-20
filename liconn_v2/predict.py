"""Can low-yield FFN steps be recognised before they run? (docs/predictability_design.md)

Pure numpy: ROC AUC, a small L2-regularised logistic regression, and
leave-one-seed-out evaluation over per-step tables built by waste.StepRecorder.
"""

import numpy as np

from liconn_v2 import waste

TARGETS = ('low_yield', 'mask_low_yield')
# Nested feature sets, to measure what the canvas values add beyond box coverage
# and position in the trace.
FEATURE_SETS = {
    'coverage': ('unpredicted_fraction',),
    'coverage_and_position': ('unpredicted_fraction', 'overlap_count', 'step_index'),
    'all': waste.FEATURES,
}
OPERATING_POINTS = (0.10, 0.25, 0.50)


def auc(scores, labels) -> float | None:
  """ROC AUC (Mann-Whitney, ties count half). None if only one class is present."""
  scores = np.asarray(scores, float)
  labels = np.asarray(labels, bool)
  pos, neg = scores[labels], scores[~labels]
  if len(pos) == 0 or len(neg) == 0:
    return None
  order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
  allv = np.concatenate([pos, neg])[order]
  ranks = np.empty(len(allv))
  i = 0
  while i < len(allv):  # average ranks over ties
    j = i
    while j + 1 < len(allv) and allv[j + 1] == allv[i]:
      j += 1
    ranks[i:j + 1] = (i + j) / 2 + 1
    i = j + 1
  r = np.empty(len(allv))
  r[order] = ranks
  r_pos = r[:len(pos)].sum()
  return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_logistic(x, y, l2=1e-2, iters=50):
  """Newton's method for L2-regularised logistic regression; returns (w, b)."""
  x = np.asarray(x, float)
  y = np.asarray(y, float)
  n, d = x.shape
  xa = np.hstack([x, np.ones((n, 1))])
  w = np.zeros(d + 1)
  reg = np.full(d + 1, l2)
  reg[-1] = 0.0  # no penalty on the intercept
  for _ in range(iters):
    p = 1 / (1 + np.exp(-xa @ w))
    grad = xa.T @ (p - y) / n + reg * w
    h = (xa * (p * (1 - p))[:, None]).T @ xa / n + np.diag(reg)
    step = np.linalg.solve(h, grad)
    w -= step
    if np.abs(step).max() < 1e-10:
      break
  # numpy 2.0 with Apple's Accelerate can print spurious matmul warnings; make sure
  # the fit itself is sound.
  if not np.isfinite(w).all():
    raise FloatingPointError('logistic regression did not converge to finite weights')
  return w[:-1], w[-1]


def predict_logistic(model, x):
  w, b = model
  return 1 / (1 + np.exp(-(np.asarray(x, float) @ w + b)))


def reference_checks(name, canvas_sha256, history_sha256, reference_records) -> dict:
  """Checks against a reference run; a seed missing from the reference fails."""
  ref = reference_records.get(name)
  if ref is None:
    return {'reference_record_present': False}
  return {'reference_record_present': True,
          'canvas_matches_reference': canvas_sha256 == ref.get('canvas_sha256'),
          'history_matches_reference': history_sha256 == ref.get('history_sha256')}


def step_table(per_seed_arrays: dict) -> dict:
  """Stacks recorder arrays over seeds; drops each trace's first step."""
  cols = {k: [] for k in (*waste.FEATURES, *TARGETS, 'frontier', 'enclosed', 'seed', 'queue_match')}
  for name, a in per_seed_arrays.items():
    c = waste.classify(a)
    n = len(a['new_fg'])
    keep = np.arange(n) >= 1  # step 0 is the seed itself and cannot be skipped
    for k in waste.FEATURES:
      cols[k].append(np.asarray(a['feat_' + k], float)[keep])
    cols['low_yield'].append(c['low_yield'][keep])
    cols['mask_low_yield'].append(c['mask_low_yield'][keep])
    cols['enclosed'].append(c['enclosed'][keep])
    cols['frontier'].append((a['new_cells'] > 0)[keep])
    cols['queue_match'].append(np.asarray(a['feat_queue_match'], bool)[keep])
    cols['seed'].append(np.array([name] * int(keep.sum())))
  return {k: np.concatenate(v) if v else np.array([]) for k, v in cols.items()}


def _standardize(train, test):
  # Missing values (none expected after step 0) take the training mean.
  fill = np.nanmean(train, axis=0)
  train = np.where(np.isnan(train), fill, train)
  test = np.where(np.isnan(test), fill, test)
  mu = train.mean(axis=0)
  sd = train.std(axis=0)
  sd[sd == 0] = 1.0
  return (train - mu) / sd, (test - mu) / sd


def leave_one_seed_out(table, target, features=waste.FEATURES) -> np.ndarray:
  """Out-of-fold scores: each seed scored by a model fit on the other seeds only."""
  x = np.column_stack([table[k] for k in features])
  y = table[target]
  seeds = table['seed']
  oof = np.full(len(y), np.nan)
  for s in np.unique(seeds):
    test = seeds == s
    train = ~test
    if len(np.unique(y[train])) < 2:
      continue
    xtr, xte = _standardize(x[train], x[test])  # scaling fit on the training fold only
    oof[test] = predict_logistic(fit_logistic(xtr, y[train]), xte)
  return oof


def operating_points(scores, labels, frontier, rates=OPERATING_POINTS) -> list:
  """For each share of low-yield steps caught: what else would be flagged."""
  scores = np.asarray(scores, float)
  labels = np.asarray(labels, bool)
  frontier = np.asarray(frontier, bool)
  out = []
  pos_scores = np.sort(scores[labels])[::-1]
  for rate in rates:
    if len(pos_scores) == 0:
      break
    k = max(int(np.ceil(rate * len(pos_scores))), 1)
    thr = pos_scores[k - 1]
    flagged = scores >= thr
    useful = ~labels
    out.append({
        'target_caught': rate,
        'threshold': float(thr),
        'low_yield_caught': float(flagged[labels].mean()),
        'useful_flagged': float(flagged[useful].mean()) if useful.any() else 0.0,
        'frontier_flagged': float(flagged[useful & frontier].mean()) if (useful & frontier).any() else 0.0,
        'flagged_share_of_all_steps': float(flagged.mean()),
        'low_yield_share_of_flagged': float(labels[flagged].mean()) if flagged.any() else 0.0,
    })
  return out


def rule_summary(flags, labels, frontier) -> dict:
  """The same numbers for a fixed yes/no rule (e.g. 'box already fully predicted')."""
  flags = np.asarray(flags, bool)
  labels = np.asarray(labels, bool)
  frontier = np.asarray(frontier, bool)
  useful = ~labels
  return {
      'low_yield_caught': float(flags[labels].mean()) if labels.any() else 0.0,
      'useful_flagged': float(flags[useful].mean()) if useful.any() else 0.0,
      'frontier_flagged': float(flags[useful & frontier].mean()) if (useful & frontier).any() else 0.0,
      'flagged_share_of_all_steps': float(flags.mean()) if len(flags) else 0.0,
  }


def subset(table, keep) -> dict:
  return {k: v[keep] for k, v in table.items()}


def near_enclosed(table) -> np.ndarray:
  """Steps whose box has fewer than 359 not-yet-predicted voxels (known in advance).

  Under the owner-approved definition a step with >= 359 new voxels is never
  low-yield, so the predictable part outside this subset holds by construction.
  """
  unpredicted = np.rint(table['unpredicted_fraction'] * waste.BOX_VOXELS)
  return unpredicted < waste.LOW_YIELD_VOXELS


def analyse(table, _nested=False) -> dict:
  """Feature AUCs, leave-one-seed-out model results and baselines for each target."""
  result = {'steps': int(len(table['seed'])), 'seeds': sorted(set(table['seed'].tolist())),
            'queue_score_matched_popped_entry': bool(table['queue_match'].all())}
  if not _nested:
    sub_table = subset(table, near_enclosed(table))
    result['near_enclosed_subset'] = analyse(sub_table, _nested=True)
    result['near_enclosed_subset']['share_of_steps'] = (
        float(near_enclosed(table).mean()) if len(table['seed']) else None)
  for target in TARGETS:
    y = table[target]
    r = {'prevalence': float(y.mean()) if len(y) else None}
    r['feature_auc'] = {}
    for k in waste.FEATURES:
      ok = ~np.isnan(table[k])
      r['feature_auc'][k] = auc(table[k][ok], y[ok])
    # AUC below 0.5 means the feature works in the opposite direction.
    r['feature_auc_oriented'] = {k: (None if v is None else max(v, 1 - v))
                                 for k, v in r['feature_auc'].items()}
    seeds = sorted(set(table['seed'].tolist()))
    if len(seeds) >= 2 and len(np.unique(y)) == 2:
      oof = leave_one_seed_out(table, target)
      ok = ~np.isnan(oof)
      r['model_oof_auc'] = auc(oof[ok], y[ok])
      per_seed = {s: auc(oof[table['seed'] == s], y[table['seed'] == s]) for s in seeds}
      r['model_per_seed_auc'] = per_seed
      vals = [v for v in per_seed.values() if v is not None]
      r['model_per_seed_auc_quartiles'] = ([float(v) for v in np.percentile(vals, [25, 50, 75])]
                                           if vals else None)
      r['model_operating_points'] = operating_points(oof[ok], y[ok], table['frontier'][ok])
      r['feature_set_oof_auc'] = {}
      for name, feats in FEATURE_SETS.items():
        s_oof = leave_one_seed_out(table, target, feats)
        s_ok = ~np.isnan(s_oof)
        s_seed = {s: auc(s_oof[table['seed'] == s], y[table['seed'] == s]) for s in seeds}
        s_vals = [v for v in s_seed.values() if v is not None]
        r['feature_set_oof_auc'][name] = {
            'pooled': auc(s_oof[s_ok], y[s_ok]),
            'per_seed_median': float(np.median(s_vals)) if s_vals else None,
        }
    else:
      r['model_oof_auc'] = None
      r['note'] = 'one seed only: no leave-one-seed-out evaluation possible'
    r['baseline_enclosed_rule'] = rule_summary(table['enclosed'], y, table['frontier'])
    result[target] = r
  return result
