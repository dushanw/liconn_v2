"""Per-step waste in FFN traces (docs/waste_analysis_design.md).

Read-only wrappers on one Canvas record what each network evaluation changed.
They copy canvas boxes and queue contents; they never write to the canvas, the
queue or the policy.
"""

import itertools

import numpy as np
from scipy.special import logit

from liconn_v2 import reference

T = np.float32(logit(reference.SEGMENT_THRESHOLD))
HALF = np.array(reference.FOV_SIZE) // 2
BOX_VOXELS = int(np.prod(reference.FOV_SIZE))
LOW_YIELD_VOXELS = BOX_VOXELS // 100  # 359
FIELDS = ('new_fg', 'new_bg', 'gained', 'lost', 'new_cells', 'mean_abs_dlogit')
# Pre-step features (docs/predictability_design.md): computed from the canvas and the
# queue before the step's evaluation, so a policy could use them.
FEATURES = ('unpredicted_fraction', 'fg_fraction', 'uncertain_fraction', 'mean_margin',
            'center_value', 'queue_score', 'overlap_count', 'step_index')
UNCERTAIN_BAND = 1.0  # |value - T| < 1 in logits, about p = 0.35-0.80
OVERLAP_DIST = 2 * int(HALF[0])  # boxes overlap if every |d| <= 32
CLASSES = ('idle', 'low_yield', 'no_final_decision',
           'mask_idle', 'mask_low_yield', 'enclosed', 'no_established_decision')


class StepRecorder:
  """Attach with `recorder.attach(cv)` before `cv.segment_at(...)`."""

  def __init__(self):
    self.rows = []
    self.positions = []
    self.owner = None          # last step that established each voxel's decision
    self.last_writer = None    # last step that wrote each voxel
    self.last_write_established = None  # did that last write establish the decision?
    self._seed_pos = None
    self.features = []
    self._queue_after_update = []  # queue contents right after the last policy update

  def attach(self, cv):
    self.owner = np.full(cv.seed.shape, -1, np.int32)
    self.last_writer = np.full(cv.seed.shape, -1, np.int32)
    self.last_write_established = np.zeros(cv.seed.shape, bool)
    orig_update_at = cv.update_at
    policy = cv.movement_policy
    orig_policy_update = policy.update

    def update_at(pos):
      if self._seed_pos is None:
        self._seed_pos = tuple(int(v) for v in pos)  # the first update is at the seed
      lo = np.array(pos) - HALF
      sel = tuple(slice(a, a + n) for a, n in zip(lo, reference.FOV_SIZE))
      old = np.array(cv.seed[sel])
      self.features.append(self._pre_step_features(old, pos, policy))
      if not self.rows:
        # init_seed pre-set the seed voxel; count it as not yet predicted.
        old[tuple(HALF)] = np.nan
      out = orig_update_at(pos)
      new = np.array(cv.seed[sel])
      was_new = np.isnan(old)
      prev = ~was_new
      fg_old, fg_new = old >= T, new >= T
      flipped = prev & (fg_old != fg_new)
      step = len(self.rows)
      self.owner[sel][was_new | flipped] = step
      self.last_writer[sel] = step
      self.last_write_established[sel] = was_new | flipped
      self.rows.append({
          'new_fg': int((was_new & fg_new).sum()),
          'new_bg': int((was_new & ~fg_new).sum()),
          'gained': int((flipped & fg_new).sum()),
          'lost': int((flipped & ~fg_new).sum()),
          'new_cells': 0,
          'mean_abs_dlogit': float(np.abs(new[prev] - old[prev]).mean()) if prev.any() else 0.0,
      })
      self.positions.append(tuple(int(v) for v in pos))
      return out

    def policy_update(prob_map, position):
      queued_before = {policy.quantize_pos(tuple(c)) for _, c in policy.scored_coords}
      n_before = len(policy.scored_coords)
      out = orig_policy_update(prob_map, position)
      appended = itertools.islice(policy.scored_coords, n_before, None)
      cells = {policy.quantize_pos(tuple(c)) for _, c in appended}
      self.rows[-1]['new_cells'] = len(cells - set(policy.done_rounded_coords) - queued_before)
      self._queue_after_update = list(policy.scored_coords)
      return out

    cv.update_at = update_at
    policy.update = policy_update

  def _pre_step_features(self, old, pos, policy) -> dict:
    """Features of the step about to run, from state that exists before it runs."""
    step = len(self.rows)
    predicted = ~np.isnan(old)
    n_pred = int(predicted.sum())
    vals = old[predicted]
    margin = np.abs(vals - T)
    # Between the last policy update and this step FFN only pops from the queue's
    # front; the last popped entry is the one being evaluated now.
    popped = len(self._queue_after_update) - len(policy.scored_coords)
    queue_score, queue_match = np.nan, step == 0
    if step > 0 and popped >= 1:
      score, coord = self._queue_after_update[popped - 1]
      queue_score = float(score)
      queue_match = tuple(int(v) for v in coord) == tuple(int(v) for v in pos)
    prev = np.array(self.positions, np.int64).reshape(-1, 3)
    overlap = int((np.abs(prev - np.array(pos)).max(axis=1) <= OVERLAP_DIST).sum()) if len(prev) else 0
    return {
        'unpredicted_fraction': 1.0 - n_pred / old.size,
        'fg_fraction': float((vals >= T).mean()) if n_pred else np.nan,
        'uncertain_fraction': float((margin < UNCERTAIN_BAND).mean()) if n_pred else np.nan,
        'mean_margin': float(margin.mean()) if n_pred else np.nan,
        'center_value': float(old[tuple(HALF)]),
        'queue_score': queue_score,
        'overlap_count': overlap,
        'step_index': step,
        'queue_match': queue_match,
    }

  def arrays(self) -> dict:
    out = {k: np.array([r[k] for r in self.rows]) for k in FIELDS}
    # As specified: the last writer, counted only if that write established the decision.
    out['final_owner_voxels'] = np.bincount(self.last_writer[self.last_write_established],
                                            minlength=len(self.rows))
    # Additional (Cursor method check): the last step that established the decision,
    # so that every predicted voxel has an owner.
    out['established_owner_voxels'] = np.bincount(self.owner[self.owner >= 0],
                                                  minlength=len(self.rows))
    out['positions'] = np.array(self.positions, np.int32).reshape(-1, 3)
    for k in (*FEATURES, 'queue_match'):
      out['feat_' + k] = np.array([f[k] for f in self.features])
    return out


def classify(a: dict) -> dict:
  """Boolean step classes from per-step arrays."""
  new_voxels = a['new_fg'] + a['new_bg']
  flips = a['gained'] + a['lost']
  no_cells = a['new_cells'] == 0
  mask_idle = (a['new_fg'] == 0) & (flips == 0) & no_cells
  idle = (new_voxels == 0) & (flips == 0) & no_cells
  return {
      # As specified in the approved design.
      'idle': idle,
      'low_yield': ~idle & (new_voxels + flips < LOW_YIELD_VOXELS) & no_cells,
      'no_final_decision': a['final_owner_voxels'] == 0,
      # Additional classes (Cursor method check); not replacements.
      'mask_idle': mask_idle,
      'mask_low_yield': ~mask_idle & (a['new_fg'] + flips < LOW_YIELD_VOXELS) & no_cells,
      'enclosed': new_voxels == 0,
      'no_established_decision': a['established_owner_voxels'] == 0,
  }


def _quartiles(x):
  return [float(v) for v in np.percentile(x, [25, 50, 75])] if len(x) else None


def seed_summary(a: dict) -> dict:
  c = classify(a)
  n = len(a['new_fg'])
  q = max(n // 4, 1)
  out = {'steps': n}
  for k, v in c.items():
    out[k] = int(v.sum())
    out[k + '_share'] = float(v.mean()) if n else 0.0
  for k in ('idle', 'low_yield', 'mask_idle', 'mask_low_yield'):
    out[k + '_share_first_quarter'] = float(c[k][:q].mean()) if n else 0.0
    out[k + '_share_middle_half'] = float(c[k][q:n - q].mean()) if n > 2 * q else None
    out[k + '_share_last_quarter'] = float(c[k][-q:].mean()) if n else 0.0
  flips = a['gained'] + a['lost']
  # Step 0 has no previously predicted voxels, so it cannot flip any.
  out['nonzero_flip_steps_after_first'] = int((flips[1:] > 0).sum())
  out['steps_after_first'] = max(n - 1, 0)
  for k, v in (('new_fg', a['new_fg']), ('new_bg', a['new_bg']), ('flips', flips),
               ('gained', a['gained']), ('lost', a['lost']),
               ('new_cells', a['new_cells']), ('mean_abs_dlogit', a['mean_abs_dlogit'])):
    out[k + '_quartiles'] = _quartiles(v)
  out['mean_abs_dlogit_mask_idle'] = (float(a['mean_abs_dlogit'][c['mask_idle']].mean())
                                      if c['mask_idle'].any() else None)
  return out


def pooled(per_seed: list) -> dict | None:
  """Step-weighted totals over seeds (shares are of all steps)."""
  if not per_seed:
    return None
  steps = sum(s['steps'] for s in per_seed)
  out = {'seeds': len(per_seed), 'steps': steps}
  for k in CLASSES:
    total = sum(s[k] for s in per_seed)
    out[k] = total
    out[k + '_share'] = total / steps if steps else 0.0
  return out
