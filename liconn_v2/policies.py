"""Movement policies that skip or postpone nearly enclosed FFN steps.

See docs/postpone_design.md. Each policy subclasses FFN's FaceMaxMovementPolicy and
keeps its proposals and pop-time checks; only the order and selection of the
candidates that pass those checks change. The 'default' policy is FFN's own.
"""

import collections

import numpy as np

from liconn_v2 import reference

NEAR_ENCLOSED_VOXELS = int(np.prod(reference.FOV_SIZE)) // 100  # 359, as in waste.py
POLICIES = ('default', 'skip_enclosed', 'postpone', 'postpone_drop_enclosed')
_HALF = np.array(reference.FOV_SIZE) // 2


def unpredicted_voxels(canvas, pos) -> int:
  """Not-yet-predicted (NaN) voxels in the FOV box centred at pos (z, y, x)."""
  lo = np.array(pos) - _HALF
  sel = tuple(slice(a, a + n) for a, n in zip(lo, reference.FOV_SIZE))
  return int(np.isnan(canvas.seed[sel]).sum())


def make_policy_class():
  """Builds the policy class lazily, so importing this module does not import FFN."""
  from ffn.inference import movement  # pylint: disable=g-import-not-at-top

  class GatedFaceMaxPolicy(movement.FaceMaxMovementPolicy):
    """FaceMaxMovementPolicy plus a skip/postpone rule for nearly enclosed boxes."""

    def __init__(self, canvas, mode, deltas, score_threshold):
      if mode not in POLICIES[1:]:
        raise ValueError(f'unknown policy {mode!r}')
      super().__init__(canvas, deltas=deltas, score_threshold=score_threshold)
      self.mode = mode
      self.deferred = collections.deque()
      self._deferred_coords = set()
      self.stats = collections.Counter()

    def reset_state(self, start_pos):
      super().reset_state(start_pos)
      self.deferred = collections.deque()
      self._deferred_coords = set()

    def __len__(self):
      return len(self.scored_coords) + len(self.deferred)

    def _mark_done(self, coord):
      self.done_rounded_coords.add(self.quantize_pos(coord))

    def __next__(self):
      # Counters per queue (main / postponed); see check_stats() for how they add up.
      while True:
        if self.scored_coords:
          _, coord = self.scored_coords.popleft()
          q = 'main'
        elif self.deferred:
          coord = self.deferred.popleft()  # postponed candidates in FIFO order
          q = 'postponed'
        else:
          raise StopIteration()
        self.stats[q + '_popped'] += 1
        coord = tuple(int(v) for v in coord)
        # FFN's own pop-time checks, unchanged: FFN marks a cell done only when a
        # step there is evaluated (in update()), never on a rejected pop.
        if self.quantize_pos(coord) in self.done_rounded_coords:
          self.stats[q + '_dropped_done'] += 1
          continue
        if not self.canvas.is_valid_pos(coord):
          self.stats[q + '_dropped_invalid'] += 1
          continue
        u = unpredicted_voxels(self.canvas, coord)
        if q == 'main' and self.mode == 'skip_enclosed' and u == 0:
          self.stats['main_skipped_enclosed'] += 1
          self._mark_done(coord)
          continue
        if (q == 'main' and self.mode in ('postpone', 'postpone_drop_enclosed')
            and u < NEAR_ENCLOSED_VOXELS and coord not in self._deferred_coords):
          # Not marked done: another candidate in the same cell may still run first.
          self.stats['postponed'] += 1
          self._deferred_coords.add(coord)
          self.deferred.append(coord)
          continue
        if q == 'postponed' and self.mode == 'postpone_drop_enclosed' and u == 0:
          self.stats['postponed_dropped_enclosed'] += 1
          self._mark_done(coord)
          continue
        self.stats[q + '_evaluated'] += 1
        if q == 'postponed' and u == 0:
          self.stats['postponed_evaluated_enclosed'] += 1
        return coord

  return GatedFaceMaxPolicy


STAT_KEYS = ('main_popped', 'main_dropped_done', 'main_dropped_invalid', 'main_skipped_enclosed',
             'postponed', 'main_evaluated', 'postponed_popped', 'postponed_dropped_done',
             'postponed_dropped_invalid', 'postponed_dropped_enclosed', 'postponed_evaluated',
             'postponed_evaluated_enclosed')


def check_stats(stats: dict, steps: int) -> list:
  """Returns the bookkeeping identities that fail (empty when consistent)."""
  s = {k: int(stats.get(k, 0)) for k in STAT_KEYS}
  problems = []
  if s['main_popped'] != (s['main_dropped_done'] + s['main_dropped_invalid'] + s['main_skipped_enclosed']
                          + s['postponed'] + s['main_evaluated']):
    problems.append('main queue counts do not add up')
  if s['postponed_popped'] != (s['postponed_dropped_done'] + s['postponed_dropped_invalid']
                               + s['postponed_dropped_enclosed'] + s['postponed_evaluated']):
    problems.append('postponed queue counts do not add up')
  if s['postponed'] != s['postponed_popped']:
    problems.append('not every postponed candidate was popped again')
  if steps != s['main_evaluated'] + s['postponed_evaluated']:
    problems.append('evaluations do not equal the trace steps')
  return problems


class PolicyFactory:
  """Callable for Canvas(movement_policy_fn=...); keeps the last policy for its stats."""

  def __init__(self, mode):
    if mode not in POLICIES:
      raise ValueError(f'unknown policy {mode!r}; known: {POLICIES}')
    self.mode = mode
    self.last = None

  def __call__(self, canvas):
    cls = make_policy_class()
    # Same arguments as Canvas's default FaceMaxMovementPolicy: model deltas in z, y, x
    # order and the move threshold (already converted to a logit by Canvas).
    self.last = cls(canvas, self.mode, deltas=reference.DELTAS[::-1],
                    score_threshold=canvas.options.move_threshold)
    return self.last


def run_policy(seeds: list, raw_zyx: np.ndarray, gt_zyx: np.ndarray, ckpt_path: str, policy: str,
               on_record=None):
  """Like benchmark.run, with the movement policy replaced. Returns (records, masks).

  The policy is installed through Tracer.trace's instrumentation hook, after the
  Canvas is created and before segment_at; 'default' leaves FFN's policy in place.
  """
  from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
  factory = PolicyFactory(policy)

  def install(canvas):
    if policy != 'default':
      canvas.movement_policy = factory(canvas)

  records, masks = [], {}
  with reference.Tracer(ckpt_path) as tracer:
    gflop = reference.gflop_per_eval(tracer.params)
    for s in seeds:
      factory.last = None
      trace = tracer.trace(raw_zyx, s['seed_zyx'], instrument=install)
      mask = benchmark.object_mask(trace.seed)
      rec = {'name': s['name'], 'class': s['class'], 'seed_zyx': list(s['seed_zyx'])}
      rec.update(benchmark.score(mask, gt_zyx, s['seed_zyx']))
      rec.update(benchmark.cost(trace.iterations, rec['correct_voxels'], gflop))
      rec.update({'trace_seconds': trace.trace_seconds, 'gflop_per_eval': gflop,
                  'number_format': 'float32', 'backend': trace.backend,
                  'device_kind': trace.device_kind,
                  'history_sha256': reference.history_sha256(trace.history),
                  'canvas_sha256': reference.canvas_sha256(trace.seed),
                  'policy': policy})
      if factory.last is not None:
        stats = {k: int(factory.last.stats.get(k, 0)) for k in STAT_KEYS}
        rec['policy_stats'] = stats
        rec['policy_stats_problems'] = check_stats(stats, trace.iterations)
      records.append(rec)
      masks[s['name']] = mask
      if on_record is not None:
        on_record(rec)
  return records, masks
