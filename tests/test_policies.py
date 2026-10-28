"""Skip/postpone movement policies on a synthetic canvas (no network needed)."""

import unittest

import numpy as np

from liconn_v2 import policies


class _Canvas:
  """What the policies read from FFN's Canvas: the seed array and is_valid_pos."""

  def __init__(self):
    self.seed = np.full((80, 80, 80), np.nan, np.float32)
    self.invalid = set()

  def is_valid_pos(self, pos):
    return tuple(pos) not in self.invalid

  def fill(self, pos, value=2.0):
    """Predict the whole 33^3 box around pos."""
    z, y, x = pos
    self.seed[z - 16:z + 17, y - 16:y + 17, x - 16:x + 17] = value


def _policy(mode, canvas, start=(40, 40, 40)):
  cls = policies.make_policy_class()
  p = cls(canvas, mode, deltas=(8, 8, 8), score_threshold=0.4)
  p.reset_state(start)
  return p


class PolicyTest(unittest.TestCase):

  def test_skip_enclosed_marks_cell_done(self):
    cv = _Canvas()
    cv.fill((40, 40, 40))                       # box at the start position: fully predicted
    p = _policy('skip_enclosed', cv)
    p.append((1.0, (40, 40, 40)))
    p.append((0.9, (41, 40, 40)))               # same 8-voxel cell, partly unpredicted
    p.append((0.8, (40, 40, 56)))               # elsewhere, unpredicted
    self.assertEqual(next(p), (40, 40, 56))
    with self.assertRaises(StopIteration):
      next(p)
    self.assertEqual(p.stats['main_skipped_enclosed'], 1)
    self.assertEqual(p.stats['main_dropped_done'], 1)    # the same-cell duplicate
    self.assertEqual(policies.check_stats(p.stats, steps=1), [])

  def test_postpone_moves_nearly_enclosed_to_the_back(self):
    cv = _Canvas()
    cv.fill((40, 40, 40))
    cv.seed[24, 24, 24] = np.nan                # 1 unpredicted voxel: nearly enclosed
    p = _policy('postpone', cv)
    p.append((1.0, (40, 40, 40)))
    p.append((0.5, (40, 40, 64)))
    self.assertEqual(next(p), (40, 40, 64))     # the open box goes first
    self.assertEqual(len(p), 1)                 # the postponed one still counts
    self.assertEqual(next(p), (40, 40, 40))     # then the postponed one
    with self.assertRaises(StopIteration):
      next(p)
    self.assertEqual((p.stats['postponed'], p.stats['postponed_evaluated']), (1, 1))
    self.assertEqual(policies.check_stats(p.stats, steps=2), [])

  def test_postpone_drop_enclosed_drops_and_marks_done(self):
    cv = _Canvas()
    cv.fill((40, 40, 40))
    cv.seed[24, 24, 24] = np.nan                # 1 unpredicted voxel: nearly enclosed
    p = _policy('postpone_drop_enclosed', cv)
    p.append((1.0, (40, 40, 40)))
    p.append((0.5, (40, 40, 64)))
    self.assertEqual(next(p), (40, 40, 64))     # (40,40,40) was postponed
    cv.seed[24, 24, 24] = 2.0                   # the open step's update encloses it
    with self.assertRaises(StopIteration):      # so it is dropped when it comes up
      next(p)
    self.assertEqual(p.stats['postponed'], 1)
    self.assertEqual(p.stats['postponed_dropped_enclosed'], 1)
    self.assertIn(p.quantize_pos((40, 40, 40)), p.done_rounded_coords)
    self.assertEqual(policies.check_stats(p.stats, steps=1), [])

  def test_invalid_and_done_follow_ffn(self):
    cv = _Canvas()
    cv.invalid.add((40, 40, 48))
    p = _policy('postpone', cv)
    p.append((1.0, (40, 40, 48)))               # fails is_valid_pos: dropped, cell not done
    p.append((0.9, (40, 40, 49)))               # same cell, valid, unpredicted: evaluated
    self.assertEqual(next(p), (40, 40, 49))
    self.assertEqual(p.stats['main_dropped_invalid'], 1)

  def test_reset_clears_postponed_queue(self):
    cv = _Canvas()
    cv.fill((40, 40, 40))
    p = _policy('postpone', cv)
    p.append((1.0, (40, 40, 40)))
    p.append((0.5, (40, 40, 64)))
    next(p)
    p.reset_state((40, 40, 40))
    self.assertEqual(len(p), 0)

  def test_check_stats_detects_inconsistencies(self):
    ok = {'main_popped': 3, 'postponed': 1, 'main_evaluated': 1, 'main_dropped_done': 1,
          'postponed_popped': 1, 'postponed_evaluated': 1}
    self.assertEqual(policies.check_stats(ok, steps=2), [])
    self.assertTrue(policies.check_stats(ok, steps=3))
    self.assertTrue(policies.check_stats({**ok, 'postponed_popped': 0, 'postponed_evaluated': 0}, steps=1))

  def test_unknown_policy_rejected(self):
    with self.assertRaises(ValueError):
      policies.PolicyFactory('teleport')


if __name__ == '__main__':
  unittest.main()
