"""Step-waste instrumentation: read-only, internally consistent, correctly classified."""

import collections
import unittest

import numpy as np

from liconn_v2 import data
from liconn_v2 import reference
from liconn_v2 import waste


class ClassifyTest(unittest.TestCase):

  def test_step_classes(self):
    a = {
        'new_fg':    np.array([100, 0, 0,  0,   0, 0]),
        'new_bg':    np.array([0,   0, 50, 0,   0, 0]),
        'gained':    np.array([0,   0, 0,  10,  0, 0]),
        'lost':      np.array([0,   0, 0,  0,   0, 400]),
        'new_cells': np.array([2,   0, 0,  0,   1, 0]),
        'final_owner_voxels': np.array([5, 0, 3, 0, 0, 1]),
        'established_owner_voxels': np.array([5, 0, 0, 2, 0, 1]),
    }
    c = waste.classify(a)
    np.testing.assert_array_equal(c['idle'],      [0, 1, 0, 0, 0, 0])
    # As specified: not idle, new_voxels + flips < 359, no new cells.
    np.testing.assert_array_equal(c['low_yield'], [0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(c['no_final_decision'], [0, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(c['mask_idle'], [0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(c['mask_low_yield'], [0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(c['enclosed'],  [0, 1, 0, 1, 1, 1])
    np.testing.assert_array_equal(c['no_established_decision'], [0, 1, 1, 0, 1, 0])

  def test_pooled_shares_are_step_weighted(self):
    base = dict.fromkeys(waste.CLASSES, 0)
    p = waste.pooled([{**base, 'steps': 10, 'idle': 1, 'mask_idle': 5, 'low_yield': 0},
                      {**base, 'steps': 30, 'idle': 3, 'mask_idle': 5, 'low_yield': 6}])
    self.assertEqual(p['steps'], 40)
    self.assertEqual(p['mask_idle_share'], 10 / 40)
    self.assertEqual(p['low_yield_share'], 6 / 40)


class _FakePolicy:
  """Mimics FaceMaxMovementPolicy's queue bookkeeping for new_cells."""

  def __init__(self):
    self.scored_coords = collections.deque()
    self.done_rounded_coords = set()

  def quantize_pos(self, pos):
    return tuple((np.array(pos) + 4) // 8)

  def update(self, prob_map, position):
    self.done_rounded_coords.add(self.quantize_pos(position))
    for c in prob_map:  # here: the candidates to append
      self.scored_coords.append((1.0, c))


class _FakeCanvas:

  def __init__(self, shape=(40, 40, 40)):
    self.seed = np.full(shape, np.nan, np.float32)
    self.movement_policy = _FakePolicy()
    self.writes = []

  def update_at(self, pos):
    lo = np.array(pos) - 16
    sel = tuple(slice(a, a + 33) for a in lo)
    self.seed[sel] = self.writes.pop(0)
    return self.seed[sel]


class RecorderSyntheticTest(unittest.TestCase):

  def test_counts_on_two_steps(self):
    cv = _FakeCanvas()
    rec = waste.StepRecorder()
    rec.attach(cv)
    pos = (20, 20, 20)
    cv.seed[pos] = 3.0  # pre-set by init_seed
    first = np.full((33, 33, 33), -2.0, np.float32)
    first[:10] = 2.0                      # 10*33*33 foreground
    second = first.copy()
    second[:5] = -2.0                     # 5*33*33 lost
    cv.writes = [first, second]
    cv.update_at(pos)
    cv.movement_policy.update([(20, 20, 28), (20, 21, 28), (20, 20, 20)], pos)
    cv.update_at(pos)
    cv.movement_policy.update([(20, 20, 28), (28, 20, 20)], pos)
    a = rec.arrays()
    face = 33 * 33
    np.testing.assert_array_equal(a['new_fg'], [10 * face, 0])
    np.testing.assert_array_equal(a['new_bg'], [23 * face, 0])  # seed voxel counted as new
    np.testing.assert_array_equal(a['lost'], [0, 5 * face])
    np.testing.assert_array_equal(a['gained'], [0, 0])
    # step 0: (20,20,28) and (20,21,28) share a cell, (20,20,20) is done -> 1 new cell.
    # step 1: (20,20,28) is still queued -> not new; (28,20,20) is new -> 1.
    np.testing.assert_array_equal(a['new_cells'], [1, 1])
    # As specified (last writer, counted if that write established the decision):
    # step 1 wrote every voxel last but established only the 5 lost rows.
    np.testing.assert_array_equal(a['final_owner_voxels'], [0, 5 * face])
    # Additional definition (last establishing step): every voxel has an owner.
    np.testing.assert_array_equal(a['established_owner_voxels'], [28 * face, 5 * face])
    self.assertEqual(int(a['established_owner_voxels'].sum()), 33 * face)
    s = waste.seed_summary(a)
    self.assertEqual((s['nonzero_flip_steps_after_first'], s['steps_after_first']), (1, 1))
    self.assertEqual(s['lost_quartiles'][1], 2.5 * face)


class InstrumentedNotebookTraceTest(unittest.TestCase):
  """The real notebook trace with recorder attached (~2.5 min)."""

  @classmethod
  def setUpClass(cls):
    raw = reference.normalize(data.load_raw_xyz())
    cls.rec = waste.StepRecorder()
    with reference.Tracer(data.checkpoint_path()) as tracer:
      cls.trace = tracer.trace(raw, reference.SEED_XYZ[::-1], instrument=cls.rec.attach)
    cls.a = cls.rec.arrays()

  def test_instrumentation_leaves_trace_bitwise_unchanged(self):
    self.assertEqual(reference.fingerprint(self.trace), reference.load_expected())

  def test_step_invariants(self):
    predicted = int((~np.isnan(self.trace.seed)).sum())
    self.assertEqual(len(self.a['new_fg']), self.trace.iterations)
    self.assertEqual(int((self.a['new_fg'] + self.a['new_bg']).sum()), predicted)
    self.assertEqual(int(self.a['established_owner_voxels'].sum()), predicted)
    self.assertLessEqual(int(self.a['final_owner_voxels'].sum()), predicted)
    self.assertEqual([tuple(p) for p in self.a['positions'].tolist()], self.trace.history)
    # Pre-step features: the queue score is the entry FFN actually popped, every step.
    self.assertTrue(self.a['feat_queue_match'].all())
    self.assertEqual(list(self.a['feat_step_index']), list(range(self.trace.iterations)))
    # Final decisions owned by steps equal the final mask.
    final_fg = int((self.trace.seed >= waste.T).sum())
    owned_fg = int((self.trace.seed[self.rec.owner >= 0] >= waste.T).sum())
    self.assertEqual(owned_fg, final_fg)


if __name__ == '__main__':
  unittest.main()
