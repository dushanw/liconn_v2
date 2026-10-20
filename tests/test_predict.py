"""Pre-step features and the predictability analysis (no GPU needed)."""

import collections
import unittest
import unittest.mock

import numpy as np

from liconn_v2 import predict
from liconn_v2 import waste


class AucTest(unittest.TestCase):

  def test_hand_computed(self):
    # Pairs (pos, neg): (0.35, 0.1) yes, (0.35, 0.4) no, (0.8, 0.1) yes, (0.8, 0.4) yes -> 3/4.
    self.assertAlmostEqual(predict.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
    self.assertAlmostEqual(predict.auc([1, 1], [0, 1]), 0.5)  # a tie counts half
    self.assertIsNone(predict.auc([0.2, 0.3], [1, 1]))


class LogisticTest(unittest.TestCase):

  def test_separable_data(self):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    y = (x[:, 0] - 0.5 * x[:, 1] > 0)
    model = predict.fit_logistic(x, y)
    self.assertGreater(model[0][0], 0)
    self.assertLess(model[0][1], 0)
    self.assertGreater(predict.auc(predict.predict_logistic(model, x), y), 0.98)

  def test_leave_one_seed_out_never_trains_on_the_held_out_seed(self):
    rng = np.random.default_rng(1)
    n = 90
    table = {k: rng.normal(size=n) for k in waste.FEATURES}
    table['low_yield'] = rng.random(n) < 0.3
    table['seed'] = np.array(['a'] * 30 + ['b'] * 30 + ['c'] * 30)
    seen = []
    real_fit = predict.fit_logistic
    def spy(x, y, **kw):
      seen.append(len(y))
      return real_fit(x, y, **kw)
    with unittest.mock.patch.object(predict, 'fit_logistic', spy):
      oof = predict.leave_one_seed_out(table, 'low_yield')
    self.assertEqual(seen, [60, 60, 60])  # each fold trains on the two other seeds
    self.assertFalse(np.isnan(oof).any())

  def test_reference_checks_fail_for_missing_seed(self):
    ref = {'a': {'canvas_sha256': 'c', 'history_sha256': 'h'}}
    self.assertEqual(predict.reference_checks('a', 'c', 'h', ref),
                     {'reference_record_present': True, 'canvas_matches_reference': True,
                      'history_matches_reference': True})
    self.assertFalse(predict.reference_checks('a', 'x', 'h', ref)['canvas_matches_reference'])
    self.assertEqual(predict.reference_checks('b', 'c', 'h', ref), {'reference_record_present': False})

  def test_feature_sets_are_nested(self):
    sets = list(predict.FEATURE_SETS.values())
    for small, large in zip(sets, sets[1:]):
      self.assertTrue(set(small) < set(large))
    self.assertEqual(tuple(predict.FEATURE_SETS['all']), waste.FEATURES)

  def test_operating_points(self):
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    labels = np.array([1, 0, 1, 0, 1, 0], bool)
    frontier = np.array([0, 1, 0, 0, 0, 1], bool)
    pts = predict.operating_points(scores, labels, frontier, rates=(0.5,))
    # Catching 2 of 3 positives needs threshold 0.7: flags 0.9, 0.8, 0.7.
    self.assertEqual(pts[0]['threshold'], 0.7)
    self.assertAlmostEqual(pts[0]['low_yield_caught'], 2 / 3)
    self.assertAlmostEqual(pts[0]['useful_flagged'], 1 / 3)
    self.assertAlmostEqual(pts[0]['frontier_flagged'], 1 / 2)
    self.assertAlmostEqual(pts[0]['flagged_share_of_all_steps'], 3 / 6)


class _Policy:

  def __init__(self):
    self.scored_coords = collections.deque()
    self.done_rounded_coords = set()

  def quantize_pos(self, pos):
    return tuple((np.array(pos) + 4) // 8)

  def update(self, candidates, position):
    self.done_rounded_coords.add(self.quantize_pos(position))
    for score, c in candidates:
      self.scored_coords.append((score, c))


class _Canvas:

  def __init__(self):
    self.seed = np.full((60, 60, 60), np.nan, np.float32)
    self.movement_policy = _Policy()
    self.writes = []

  def update_at(self, pos):
    lo = np.array(pos) - 16
    sel = tuple(slice(a, a + 33) for a in lo)
    self.seed[sel] = self.writes.pop(0)
    return self.seed[sel]


class PreStepFeatureTest(unittest.TestCase):

  def test_features_use_only_pre_step_state(self):
    cv = _Canvas()
    rec = waste.StepRecorder()
    rec.attach(cv)
    first = np.full((33, 33, 33), -3.0, np.float32)
    first[:, :, :20] = 3.0                       # 20 of 33 x-columns are object
    cv.writes = [first, np.zeros((33, 33, 33), np.float32)]
    cv.update_at((20, 20, 20))
    cv.movement_policy.update([(1.5, (20, 20, 28)), (0.9, (20, 28, 20))], (20, 20, 20))
    # FFN pops from the front before the next update: the evaluated entry is (1.5, ...).
    cv.movement_policy.scored_coords.popleft()
    cv.update_at((20, 20, 28))
    f = rec.arrays()
    self.assertEqual(list(f['feat_step_index']), [0, 1])
    self.assertEqual(f['feat_overlap_count'][1], 1)
    self.assertEqual(f['feat_queue_score'][1], 1.5)
    self.assertTrue(f['feat_queue_match'][1])
    # Box at x 12..44 overlaps the first box's x 4..36: x 12..36 predicted (25 of 33).
    self.assertAlmostEqual(f['feat_unpredicted_fraction'][1], 8 / 33)
    # Predicted part: x 12..23 object (12 columns), x 24..36 background (13).
    self.assertAlmostEqual(f['feat_fg_fraction'][1], 12 / 25)
    self.assertEqual(f['feat_center_value'][1], -3.0)  # x = 28 was background
    t = float(waste.T)
    self.assertAlmostEqual(f['feat_mean_margin'][1], (12 * (3 - t) + 13 * (3 + t)) / 25, places=5)
    self.assertAlmostEqual(f['feat_uncertain_fraction'][1], 0.0)

  def test_near_enclosed_subset(self):
    t = {'unpredicted_fraction': np.array([0.0, 358 / waste.BOX_VOXELS, 359 / waste.BOX_VOXELS, 0.5])}
    self.assertEqual(list(predict.near_enclosed(t)), [True, True, False, False])

  def test_step_table_drops_first_steps_and_marks_frontier(self):
    n = 4
    a = {k: np.zeros(n, int) for k in waste.FIELDS}
    a['new_fg'] = np.array([100, 0, 500, 0])
    a['new_cells'] = np.array([2, 0, 1, 0])
    a['mean_abs_dlogit'] = np.zeros(n)
    a['final_owner_voxels'] = np.ones(n, int)
    a['established_owner_voxels'] = np.ones(n, int)
    for k in waste.FEATURES:
      a['feat_' + k] = np.arange(n, dtype=float)
    a['feat_queue_match'] = np.ones(n, bool)
    t = predict.step_table({'s': a})
    self.assertEqual(len(t['seed']), 3)
    self.assertEqual(list(t['frontier']), [False, True, False])
    self.assertEqual(list(t['low_yield']), [False, False, False])  # flips/new_bg all 0 -> idle
    self.assertEqual(list(t['enclosed']), [True, False, True])


if __name__ == '__main__':
  unittest.main()
