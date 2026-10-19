"""Benchmark metrics, seed selection and model reuse."""

import json
import unittest
import unittest.mock

import numpy as np

from liconn_v2 import benchmark
from liconn_v2 import data
from liconn_v2 import reference


class MetricsTest(unittest.TestCase):

  def test_score_hand_computed(self):
    gt = np.zeros((4, 4, 4), np.uint64)
    gt[0, :, :] = 7      # target object G, 16 voxels
    gt[1, :2, :] = 9     # neighbour, 8 voxels
    mask = np.zeros(gt.shape, bool)
    mask[0, :2, :] = True    # 8 voxels of G
    mask[1, :, :] = True     # 8 voxels of object 9 + 8 unlabeled
    with unittest.mock.patch.object(benchmark, 'MERGE_MIN_VOXELS', 5):
      s = benchmark.score(mask, gt, (0, 0, 0))
    self.assertEqual(s['object_id'], 7)
    self.assertEqual(s['mask_voxels'], 24)
    self.assertEqual(s['correct_voxels'], 8)
    self.assertEqual(s['object_voxels'], 16)
    self.assertAlmostEqual(s['recall'], 8 / 16)
    self.assertAlmostEqual(s['precision'], 8 / 16)   # labeled part of M: 8 (G) + 8 (obj 9)
    self.assertAlmostEqual(s['unlabeled_fraction'], 8 / 24)
    self.assertEqual(s['merged_voxels'], 8)
    self.assertEqual(s['merges'], [[9, 8]])

  def test_score_rejects_unlabeled_seed(self):
    gt = np.zeros((2, 2, 2), np.uint64)
    with self.assertRaises(ValueError):
      benchmark.score(np.ones(gt.shape, bool), gt, (0, 0, 0))

  def test_object_mask_uses_segment_threshold(self):
    canvas = np.array([np.nan, -1.0, 0.40, 0.41, 2.0], np.float32)
    # logit(0.6) = 0.405465
    np.testing.assert_array_equal(benchmark.object_mask(canvas), [False, False, False, True, True])

  def test_cost_and_summary_totals(self):
    c = benchmark.cost(steps=10, correct_voxels=4000, gflop_per_eval=2.0)
    self.assertEqual(c['network_gflop'], 20.0)
    self.assertEqual(c['evals_per_1000_correct_voxels'], 2.5)
    self.assertEqual(c['gflop_per_correct_voxel'], 0.005)
    recs = [dict(class_='axon', recall=0.5, precision=1.0, unlabeled_fraction=0.1, merges=[], merged_voxels=0,
                 steps=10, network_gflop=20.0, correct_voxels=4000, trace_seconds=1.0),
            dict(class_='dendrite', recall=1.0, precision=0.5, unlabeled_fraction=0.3, merges=[[1, 2000]],
                 merged_voxels=2000, steps=30, network_gflop=60.0, correct_voxels=1000, trace_seconds=3.0)]
    for r in recs:
      r['class'] = r.pop('class_')
    s = benchmark.summarize(recs)['all']
    self.assertEqual(s['mean_recall'], 0.75)
    self.assertEqual(s['total_merges'], 1)
    self.assertEqual(s['total_steps'], 40)
    self.assertEqual(s['evals_per_1000_correct_voxels'], 1000 * 40 / 5000)  # ratio of totals
    self.assertEqual(s['gflop_per_correct_voxel'], 80.0 / 5000)

  def test_seed_point_synthetic(self):
    gt = np.zeros((60, 60, 60), np.uint64)
    zz, yy, xx = np.mgrid[:60, :60, :60]
    gt[(zz - 30) ** 2 + (yy - 31) ** 2 + (xx - 29) ** 2 <= 100] = 5
    self.assertEqual(benchmark.seed_point(gt, 5), (30, 31, 29))
    # A slab cut by the z=0 crop face: the face is boundary, so the deepest point is
    # not on the face; the margin keeps it at z >= 16.
    gt2 = np.zeros((60, 60, 60), np.uint64)
    gt2[:40, 10:50, 10:50] = 3
    z, y, x = benchmark.seed_point(gt2, 3)
    self.assertTrue(16 <= z < 40 and 16 <= y < 44 and 16 <= x < 44)
    self.assertEqual(gt2[z, y, x], 3)
    gt3 = np.zeros((60, 60, 60), np.uint64)
    gt3[:10] = 4  # entirely inside the margin band
    with self.assertRaises(ValueError):
      benchmark.seed_point(gt3, 4)


class SeedSelectionTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
    cls.seeds = benchmark.select_seeds(cls.gt)

  def test_classes_file(self):
    c = benchmark.load_classes()
    self.assertEqual((len(c['axons']), len(c['dendrites'])), (544, 311))
    self.assertFalse(set(c['axons']) & set(c['dendrites']))
    self.assertIn('ng_states', json.loads(benchmark.CLASSES_FILE.read_text())['source'])

  def test_seeds_follow_the_rules(self):
    classes = benchmark.load_classes()
    names = [s['name'] for s in self.seeds]
    self.assertEqual(len(names), len(set(names)))
    self.assertEqual([s['class'] for s in self.seeds], ['axon'] * 6 + ['dendrite'] * 5 + ['unclassified'])
    sizes = dict(zip(*[a.tolist() for a in np.unique(self.gt, return_counts=True)]))
    inner = self.gt[16:-16, 16:-16, 16:-16]
    for s in self.seeds:
      z, y, x = s['seed_zyx']
      self.assertEqual(int(self.gt[z, y, x]), s['object_id'], s)
      self.assertTrue(all(16 <= v < 484 for v in (z, y, x)), s)
      if s['class'] != 'unclassified':
        self.assertIn(s['object_id'], classes[s['class'] + 's'])
        self.assertLessEqual(sizes[s['object_id']], benchmark.MAX_CROP_VOXELS)
        self.assertGreaterEqual(int((inner == s['object_id']).sum()), benchmark.MIN_INTERIOR_VOXELS)
    self.assertEqual(self.seeds[-1]['seed_zyx'], reference.SEED_XYZ[::-1])

  def test_selection_is_deterministic(self):
    self.assertEqual(benchmark.select_seeds(self.gt), self.seeds)


class TracerReuseTest(unittest.TestCase):

  def test_reused_tracer_gives_identical_traces(self):
    # Short traces (2 and 4 FFN steps), chosen by probing the crop.
    a, b = (300, 400, 200), (304, 400, 200)
    raw = reference.normalize(data.load_raw_xyz())
    with reference.Tracer(data.checkpoint_path()) as tracer:
      self.assertAlmostEqual(reference.gflop_per_eval(tracer.params), 81.600564672, places=6)
      first = tracer.trace(raw, a)
      other = tracer.trace(raw, b)
      again = tracer.trace(raw, a)
    self.assertNotEqual(first.history, other.history)
    self.assertEqual(first.history, again.history)
    self.assertEqual(first.seed.tobytes(), again.seed.tobytes())
    self.assertEqual(again.counters['executor-inference-calls'], again.iterations)


if __name__ == '__main__':
  unittest.main()
