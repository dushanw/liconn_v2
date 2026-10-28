"""Held-out crop B and its seed rule."""

import unittest

import numpy as np

from liconn_v2 import benchmark
from liconn_v2 import heldout


class HeldoutTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.gt = np.ascontiguousarray(np.transpose(heldout.load_proofread_xyz(), [2, 1, 0]))
    cls.seeds = heldout.select_seeds(cls.gt)

  def test_cache_matches_pins(self):
    heldout.verify_cache()

  def test_crop_does_not_overlap_the_benchmark_crop(self):
    from liconn_v2 import data  # pylint: disable=g-import-not-at-top
    a, b = np.array(data.CROP_XYZ), np.array(heldout.CROP_XYZ)
    overlap = np.all(np.maximum(a, b) < np.minimum(a + data.CROP_SIZE, b + heldout.CROP_SIZE))
    self.assertFalse(overlap)

  def test_seed_rule(self):
    classes = benchmark.load_classes()
    self.assertEqual([s['class'] for s in self.seeds], ['axon'] * 6 + ['dendrite'] * 6)
    sizes = dict(zip(*[x.tolist() for x in np.unique(self.gt, return_counts=True)]))
    inner = self.gt[16:-16, 16:-16, 16:-16]
    for s in self.seeds:
      self.assertNotIn(s['object_id'], heldout.BENCHMARK_SEED_OBJECTS)
      self.assertIn(s['object_id'], classes[s['class'] + 's'])
      z, y, x = s['seed_zyx']
      self.assertEqual(int(self.gt[z, y, x]), s['object_id'])
      self.assertTrue(all(16 <= v < 484 for v in (z, y, x)))
      self.assertLessEqual(sizes[s['object_id']], benchmark.MAX_CROP_VOXELS)
      self.assertGreaterEqual(int((inner == s['object_id']).sum()), benchmark.MIN_INTERIOR_VOXELS)
    self.assertEqual(heldout.select_seeds(self.gt), self.seeds)
    self.assertEqual(len({s['name'] for s in self.seeds}), 12)


if __name__ == '__main__':
  unittest.main()
