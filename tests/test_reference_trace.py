"""A fresh CPU run reproduces the committed reference trace exactly (~2.5 min)."""

import os
import tempfile
import unittest

import numpy as np

from liconn_v2 import benchmark
from liconn_v2 import data
from liconn_v2 import reference


class ReferenceTraceTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.trace, cls.fp = reference.run_reference()
    cls.expected = reference.load_expected()

  def test_canvas_is_bitwise_identical(self):
    self.assertEqual(self.fp['seed_sha256'], self.expected['seed_sha256'])
    self.assertEqual(self.fp['predicted_mask_sha256'], self.expected['predicted_mask_sha256'])
    # Every committed field, not only the ones checked individually here.
    self.assertEqual(self.fp, self.expected)

  def test_visit_order_is_identical(self):
    self.assertEqual(self.fp['iterations'], self.expected['iterations'])
    self.assertEqual(self.fp['history_sha256'], self.expected['history_sha256'])
    self.assertEqual(self.trace.history[0], reference.SEED_XYZ[::-1])
    # FaceMaxMovementPolicy (ffn/inference/movement.py): every move is exactly
    # +/-delta along one axis to a face maximum, within +/-delta on the others,
    # from some earlier FOV center.
    visited = np.array(self.trace.history)
    for i, pos in enumerate(self.trace.history):
      self.assertTrue(all(16 <= p < 500 - 16 for p in pos), pos)
      if i == 0:
        continue
      steps = np.abs(np.array(pos) - visited[:i])
      self.assertTrue((steps.max(axis=1) == 8).any(), pos)

  def test_masks_are_identical(self):
    for key in ('notebook_mask_sha256', 'notebook_mask_voxels',
                'seed_component_sha256', 'seed_component_voxels'):
      self.assertEqual(self.fp[key], self.expected[key], key)
    self.assertTrue(self.fp['notebook_mask_contains_seed'])
    mask = reference.notebook_mask(self.trace.seed)
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'mask.npz')
      reference.save_mask_npz(path, mask)
      np.testing.assert_array_equal(reference.load_mask_npz(path), mask)

  def test_canvas_values_are_valid(self):
    seed = self.trace.seed
    self.assertEqual(seed.dtype, np.float32)
    self.assertEqual(seed.shape, (500, 500, 500))
    self.assertFalse(np.isinf(seed).any())
    # The upfront compile call must not be counted as a trace step.
    self.assertEqual(self.trace.counters['executor-inference-calls'], self.trace.iterations)
    self.assertEqual(self.trace.backend, 'cpu')

  def test_canvas_support_is_union_of_fovs(self):
    # Canvas starts all-NaN; each step writes exactly one 33^3 box around its center.
    expected = np.zeros(self.trace.seed.shape, bool)
    for z, y, x in self.trace.history:
      expected[z - 16:z + 17, y - 16:y + 17, x - 16:x + 17] = True
    np.testing.assert_array_equal(~np.isnan(self.trace.seed), expected)

  def test_scores_against_proofread(self):
    gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
    seed = reference.SEED_XYZ[::-1]
    s = benchmark.score(benchmark.object_mask(self.trace.seed), gt, seed)
    self.assertEqual(s['object_id'], 97257)
    self.assertTrue(0 < s['recall'] <= 1 and 0 < s['precision'] <= 1, s)
    self.assertEqual(s['mask_voxels'], int(benchmark.object_mask(self.trace.seed).sum()))


if __name__ == '__main__':
  unittest.main()
