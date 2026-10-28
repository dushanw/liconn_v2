"""Tests for liconn_v2.segneuron (docs/segneuron_design.md) on synthetic volumes."""

import unittest

import numpy as np

from liconn_v2 import segneuron
from liconn_v2 import vitseg


def authors_affinities(seg):
  """SegNeuron's affinity targets (seg_to_affgraph with mknhood3d(1), offsets (-1,0,0),
  (0,-1,0), (0,0,-1); docs/segneuron_sources.md): channel c at v is 1 when v and v + offset
  carry the same nonzero label. Border voxels (no partner) are left 0."""
  aff = np.zeros((3,) + seg.shape, np.float32)
  for c, off in enumerate(((-1, 0, 0), (0, -1, 0), (0, 0, -1))):
    here = tuple(slice(max(0, -o), n - max(0, o)) for o, n in zip(off, seg.shape))
    there = tuple(slice(max(0, o), n + min(0, o)) for o, n in zip(off, seg.shape))
    aff[c][here] = (seg[here] == seg[there]) & (seg[here] > 0) & (seg[there] > 0)
  return aff


def blobs(shape=(12, 24, 24), seed=0):
  """A label volume of a few boxes on background."""
  rng = np.random.default_rng(seed)
  seg = np.zeros(shape, np.int64)
  for k in range(1, 7):
    z0, y0, x0 = rng.integers(0, 6), rng.integers(0, 16), rng.integers(0, 16)
    seg[z0:z0 + rng.integers(3, 7), y0:y0 + rng.integers(4, 9), x0:x0 + rng.integers(4, 9)] = k
  return seg


class SegneuronTest(unittest.TestCase):

  def test_convention_matches_vitseg_targets(self):
    """The authors' affinity at v is the pair (v, v - o), our convention: on pairs of two
    labelled voxels it equals vitseg.affinity_targets."""
    seg = blobs()
    theirs = authors_affinities(seg)
    target, mask = vitseg.affinity_targets(seg, vitseg.OFFSETS[:vitseg.NEAREST])
    for c in range(3):
      m = mask[..., c] > 0
      self.assertTrue(m.any())
      np.testing.assert_array_equal(theirs[c][m], target[..., c][m])

  def test_to_crop_grid_recovers_coarse_affinities(self):
    """Affinities of a label volume upsampled 2x in y and x (nearest), mapped back, are the
    coarse volume's affinities (away from the crop's first row and column)."""
    seg = blobs()
    fine = np.repeat(np.repeat(seg, 2, axis=1), 2, axis=2)
    aff, fg = segneuron.to_crop_grid(authors_affinities(fine), (fine > 0).astype(np.float32))
    want = authors_affinities(seg)
    np.testing.assert_array_equal(fg, (seg > 0).astype(np.float32))
    np.testing.assert_array_equal(aff[0], want[0])
    np.testing.assert_array_equal(aff[1][:, 1:], want[1][:, 1:])
    np.testing.assert_array_equal(aff[2][:, :, 1:], want[2][:, :, 1:])

  def test_to_crop_grid_keeps_boundaries_inside_a_block(self):
    """y affinity of coarse v (pair v, v - y) is the minimum of the fine edges between the two
    centres: crossing (fine row 2i), inside v's block (2i + 1) and inside v - y's block
    (2i - 1); the mean over the block's 2 columns. x likewise."""
    z, y, x = 2, 3, 4
    aff_fine = np.ones((3, z, 2 * y, 2 * x), np.float32)
    aff_fine[1, :, 3, :] = 0.2  # inside block 1, fine edge (3, 2)
    aff_fine[2, :, :, 4] = 0.4  # crossing from block 1 into block 2, fine edge (4, 3)
    aff_fine[2, :, 0, 7] = 0.0  # inside block 3, one of the 2 rows of coarse row 0
    aff_fine[0] = np.arange(2 * y * 2 * x, dtype=np.float32).reshape(1, 2 * y, 2 * x)
    aff, _ = segneuron.to_crop_grid(aff_fine, np.zeros((z, 2 * y, 2 * x), np.float32))
    want_y = np.ones((z, y, x), np.float32)
    want_y[:, 1:3] = 0.2  # pairs (1, 0) and (2, 1) both touch block 1
    np.testing.assert_allclose(aff[1], want_y)
    want_x = np.ones((z, y, x), np.float32)
    want_x[:, :, 2] = 0.4
    want_x[:, 0, 3] = 0.5  # minimum 0 on one of the 2 rows, 1 on the other
    np.testing.assert_allclose(aff[2], want_x)
    np.testing.assert_allclose(aff[0][0, 0, 0], np.mean([0, 1, 8, 9]))
    with self.assertRaises(ValueError):
      segneuron.to_crop_grid(aff_fine[:, :, :5], np.zeros((z, 5, 2 * x), np.float32))

  def test_upsample_xy2(self):
    raw = np.random.default_rng(1).integers(0, 256, (3, 5, 7), dtype=np.uint8)
    up = segneuron.upsample_xy2(raw)
    self.assertEqual((up.shape, up.dtype), ((3, 10, 14), np.uint8))
    # pixels as areas: a linear ramp in x is sampled at coarse positions j - 0.25 and j + 0.25
    ramp = np.broadcast_to((20 * np.arange(8) + 40).astype(np.uint8), (2, 3, 8)).copy()
    fine = segneuron.upsample_xy2(ramp)[0, 0].astype(float)
    np.testing.assert_allclose(fine[1:-1], [20 * (k / 2 - 0.25) + 40 for k in range(1, 15)], atol=0.5)
    np.testing.assert_array_equal(segneuron.upsample_xy2(ramp)[:, 0::2], segneuron.upsample_xy2(ramp)[:, 1::2])
    const = np.full((2, 4, 4), 77, np.uint8)
    np.testing.assert_array_equal(segneuron.upsample_xy2(const), 77)
    with self.assertRaises(ValueError):
      segneuron.upsample_xy2(raw.astype(np.float32))

  def test_vitseg_layout(self):
    aff = np.random.default_rng(2).random((3, 4, 5, 6)).astype(np.float32)
    fg = np.random.default_rng(3).random((4, 5, 6)).astype(np.float32)
    a = segneuron.vitseg_layout(aff, fg)
    self.assertEqual(a.shape, (4, 5, 6, vitseg.CHANNELS))
    for c in range(3):
      np.testing.assert_array_equal(a[..., c], aff[c])
    np.testing.assert_array_equal(a[..., vitseg.NEURON_CHANNEL], fg)
    np.testing.assert_array_equal(a[..., 3:vitseg.NEURON_CHANNEL], segneuron.LONG_RANGE_FILL)

  def test_ours_with_label_outputs_recovers_the_labels(self):
    """Our pipeline on SegNeuron-layout outputs derived from the labels gives each object's
    seed segment nearly its whole 6-connected piece. Not exactly: the watershed can hand a
    voxel on a face between two touching objects to the neighbour's fragment (the oracle
    check in docs/vitseg_design.md likewise reached precision 0.997, not 1)."""
    seg = blobs()
    aff, fg = authors_affinities(seg), (seg > 0).astype(np.float32)
    objects = [{'name': f'o{k}', 'object_id': k, 'seed_zyx': [int(v) for v in np.argwhere(seg == k)[0]]}
               for k in np.unique(seg[seg > 0])]
    r = segneuron.ours(aff, fg, seg, objects)
    acc = segneuron.object_accounting(r['segmentation'], seg, objects)
    self.assertGreater(acc['per_object_mean']['precision'], 0.99)
    from scipy import ndimage  # pylint: disable=g-import-not-at-top
    for row, o in zip(acc['objects'], objects):
      piece, _ = ndimage.label(seg == o['object_id'])
      want = (piece == piece[tuple(o['seed_zyx'])]).sum() / (seg == o['object_id']).sum()
      self.assertAlmostEqual(row['recall'], want, delta=0.03)
    self.assertEqual(acc['background']['precision'], 1.0)

  def test_object_accounting(self):
    gt = np.zeros((1, 1, 10), np.int64)
    gt[0, 0, :8] = 5
    seg = np.array([[[1, 1, 1, 1, 0, 2, 2, 3, 1, 0]]], np.uint32)
    obj = [{'name': 'a', 'object_id': 5, 'seed_zyx': [0, 0, 0]}]
    acc = segneuron.object_accounting(seg, gt, obj)
    row = acc['objects'][0]
    self.assertAlmostEqual(row['recall'], 4 / 8)
    self.assertAlmostEqual(row['precision'], 1.0)  # the label-0 voxel in segment 1 is ignored
    self.assertAlmostEqual(row['lost_to_background'], 1 / 8)
    self.assertAlmostEqual(row['lost_to_other_segments'], 3 / 8)
    self.assertEqual(row['pieces_over_1pct'], 3)
    self.assertEqual(segneuron.summary(acc)['mean_f1'], row['f1'])

  def test_mask_background(self):
    seg = np.arange(1, 7, dtype=np.uint32).reshape(1, 2, 3)
    fg = np.array([[[0.0, 0.5, 0.49], [1.0, 0.2, 0.7]]], np.float32)
    np.testing.assert_array_equal(segneuron.mask_background(seg, fg), [[[0, 2, 0], [4, 0, 6]]])

  def test_output_scores(self):
    seg = blobs()
    aff, fg = authors_affinities(seg), (seg > 0).astype(np.float32)
    s = segneuron.output_scores(aff, fg, seg)
    self.assertEqual((s['foreground']['precision'], s['foreground']['recall']), (1.0, 1.0))
    for a in s['affinities']:
      self.assertEqual(a['same_below_0.5'], 0.0)
      self.assertEqual(a['diff_at_or_above_0.5'], 0.0)
      self.assertEqual(a['neuron_background_at_or_above_0.5'], 0.0)
      self.assertGreater(a['same_pairs'], 0)
      self.assertEqual(a['auc_same_vs_diff'], 1.0)
      self.assertEqual(a['auc_same_vs_neuron_background'], 1.0)
    self.assertEqual(s['foreground']['auc'], 1.0)
    flipped = segneuron.output_scores(1.0 - aff, fg, seg)
    for a in flipped['affinities']:
      self.assertEqual(a['same_below_0.5'], 1.0)
      self.assertEqual(a['auc_same_vs_diff'], 0.0)

  def test_auc(self):
    rng = np.random.default_rng(4)
    pos, neg = rng.random(4000) * 0.3 + 0.2, rng.random(3000) * 0.3 + 0.1
    exact = np.mean(pos[:, None] > neg[None, :])
    self.assertAlmostEqual(segneuron.auc(pos, neg), exact, delta=2e-3)
    self.assertEqual(segneuron.auc(np.full(5, 0.3), np.full(5, 0.3)), 0.5)
    self.assertTrue(np.isnan(segneuron.auc(pos, pos[:0])))


if __name__ == '__main__':
  unittest.main()
