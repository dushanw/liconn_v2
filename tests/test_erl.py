"""Adapted ERL (liconn_v2/erl.py, docs/fresh_design.md)."""

import unittest

import numpy as np

from liconn_v2 import erl


class SkeletonTest(unittest.TestCase):

  def test_straight_tube_has_its_length(self):
    obj = np.zeros((240, 30, 30), bool)  # x, y, z
    x, y, z = np.ogrid[:240, :30, :30]
    obj[20:220] = ((y - 15) ** 2 * 18 ** 2 + (z - 15) ** 2 * 24 ** 2 <= 90 ** 2)[0]
    sk = erl.skeleton(obj)
    tube_um = 200 * 18 / 1000.0
    self.assertLess(abs(sk['length_um'] - tube_um) / tube_um, 0.10)
    self.assertLessEqual(sk['length_um'], erl.GATE_RATIO * sk['bbox_diagonal_um'])
    v = sk['voxels_xyz']
    self.assertTrue(obj[v[:, 0], v[:, 1], v[:, 2]].all())  # vertices map back inside the object

  def test_origin_shifts_voxel_coordinates(self):
    obj = np.zeros((120, 20, 20), bool)
    obj[10:110, 6:14, 6:14] = True
    a, b = erl.skeleton(obj), erl.skeleton(obj, (5, 7, 9))
    np.testing.assert_array_equal(b['voxels_xyz'], a['voxels_xyz'] + [5, 7, 9])


class AxisOrderTest(unittest.TestCase):

  def test_tube_along_z_maps_to_the_mask_in_zyx(self):
    obj = np.zeros((20, 24, 160), bool)  # x, y, z; tube along z (24 nm voxels)
    obj[6:14, 8:16, 10:150] = True
    sk = erl.skeleton(obj)
    tube_um = 140 * 24 / 1000.0
    self.assertLess(abs(sk['length_um'] - tube_um) / tube_um, 0.10)
    mask = np.transpose(obj, [2, 1, 0])  # z, y, x
    t = erl.seed_terms(sk, mask, merged=False)
    self.assertAlmostEqual(t['sum_sq_um2'], sk['length_um'] ** 2, places=6)  # one run, fully covered
    half = mask.copy()
    half[80:] = False  # only the first half of the tube along z
    th = erl.seed_terms(sk, half, merged=False)
    self.assertLess(abs(th['longest_run_um'] - sk['length_um'] / 2) / (sk['length_um'] / 2), 0.15)
    # the same mask passed in the wrong axis order (x, y, z read as z, y, x) does not cover the tube
    self.assertLess(erl.seed_terms(sk, obj, merged=False)['sum_sq_um2'], 0.01 * t['sum_sq_um2'])


class RunsTest(unittest.TestCase):

  def test_runs_need_both_ends_covered(self):
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    lengths = np.array([1.0, 2.0, 3.0, 4.0])
    self.assertEqual(erl.run_lengths(np.array([1, 1, 0, 1, 1], bool), edges, lengths), [4.0, 1.0])
    self.assertEqual(erl.run_lengths(np.ones(5, bool), edges, lengths), [10.0])
    self.assertEqual(erl.run_lengths(np.zeros(5, bool), edges, lengths), [])

  def test_branches_join_into_one_run(self):
    edges = np.array([[0, 1], [1, 2], [1, 3]])
    self.assertEqual(erl.run_lengths(np.ones(4, bool), edges, np.array([1.0, 1.0, 2.0])), [4.0])

  def test_seed_terms_and_merge_zeroing(self):
    skel = {'voxels_xyz': np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
            'edges': np.array([[0, 1], [1, 2], [2, 3]]), 'lengths_um': np.array([1.0, 1.0, 1.0]),
            'length_um': 3.0}
    mask = np.zeros((1, 1, 4), bool)  # z, y, x
    mask[0, 0, :3] = True
    t = erl.seed_terms(skel, mask, merged=False)
    self.assertEqual((t['sum_sq_um2'], t['longest_run_um'], t['runs']), (4.0, 2.0, 1))
    t = erl.seed_terms(skel, mask, merged=True)
    self.assertEqual((t['sum_sq_um2'], t['sum_sq_no_merge_um2']), (0.0, 4.0))

  def test_vertices_outside_the_mask_volume_are_not_covered(self):
    skel = {'voxels_xyz': np.array([[0, 0, 0], [5, 0, 0]]), 'edges': np.array([[0, 1]]),
            'lengths_um': np.array([1.0]), 'length_um': 1.0}
    self.assertEqual(erl.seed_terms(skel, np.ones((1, 1, 2), bool), merged=False)['sum_sq_um2'], 0.0)


class SummaryTest(unittest.TestCase):

  def test_length_weighting_classes_and_merges(self):
    terms = [{'skeleton_um': 10.0, 'sum_sq_um2': 100.0, 'sum_sq_no_merge_um2': 100.0, 'merged': False, 'class': 'axon'},
             {'skeleton_um': 30.0, 'sum_sq_um2': 0.0, 'sum_sq_no_merge_um2': 900.0, 'merged': True, 'class': 'dendrite'}]
    s = erl.summarize(terms)
    self.assertEqual((s['erl_um'], s['erl_no_merge_penalty_um']), (2.5, 25.0))
    self.assertEqual((s['erl_um_axon'], s['erl_um_dendrite']), (10.0, 0.0))
    self.assertEqual((s['seed_mean_erl_um'], s['merged_seeds']), (5.0, 1))
    self.assertEqual((s['erl_no_merge_penalty_um_axon'], s['erl_no_merge_penalty_um_dendrite']), (10.0, 30.0))

  def test_merge_definition(self):
    self.assertTrue(erl.merged({'merges': [[5, 1000]]}))
    self.assertFalse(erl.merged({'merges': []}))

  def test_gate(self):
    ok = {'length_um': 2.5, 'bbox_diagonal_um': 1.0}
    self.assertEqual(erl.GATE_RATIO, 3.0)  # the owner-approved gate (docs/fresh_design.md)
    bad = {'length_um': 50.0, 'bbox_diagonal_um': 1.0}
    self.assertTrue(erl.gate([ok] * 19 + [bad])['passes'])
    self.assertFalse(erl.gate([ok] * 18 + [bad] * 2)['passes'])
    self.assertEqual(erl.gate([ok, bad])['max_ratio'], 50.0)
    empty = {'length_um': 0.0, 'bbox_diagonal_um': 1.0}
    self.assertEqual(erl.gate([ok, empty])['within_ratio'], 1)
    self.assertEqual(erl.gate([ok, empty])['empty'], 1)


if __name__ == '__main__':
  unittest.main()
