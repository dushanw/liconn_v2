"""Many-seed tolerance: crop rule, seed rule and verdict (docs/manyseed_design.md)."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from liconn_v2 import benchmark
from liconn_v2 import data
from liconn_v2 import manyseed

REPO = Path(__file__).resolve().parent.parent


def run_pair(values, inputs='same', cls='dendrite', prefix='s'):
  """Reference and candidate runs from [(ref_recall, ref_prec, cand_recall, cand_prec), ...]."""
  ref, cand = {'seeds': [], 'records': [], 'inputs_sha256': {'raw': 'a'}}, \
      {'seeds': [], 'records': [], 'inputs_sha256': {'raw': 'a' if inputs == 'same' else 'b'}}
  for i, (rr, rp, cr, cp) in enumerate(values):
    name = f'{prefix}{i}'
    seed = {'name': name, 'class': cls, 'object_id': i + 1, 'seed_zyx': [i, i, i]}
    for run, r, p in ((ref, rr, rp), (cand, cr, cp)):
      run['seeds'].append(dict(seed))
      run['records'].append({'name': name, 'recall': r, 'precision': p})
  return ref, cand


def steady(n):
  return [(0.9, 0.95, 0.9, 0.95)] * n


class CropRuleTest(unittest.TestCase):

  def test_grid_inside_volume_and_through_crop_a(self):
    pos = manyseed.candidate_positions()
    self.assertEqual(len(pos), 63)
    self.assertIn(tuple(manyseed.CROP_A_XYZ), pos)
    for p in pos:
      for i in range(3):
        self.assertLessEqual(p[i] + manyseed.CROP_SIZE, manyseed.VOLUME_XYZ[i])

  def test_overlap(self):
    self.assertTrue(manyseed.overlaps((0, 0, 0), (499, 0, 0)))
    self.assertFalse(manyseed.overlaps((0, 0, 0), (500, 0, 0)))

  def test_coverage_uses_every_touched_cell(self):
    c = np.zeros((240, 305, 50), np.uint8)
    c[6:37, 5:37, 6:39] = 1  # crop at x 100-600 also touches coarse x 37 (600/16 = 37.5)
    self.assertLess(manyseed.coverage(c, (100, 83, 111)), 1.0)
    c[6:38, 5:37, 6:39] = 1
    self.assertEqual(manyseed.coverage(c, (100, 83, 111)), 1.0)

  def test_selection_excludes_overlaps_low_coverage_and_is_deterministic(self):
    c = np.ones((240, 305, 50), np.uint8)
    c[:60] = 0  # x < 960 unlabeled: crops starting at x 100 and 600 fail coverage
    chosen = manyseed.select_crops(c)
    self.assertEqual(len(chosen), manyseed.N_CROPS)
    self.assertEqual(chosen, manyseed.select_crops(c))
    for k, x in enumerate(chosen):
      self.assertEqual(x['index'], k)
      self.assertGreaterEqual(x['coverage'], manyseed.MIN_COVERAGE)
      self.assertFalse(manyseed.overlaps(x['xyz'], manyseed.CROP_A_XYZ))
      self.assertFalse(manyseed.overlaps(x['xyz'], manyseed.CROP_B_XYZ))
      self.assertGreaterEqual(x['xyz'][0], 1100)

  def test_recorded_crops_follow_the_rule(self):
    coarse = manyseed.cache_dir() / manyseed.COARSE_FILE
    manifest = manyseed.load_manifest()
    self.assertEqual(data.sha256_file(coarse), manifest['coarse_sha256'])
    expected = manyseed.select_crops(np.load(coarse), None)[:len(manifest['crops'])]
    self.assertGreaterEqual(len(manifest['crops']), manyseed.N_CROPS)
    self.assertEqual([(c['index'], c['xyz'], c['coverage']) for c in manifest['crops']],
                     [(c['index'], c['xyz'], c['coverage']) for c in expected])
    self.assertEqual(expected[:manyseed.N_CROPS], manyseed.select_crops(np.load(coarse)))

  def test_recorded_crops_match_pinned_hashes(self):
    manifest = manyseed.load_manifest()
    for c in manifest['crops']:
      manyseed.verify_crop(c['index'], manifest)


class SeedRuleTest(unittest.TestCase):

  def _volume(self, n_dend, n_ax):
    gt = np.zeros((40 + 40 * (n_dend + n_ax), 60, 60), np.uint32)
    ids_d, ids_a = list(range(1, n_dend + 1)), list(range(100, 100 + n_ax))
    for j, i in enumerate(ids_d + ids_a):
      gt[20 + 40 * j:50 + 40 * j, 17:43, 17:43] = i  # 30 x 26 x 26 > 20k interior voxels
    return gt, {'dendrites': ids_d, 'axons': ids_a}

  def test_fifteen_dendrites_then_axons_to_thirty(self):
    gt, classes = self._volume(18, 20)
    seeds = manyseed.select_seeds_for_crop(gt, 3, set(), classes)
    self.assertEqual(sum(s['class'] == 'dendrite' for s in seeds), 15)
    self.assertEqual(len(seeds), 30)
    self.assertTrue(all(s['name'].startswith('m3_') and s['crop'] == 3 for s in seeds))
    self.assertEqual(seeds, manyseed.select_seeds_for_crop(gt, 3, set(), classes))

  def test_few_dendrites_filled_with_axons_and_exclusions_respected(self):
    gt, classes = self._volume(4, 40)
    seeds = manyseed.select_seeds_for_crop(gt, 0, {1, 100}, classes)
    ids = {s['object_id'] for s in seeds}
    self.assertEqual(sum(s['class'] == 'dendrite' for s in seeds), 3)
    self.assertEqual(len(seeds), 30)
    self.assertNotIn(1, ids)
    self.assertNotIn(100, ids)

  def test_crop_b_seed_objects_match_the_recorded_run(self):
    run = json.loads((data.cache_dir() / 'runs' / 'heldout-fp32-20261021' / 'merged' / 'benchmark.json').read_text())
    self.assertEqual({s['object_id'] for s in run['seeds']}, set(manyseed.CROP_B_SEED_OBJECTS))

  def test_recorded_seeds_are_fresh_and_within_limits(self):
    seeds = manyseed.recorded_seeds()
    self.assertEqual(len(seeds), manyseed.TARGET_SEEDS)
    crops = [s['crop'] for s in seeds]
    self.assertEqual(crops, sorted(crops))
    # extra crops only while short of 300: without the last crop there were fewer than 300
    self.assertLess(sum(c != max(crops) for c in crops), manyseed.TARGET_SEEDS)
    ids = [s['object_id'] for s in seeds]
    self.assertEqual(len(ids), len(set(ids)))
    self.assertFalse(set(ids) & manyseed.excluded_objects())
    self.assertEqual(len({s['name'] for s in seeds}), len(seeds))
    for k in sorted(set(crops)):
      mine = [s for s in seeds if s['crop'] == k]
      self.assertLessEqual(len(mine), manyseed.SEEDS_PER_CROP)
      self.assertLessEqual(sum(s['class'] == 'dendrite' for s in mine), manyseed.MAX_DENDRITES_PER_CROP)

  def test_later_and_last_crop_seeds_rederive_from_their_crops(self):
    seeds = manyseed.recorded_seeds()
    last = max(s['crop'] for s in seeds)
    for k in (1, last):  # cross-crop exclusion; the last crop, cut to reach exactly 300
      mine = manyseed.seeds_for_crop(k, manyseed.load_proofread_zyx(k))
      self.assertEqual(len(mine), sum(s['crop'] == k for s in seeds))
    self.assertEqual(sum(s['crop'] < last for s in seeds) + len(mine), manyseed.TARGET_SEEDS)

  def test_first_crop_seeds_rederive_from_the_crop(self):
    gt = manyseed.load_proofread_zyx(0)
    self.assertEqual(len(manyseed.seeds_for_crop(0, gt)), sum(s['crop'] == 0 for s in manyseed.recorded_seeds()))
    for s in manyseed.seeds_for_crop(0):
      self.assertEqual(int(gt[tuple(s['seed_zyx'])]), s['object_id'])
      self.assertTrue(all(benchmark.MARGIN <= v < n - benchmark.MARGIN for v, n in zip(s['seed_zyx'], gt.shape)))


class VerdictTest(unittest.TestCase):

  def test_steady_300_passes(self):
    v = manyseed.verdict(*[[r] for r in run_pair(steady(300))])
    self.assertTrue(v['passes'], v['rules'])
    self.assertEqual(v['counts'], {'new_merge': 0, 'new_early_stop': 0, 'avoided_merge': 0, 'extension': 0})

  def test_exactly_ten_points_is_not_a_flip(self):
    v = manyseed.verdict(*[[r] for r in run_pair([(0.5, 0.5, 0.4, 0.4)] + steady(299))])
    self.assertEqual(v['counts']['new_merge'], 0)
    self.assertEqual(v['counts']['new_early_stop'], 0)
    v = manyseed.verdict(*[[r] for r in run_pair([(0.5, 0.5, 0.39, 0.39)] + steady(299))])
    self.assertEqual(v['counts']['new_merge'], 1)
    self.assertEqual(v['counts']['new_early_stop'], 1)

  def test_limits_are_not_crossed_by_float_rounding(self):
    v = manyseed.verdict(*[[r] for r in run_pair([(0.3, 0.3, 0.2, 0.2)] + steady(299))])  # 0.2 - 0.3 < -0.1 in floats
    self.assertEqual(v['counts']['new_merge'], 0)
    self.assertEqual(v['counts']['new_early_stop'], 0)
    v = manyseed.verdict(*[[r] for r in run_pair([(0.90, 0.95, 0.89, 0.94)] * 300)])  # exactly 1 point
    self.assertTrue(v['rules']['mean_recall_drop'])
    self.assertTrue(v['rules']['mean_precision_drop'])

  def test_missing_or_unpinned_inputs_fail(self):
    ref, cand = run_pair(steady(300))
    del ref['inputs_sha256'], cand['inputs_sha256']
    self.assertFalse(manyseed.verdict([ref], [cand])['rules']['complete_same_inputs'])
    ref, cand = run_pair(steady(300))
    self.assertTrue(manyseed.verdict([ref], [cand], None, [{'raw': 'a'}])['passes'])
    self.assertFalse(manyseed.verdict([ref], [cand], None, [{'raw': 'z'}])['passes'])

  def test_three_new_merges_pass_four_fail(self):
    merge = (0.9, 0.95, 0.9, 0.5)
    v3 = manyseed.verdict(*[[r] for r in run_pair([merge] * 3 + steady(297))])
    self.assertTrue(v3['rules']['new_merges'])
    v4 = manyseed.verdict(*[[r] for r in run_pair([merge] * 4 + steady(296))])
    self.assertFalse(v4['rules']['new_merges'])
    self.assertFalse(v4['passes'])

  def test_avoided_merges_do_not_offset_new_ones(self):
    v = manyseed.verdict(*[[r] for r in run_pair([(0.9, 0.95, 0.9, 0.8)] * 4 + [(0.9, 0.2, 0.9, 0.95)] * 10
                                                  + steady(286))])
    self.assertEqual(v['counts']['new_merge'], 4)
    self.assertEqual(v['counts']['avoided_merge'], 10)
    self.assertFalse(v['rules']['new_merges'])

  def test_four_new_early_stops_fail(self):
    v = manyseed.verdict(*[[r] for r in run_pair([(0.9, 0.95, 0.7, 0.95)] * 4 + steady(296))])
    self.assertFalse(v['rules']['new_early_stops'])

  def test_mean_drops_over_one_point_fail(self):
    v = manyseed.verdict(*[[r] for r in run_pair([(0.9, 0.95, 0.88, 0.95)] * 300)])
    self.assertFalse(v['rules']['mean_recall_drop'])
    v = manyseed.verdict(*[[r] for r in run_pair([(0.9, 0.95, 0.9, 0.93)] * 300)])
    self.assertFalse(v['rules']['mean_precision_drop'])

  def test_incomplete_or_mismatched_runs_fail(self):
    ref, cand = run_pair(steady(300))
    cand['records'].pop()
    self.assertFalse(manyseed.verdict([ref], [cand])['rules']['complete_same_inputs'])
    ref, cand = run_pair(steady(300), inputs='different')
    self.assertFalse(manyseed.verdict([ref], [cand])['passes'])
    ref, cand = run_pair(steady(300))
    cand['seeds'][0]['seed_zyx'] = [9, 9, 9]
    self.assertFalse(manyseed.verdict([ref], [cand])['passes'])
    ref, cand = run_pair(steady(300))
    registered = [{'name': f's{i}', 'crop': 0, 'object_id': i + 1, 'seed_zyx': [i, i, i]} for i in range(301)]
    v = manyseed.verdict([ref], [cand], registered)
    self.assertFalse(v['rules']['complete_same_inputs'])

  def test_runs_must_trace_the_registered_seeds_on_their_crops(self):
    registered = [{'name': f's{i}', 'crop': 0 if i < 150 else 1, 'object_id': i + 1, 'seed_zyx': [i, i, i]}
                  for i in range(300)]

    def split(ref, cand):
      halves = lambda run: [{**run, 'seeds': run['seeds'][a:b], 'records': run['records'][a:b]}
                            for a, b in ((0, 150), (150, 300))]
      return halves(ref), halves(cand)

    ref, cand = run_pair(steady(300))
    for run in (ref, cand):
      for rec, seed in zip(run['records'], run['seeds']):
        rec.update(object_id=seed['object_id'], seed_zyx=list(seed['seed_zyx']))
    self.assertTrue(manyseed.verdict(*split(ref, cand), registered)['passes'])
    # the same records, but with crop 1's seeds in the first pair: fails
    swapped = [{**r, 'crop': 1 - r['crop']} for r in registered]
    self.assertFalse(manyseed.verdict(*split(ref, cand), swapped)['rules']['complete_same_inputs'])
    # both runs agree with each other, but not with the registered seed point: fails
    for run in (ref, cand):
      run['seeds'][5]['seed_zyx'] = [9, 9, 8]
      run['records'][5]['seed_zyx'] = [9, 9, 8]
    self.assertFalse(manyseed.verdict(*split(ref, cand), registered)['rules']['complete_same_inputs'])

  def test_fewer_than_300_seeds_cannot_pass(self):
    v = manyseed.verdict(*[[r] for r in run_pair(steady(299))])
    self.assertFalse(v['rules']['all_300_seeds'])
    self.assertFalse(v['passes'])

  def test_upper_bound(self):
    self.assertAlmostEqual(manyseed.upper_bound(0, 300), 1 - 0.05 ** (1 / 300), places=6)
    self.assertAlmostEqual(manyseed.upper_bound(3, 300), 0.0258, places=3)

  def test_development_verdict(self):
    with tempfile.TemporaryDirectory() as d:
      subprocess.run([sys.executable, str(REPO / 'examples' / 'judge_manyseed.py'), '--dev', '--out', d],
                     check=True, capture_output=True)
      v = json.loads((Path(d) / 'manyseed_dev_verdict.json').read_text())
    self.assertEqual(v['seeds'], 24)
    self.assertEqual(v['counts'], {'new_merge': 2, 'new_early_stop': 0, 'avoided_merge': 2, 'extension': 2})
    self.assertEqual(sorted(n for n, e in v['flipped_seeds'].items() if e['new_merge']),
                     ['heldout_dendrite_245099', 'heldout_dendrite_250842'])
    self.assertFalse(v['passes'])


if __name__ == '__main__':
  unittest.main()
