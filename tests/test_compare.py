"""Backend selection, seed picking, merging and comparing runs (no GPU needed)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from liconn_v2 import benchmark
from liconn_v2 import compare
from liconn_v2 import reference

SHAPE = (20, 20, 20)


def _record(name, cls, recall, precision, merges=(), steps=10, correct=1000, merged=0,
            mask_voxels=1):
  return {'name': name, 'class': cls, 'recall': recall, 'precision': precision,
          'seed_zyx': [1, 1, 1],
          'object_id': 1, 'mask_voxels': mask_voxels,
          'unlabeled_fraction': 0.1, 'merges': [list(m) for m in merges], 'merged_voxels': merged,
          'steps': steps, 'network_gflop': steps * 2.0, 'correct_voxels': correct,
          'trace_seconds': 1.0, 'evals_per_1000_correct_voxels': 1000 * steps / correct,
          'history_sha256': f'h-{name}-{steps}', 'canvas_sha256': f'c-{name}-{steps}'}


def _write_run(d, records, masks):
  d = Path(d)
  d.mkdir(parents=True, exist_ok=True)
  seeds = [{'name': r['name'], 'class': r['class'], 'seed_zyx': [1, 1, 1], 'object_id': r['object_id']}
           for r in records]
  (d / 'benchmark.json').write_text(json.dumps({'seeds': seeds, 'records': records,
                                                'inputs_sha256': {'raw': 'same'}}))
  arrays = {'shape_zyx': np.array(SHAPE)}
  for name, m in masks.items():
    zz, yy, xx = np.nonzero(m)
    lo = np.array([zz.min(), yy.min(), xx.min()])
    hi = np.array([zz.max(), yy.max(), xx.max()]) + 1
    arrays['mask_' + name] = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    arrays['offset_zyx_' + name] = lo
  np.savez_compressed(d / 'benchmark_masks.npz', **arrays)


def _box(lo, hi):
  m = np.zeros(SHAPE, bool)
  m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
  return m


class BackendTest(unittest.TestCase):

  def test_backend_settings(self):
    self.assertEqual(reference.backend_settings('cpu'), ('cpu', 'cpu'))
    # JAX names the platform 'cuda' but reports the backend as 'gpu'.
    self.assertEqual(reference.backend_settings('gpu'), ('cuda', 'gpu'))
    with self.assertRaises(ValueError):
      reference.backend_settings('metal')

  def test_default_backend_is_cpu(self):
    # The controller does not pass LICONN_BACKEND; the bitwise reference is CPU-only.
    self.assertEqual((reference.JAX_PLATFORM, reference.EXPECTED_BACKEND), ('cpu', 'cpu'))


class PickTest(unittest.TestCase):

  def test_pick_keeps_canonical_order_and_rejects_unknown(self):
    seeds = [{'name': n} for n in ('a', 'b', 'c')]
    self.assertEqual([s['name'] for s in benchmark.pick(seeds, ['c', 'a'])], ['a', 'c'])
    self.assertEqual(benchmark.pick(seeds, []), seeds)
    with self.assertRaises(ValueError):
      benchmark.pick(seeds, ['a', 'zzz'])


class MergeCompareTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.root = Path(self.tmp.name)

  def tearDown(self):
    self.tmp.cleanup()

  def test_merge_matches_single_run_and_checks_completeness(self):
    ra = _record('a', 'axon', 0.5, 1.0)
    rb = _record('b', 'dendrite', 0.9, 0.8, merges=[(7, 2000)], steps=30, merged=2000)
    _write_run(self.root / 'p_b', [rb], {'b': _box((5, 5, 5), (9, 9, 9))})
    _write_run(self.root / 'p_a', [ra], {'a': _box((0, 0, 0), (4, 4, 4))})
    out = compare.merge_runs([self.root / 'p_b', self.root / 'p_a'], self.root / 'm', ['a', 'b'])
    self.assertEqual([r['name'] for r in out['records']], ['a', 'b'])  # canonical, not part order
    self.assertEqual(out['summary'], benchmark.summarize([ra, rb]))
    masks, shape = compare.load_masks(self.root / 'm')
    np.testing.assert_array_equal(masks['b'], _box((5, 5, 5), (9, 9, 9)))
    with self.assertRaises(ValueError):
      compare.merge_runs([self.root / 'p_a'], self.root / 'm2', ['a', 'b'])
    with self.assertRaises(ValueError):
      compare.merge_runs([self.root / 'p_a', self.root / 'p_a'], self.root / 'm3', ['a'])

  def test_compare_iou_flags_and_tolerance(self):
    base = [_record('a', 'axon', 0.80, 1.0), _record('b', 'dendrite', 0.90, 0.9, merges=[(7, 2000)], merged=2000)]
    cand = [_record('a', 'axon', 0.80, 1.0),
            _record('b', 'dendrite', 0.80, 0.9, merges=[(7, 2000), (8, 1500)], steps=12, merged=3500)]
    _write_run(self.root / 'base', base, {'a': _box((0, 0, 0), (4, 4, 4)), 'b': _box((5, 5, 5), (9, 9, 9))})
    # b's candidate mask is shifted by one voxel along z: IoU = 48 / 80.
    _write_run(self.root / 'cand', cand, {'a': _box((0, 0, 0), (4, 4, 4)), 'b': _box((6, 5, 5), (10, 9, 9))})
    r = compare.compare_runs(self.root / 'base', self.root / 'cand')
    rows = {x['name']: x for x in r['per_seed']}
    self.assertEqual(rows['a']['mask_iou'], 1.0)
    self.assertAlmostEqual(rows['b']['mask_iou'], 48 / 80)
    self.assertFalse(rows['a']['flag_for_review'])
    self.assertTrue(rows['b']['flag_for_review'])
    self.assertEqual(r['seeds_flagged_for_review'], ['b'])
    self.assertTrue(rows['a']['same_history'])
    self.assertFalse(rows['b']['same_history'])
    tol = r['overall']['all']['tolerance']
    self.assertAlmostEqual(tol['mean_recall_drop'], 0.05)
    self.assertEqual(tol['seeds_with_new_merges'], {'b': [8]})
    self.assertEqual(rows['b']['cand_merges'], [[7, 2000], [8, 1500]])
    self.assertAlmostEqual(tol['merged_voxels_growth'], 0.75)
    self.assertEqual(list(tol['seeds_with_recall_drop_over_limit']), ['b'])
    self.assertAlmostEqual(tol['seeds_with_recall_drop_over_limit']['b'], 0.10)
    self.assertFalse(tol['within_tolerance'])

  def test_identical_runs_are_within_tolerance(self):
    recs = [_record('a', 'axon', 0.8, 1.0), _record('axon_218', 'axon', 0.2, 1.0)]
    masks = {'a': _box((0, 0, 0), (4, 4, 4)), 'axon_218': _box((10, 10, 10), (12, 12, 12))}
    _write_run(self.root / 'x', recs, masks)
    _write_run(self.root / 'y', recs, masks)
    r = compare.compare_runs(self.root / 'x', self.root / 'y')
    self.assertEqual(r['seeds_flagged_for_review'], [])
    self.assertTrue(r['overall']['all']['tolerance']['within_tolerance'])
    self.assertEqual(r['overall']['without_sensitive_seeds']['seeds'], 1)

  def test_merge_rejects_parts_with_different_inputs_or_volumes(self):
    _write_run(self.root / 'p_a', [_record('a', 'axon', 0.5, 1.0)], {'a': _box((0, 0, 0), (2, 2, 2))})
    _write_run(self.root / 'p_b', [_record('b', 'axon', 0.5, 1.0)], {'b': _box((0, 0, 0), (2, 2, 2))})
    for name, h in (('p_a', 'x'), ('p_b', 'y')):
      run = json.loads((self.root / name / 'benchmark.json').read_text())
      run['inputs_sha256'] = {'raw': h}
      (self.root / name / 'benchmark.json').write_text(json.dumps(run))
    with self.assertRaises(ValueError):
      compare.merge_runs([self.root / 'p_a', self.root / 'p_b'], self.root / 'm', ['a', 'b'])
    run = json.loads((self.root / 'p_b' / 'benchmark.json').read_text())
    run['inputs_sha256'] = {'raw': 'x'}
    (self.root / 'p_b' / 'benchmark.json').write_text(json.dumps(run))
    out = compare.merge_runs([self.root / 'p_a', self.root / 'p_b'], self.root / 'm', ['a', 'b'])
    self.assertEqual(out['inputs_sha256'], {'raw': 'x'})
    np.savez_compressed(self.root / 'p_b' / 'benchmark_masks.npz', shape_zyx=np.array((5, 5, 5)))
    with self.assertRaises(ValueError):
      compare.merge_runs([self.root / 'p_a', self.root / 'p_b'], self.root / 'm2', ['a', 'b'])

  def test_per_seed_rules_and_merges_from_zero(self):
    base = [_record('a', 'axon', 0.80, 1.0), _record('c', 'axon', 0.80, 1.0)]
    cand = [_record('a', 'axon', 0.70, 1.0), _record('c', 'axon', 0.80, 0.9, merges=[(9, 1200)], merged=1200)]
    masks = {'a': _box((0, 0, 0), (4, 4, 4)), 'c': _box((5, 5, 5), (9, 9, 9))}
    _write_run(self.root / 'base', base, masks)
    _write_run(self.root / 'cand', cand, masks)
    r = compare.compare_runs(self.root / 'base', self.root / 'cand')
    rows = {x['name']: x['seed_tolerance'] for x in r['per_seed']}
    self.assertFalse(rows['a']['recall_drop_within_limit'])
    self.assertEqual(rows['c']['new_merges'], [9])
    self.assertEqual(r['seeds_outside_seed_rules'], ['a', 'c'])
    tol = r['overall']['all']['tolerance']
    self.assertIsNone(tol['merged_voxels_growth'])  # base had none: no Infinity
    self.assertEqual(tol['merged_voxels_base_and_candidate'], [0, 1200])
    self.assertFalse(tol['within_tolerance'])
    json.dumps(r, allow_nan=False)  # strict JSON

  def test_sensitive_only_and_zero_correct_runs_do_not_crash(self):
    recs = [_record('axon_218', 'axon', 0.0, 0.0, correct=1, mask_voxels=0)]
    recs[0]['correct_voxels'] = 0
    recs[0]['evals_per_1000_correct_voxels'] = None
    _write_run(self.root / 'x', recs, {})
    _write_run(self.root / 'y', recs, {})
    r = compare.compare_runs(self.root / 'x', self.root / 'y')
    self.assertIsNone(r['per_seed'][0]['delta_evals_per_1000_correct_voxels'])
    self.assertEqual(r['per_seed'][0]['mask_iou'], 1.0)  # both empty
    self.assertEqual(r['overall']['without_sensitive_seeds']['seeds'], 0)
    self.assertIsNone(r['overall']['without_sensitive_seeds']['tolerance'])

  def test_runs_with_different_inputs_or_seeds_are_rejected(self):
    _write_run(self.root / 'x', [_record('a', 'axon', 0.5, 1.0)], {'a': _box((0, 0, 0), (2, 2, 2))})
    _write_run(self.root / 'y', [_record('a', 'axon', 0.5, 1.0)], {'a': _box((0, 0, 0), (2, 2, 2))})
    r = compare.compare_runs(self.root / 'x', self.root / 'y')
    self.assertTrue(r['inputs_sha256_checked'])
    self.assertTrue(r['passes_tolerance'])
    run = json.loads((self.root / 'y' / 'benchmark.json').read_text())
    del run['inputs_sha256']
    (self.root / 'y' / 'benchmark.json').write_text(json.dumps(run))
    r = compare.compare_runs(self.root / 'x', self.root / 'y')
    self.assertFalse(r['inputs_sha256_checked'])
    self.assertFalse(r['passes_tolerance'])  # unverifiable inputs never pass
    for name, h in (('x', 'p'), ('y', 'q')):
      run = json.loads((self.root / name / 'benchmark.json').read_text())
      run['inputs_sha256'] = {'raw': h}
      (self.root / name / 'benchmark.json').write_text(json.dumps(run))
    with self.assertRaises(ValueError):
      compare.compare_runs(self.root / 'x', self.root / 'y')
    run = json.loads((self.root / 'y' / 'benchmark.json').read_text())
    run['inputs_sha256'] = {'raw': 'p'}
    run['seeds'][0]['seed_zyx'] = [2, 2, 2]
    (self.root / 'y' / 'benchmark.json').write_text(json.dumps(run))
    with self.assertRaises(ValueError):
      compare.compare_runs(self.root / 'x', self.root / 'y')

  def test_confirmed_tolerance_is_one_percent(self):
    self.assertEqual((compare.TOL_MEAN_RECALL_DROP, compare.TOL_MEAN_PRECISION_DROP,
                      compare.TOL_MEAN_UNLABELED_RISE, compare.TOL_MERGED_VOXELS_GROWTH,
                      compare.TOL_SEED_RECALL_DROP), (0.01,) * 5)

  def test_disjoint_runs_report_missing_seeds(self):
    _write_run(self.root / 'x', [_record('a', 'axon', 0.5, 1.0)], {'a': _box((0, 0, 0), (2, 2, 2))})
    _write_run(self.root / 'y', [_record('b', 'axon', 0.5, 1.0)], {'b': _box((0, 0, 0), (2, 2, 2))})
    r = compare.compare_runs(self.root / 'x', self.root / 'y')
    self.assertEqual((r['missing_in_candidate'], r['missing_in_base']), (['a'], ['b']))
    self.assertFalse(r['complete'])
    self.assertFalse(r['passes_tolerance'])
    self.assertIsNone(r['overall']['all']['tolerance'])  # no shared seeds

  def test_partial_run_never_passes_and_missing_masks_are_rejected(self):
    recs = [_record('a', 'axon', 0.8, 1.0), _record('b', 'axon', 0.8, 1.0)]
    masks = {'a': _box((0, 0, 0), (2, 2, 2)), 'b': _box((4, 4, 4), (6, 6, 6))}
    _write_run(self.root / 'full', recs, masks)
    _write_run(self.root / 'part', recs[:1], {'a': masks['a']})
    r = compare.compare_runs(self.root / 'full', self.root / 'part')
    self.assertTrue(r['overall']['all']['tolerance']['within_tolerance'])  # shared seed is fine
    self.assertFalse(r['passes_tolerance'])                                 # but b is missing
    _write_run(self.root / 'nomask', recs, {'a': masks['a']})               # b has voxels, no mask
    with self.assertRaises(ValueError):
      compare.compare_runs(self.root / 'full', self.root / 'nomask')

  def test_records_must_match_declared_seeds(self):
    recs = [_record('a', 'axon', 0.8, 1.0), _record('b', 'axon', 0.8, 1.0)]
    masks = {'a': _box((0, 0, 0), (2, 2, 2)), 'b': _box((4, 4, 4), (6, 6, 6))}
    _write_run(self.root / 'x', recs, masks)
    for name, edit in (('dropped', lambda run: run['records'].pop()),          # b declared, no record
                       ('moved', lambda run: run['records'][0].update(seed_zyx=[3, 3, 3]))):
      _write_run(self.root / name, recs, masks)
      run = json.loads((self.root / name / 'benchmark.json').read_text())
      edit(run)
      (self.root / name / 'benchmark.json').write_text(json.dumps(run))
      with self.assertRaises(ValueError, msg=name):
        compare.compare_runs(self.root / 'x', self.root / name)

  def test_exact_one_point_change_passes(self):
    base = [_record('a', 'axon', 0.80, 1.0)]
    cand = [_record('a', 'axon', 0.79, 1.0)]
    m = {'a': _box((0, 0, 0), (2, 2, 2))}
    _write_run(self.root / 'b', base, m)
    _write_run(self.root / 'c', cand, m)
    r = compare.compare_runs(self.root / 'b', self.root / 'c')
    self.assertTrue(r['per_seed'][0]['seed_tolerance']['within_seed_rules'])
    self.assertTrue(r['passes_tolerance'])


if __name__ == '__main__':
  unittest.main()
