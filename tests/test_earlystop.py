"""Early-stop analysis (docs/earlystop_design.md) and the many-seed fetch fix."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from liconn_v2 import data
from liconn_v2 import earlystop
from liconn_v2 import manyseed
from liconn_v2 import precision
from liconn_v2 import reference

T = float(earlystop.move_threshold())
ROOT = Path(__file__).resolve().parent.parent


def step(pos, scores, rels=None):
  """A recorded step with six face maxima (scores) at FFN's face order."""
  out = []
  k = 0
  for axis in range(3):
    for offset in (-8, 8):
      rel = list(rels[k]) if rels else [0, 0]
      rel.insert(axis, offset)
      out.append((axis, offset, scores[k], tuple(rel), scores[k] >= T))
      k += 1
  return {'pos': list(pos), 'faces': out}


class FacesTest(unittest.TestCase):

  def test_faces_match_ffn_move_offsets_and_order(self):
    from ffn.inference import movement  # pylint: disable=g-import-not-at-top
    rng = np.random.default_rng(1)
    for _ in range(20):
      prob = rng.normal(0, 1, reference.FOV_SIZE).astype(np.float32)
      got = earlystop.faces(prob, T)
      ffn = sorted(movement.get_scored_move_offsets((8, 8, 8), prob, threshold=T), reverse=True)
      self.assertEqual(earlystop.queued(got), [tuple(int(v) for v in rel) for _, rel in ffn])
      # two faces can share an edge voxel; FFN then queues that move once
      self.assertEqual(sorted({(f[2], f[3]) for f in got if f[4]}),
                       sorted((float(s), tuple(int(v) for v in rel)) for s, rel in ffn))

  def test_seed_consistency_matches_ffn_update_at(self):
    logits = np.array([1.0, 1.0, -2.0, 0.5], np.float32)
    old = np.array([np.nan, -1.0, -3.0, 2.0], np.float32)
    got = earlystop.seed_consistency(logits, old, T)
    np.testing.assert_array_equal(got, np.array([1.0, -1.0, -3.0, 0.5], np.float32))
    np.testing.assert_array_equal(earlystop.seed_consistency(np.full(4, -5, np.float32), old, T),
                                  np.full(4, -5, np.float32))


class FirstDifferenceTest(unittest.TestCase):

  def test_threshold_argmax_order_and_none(self):
    a = [step((0, 0, 0), [1, -1, -1, -1, -1, -1]), step((8, 0, 0), [2, 1, -1, -1, -1, -1])]
    self.assertIsNone(earlystop.first_difference(a, a))
    b = [a[0], step((8, 0, 0), [2, T - 0.01, -1, -1, -1, -1])]
    d = earlystop.first_difference(a, b)
    self.assertEqual((d['step'], d['kind'], d['face']), (1, 'threshold', [0, 8]))
    self.assertAlmostEqual(d['margin'], 1 - T, places=6)
    self.assertTrue(d['ref_moves'] and not d['cand_moves'])
    c = [a[0], step((8, 0, 0), [2, 1, -1, -1, -1, -1], [(0, 0), (3, 1), (0, 0), (0, 0), (0, 0), (0, 0)])]
    self.assertEqual(earlystop.first_difference(a, c)['kind'], 'argmax')
    e = [a[0], step((8, 0, 0), [1, 2, -1, -1, -1, -1])]
    self.assertEqual(earlystop.first_difference(a, e)['kind'], 'order')

  def test_centers_visited_by_one_run_only_are_skipped(self):
    a = [step((0, 0, 0), [1, -1, -1, -1, -1, -1]), step((8, 0, 0), [1, -1, -1, -1, -1, -1])]
    b = [step((0, 0, 0), [1, -1, -1, -1, -1, -1]), step((0, 0, 8), [-1, -1, -1, -1, -1, -1])]
    self.assertIsNone(earlystop.first_difference(a, b))


class FrontierTest(unittest.TestCase):

  def _runs(self, cand_x_plus, cand_rel=(0, 0), cand_visits=False):
    up = [-1, -1, -1, -1, -1, 2]  # only the +x face (axis 2, +8) moves
    ref = [step((0, 0, 0), up), step((0, 0, 8), [-1] * 6)]
    cand = [step((0, 0, 0), up[:5] + [cand_x_plus], [(0, 0)] * 5 + [cand_rel])]
    if cand_visits:
      cand.append(step((0, 0, 8), [-1] * 6))
    return ref, cand

  def test_threshold_argmax_dropped_and_none(self):
    f = earlystop.frontier(*self._runs(T - 0.01))
    self.assertEqual((f['kind'], f['step'], f['parent_step'], f['cell'], f['face']), ('threshold', 1, 0, [0, 0, 1], [2, 8]))
    self.assertAlmostEqual(f['margin'], 2 - T, places=6)
    self.assertTrue(f['same_center'])
    self.assertEqual(earlystop.frontier(*self._runs(2, cand_rel=(8, 0)))['kind'], 'argmax')
    self.assertEqual(earlystop.frontier(*self._runs(2))['kind'], 'queued_not_evaluated')
    self.assertIsNone(earlystop.frontier(*self._runs(2, cand_visits=True)))

  def test_only_the_first_lost_cell_whose_parent_was_reached(self):
    ref = [step((0, 0, 0), [-1, -1, -1, -1, -1, 2]), step((0, 0, 8), [-1, -1, -1, -1, -1, 2]),
           step((0, 0, 16), [-1] * 6)]
    cand = [step((0, 0, 0), [-1, -1, -1, -1, -1, T - 0.5])]
    f = earlystop.frontier(ref, cand)
    self.assertEqual((f['step'], f['kind']), (1, 'threshold'))  # (0,0,16) is downstream of the loss

  def test_quantize_is_ffns(self):
    self.assertEqual(earlystop.quantize((3, -4, 12), (0, 0, 0)), (0, 0, 2))
    self.assertEqual(earlystop.quantize((4, -5, 11), (0, 0, 0)), (1, -1, 1))


class RoundTripTest(unittest.TestCase):

  def test_saved_steps_analyse_like_in_memory(self):
    a = [step((0, 0, 0), [1, -1, -1, -1, -1, 2]), step((0, 0, 8), [2, 1, -1, -1, -1, -1])]
    b = [a[0], step((0, 0, 8), [2, T - 0.01, -1, -1, -1, -1])]
    for x in a + b:
      x['open_loop'] = {'fp16': {'faces': x['faces'], 'voxel_flips': 0}}
    with tempfile.TemporaryDirectory() as d:
      earlystop.save_steps(Path(d) / 'a.json', a)
      earlystop.save_steps(Path(d) / 'b.json', b)
      la, lb = earlystop.load_steps(Path(d) / 'a.json'), earlystop.load_steps(Path(d) / 'b.json')
    self.assertEqual(earlystop.first_difference(la, lb), earlystop.first_difference(a, b))
    self.assertEqual(earlystop.frontier(la, lb), earlystop.frontier(a, b))
    self.assertEqual(earlystop.queued(la[0]['faces']), earlystop.queued(a[0]['faces']))
    self.assertEqual(earlystop.screen({'x': la}, 'fp16'), earlystop.screen({'x': a}, 'fp16'))


class BiasTest(unittest.TestCase):

  def test_near_threshold_and_bootstrap(self):
    errs = [(T + 0.01, -0.02), (T - 0.04, -0.01), (T + 0.2, 0.5)]
    np.testing.assert_array_equal(earlystop.near_threshold_errors(errs), [-0.02, -0.01])
    b = earlystop.bootstrap_mean({'a': [-0.02, -0.01], 'b': [-0.03], 'c': [-0.01, -0.01, -0.02]})
    self.assertAlmostEqual(b['mean'], -0.1 / 6)
    self.assertLess(b['ci95'][1], 0)
    self.assertLessEqual(b['ci95'][0], b['mean'])
    self.assertEqual(b, earlystop.bootstrap_mean({'a': [-0.02, -0.01], 'b': [-0.03], 'c': [-0.01, -0.01, -0.02]}))
    one = earlystop.bootstrap_mean({'a': [0.5, 0.5]})
    self.assertEqual(one['ci95'], [0.5, 0.5])

  def test_screen_and_losses(self):
    s = step((0, 0, 0), [1, 1, -1, -1, -1, -1])
    s['open_loop'] = {'fp16': {'faces': step((0, 0, 0), [1, T - 0.001, T + 0.001, -1, -1, -1])['faces'],
                               'voxel_flips': 3}}
    self.assertEqual(earlystop.losses_gains([s], 'fp16'), {'lost': 1, 'gained': 1})
    r = earlystop.screen({'x': [s, s]}, 'fp16')
    self.assertEqual(r['move_decision_flip_rate'], 1.0)
    self.assertEqual(r['voxel_flip_rate'], 6 / (2 * 33 ** 3))


class SelectionTest(unittest.TestCase):

  def test_controls(self):
    recs = {}
    for i, n in enumerate(earlystop.EARLY_STOP_SEEDS):
      crop = int(n[1:n.index('_')])
      recs[n] = {'name': n, 'class': n.split('_')[1], 'crop': crop, 'steps': 100 + i}
    recs['m1_axon_1'] = {'name': 'm1_axon_1', 'class': 'axon', 'crop': 1, 'steps': 101}
    recs['m1_axon_2'] = {'name': 'm1_axon_2', 'class': 'axon', 'crop': 1, 'steps': 300}
    recs['m9_dendrite_1'] = {'name': 'm9_dendrite_1', 'class': 'dendrite', 'crop': 9, 'steps': 102}
    recs['m4_axon_9'] = {'name': 'm4_axon_9', 'class': 'axon', 'crop': 4, 'steps': 1}
    verdict = {'flipped_seeds': {n: {} for n in earlystop.EARLY_STOP_SEEDS} | {'m4_axon_9': {}}}
    for k in range(20):
      recs[f'm0_axon_{100 + k}'] = {'name': f'm0_axon_{100 + k}', 'class': 'axon', 'crop': 0, 'steps': 50 + k}
      recs[f'm0_dendrite_{200 + k}'] = {'name': f'm0_dendrite_{200 + k}', 'class': 'dendrite', 'crop': 0,
                                        'steps': 500 + k}
    c = earlystop.select_controls(verdict, recs)
    self.assertEqual(len(c), 10)
    self.assertEqual(len(set(c)), 10)
    self.assertEqual(c[:2], ['m1_axon_1', 'm1_axon_2'])  # same crop first, closest steps
    self.assertEqual(c[2], 'm9_dendrite_1')  # no dendrite in crop 2: any crop
    self.assertNotIn('m4_axon_9', c)  # flagged seeds are never controls

  def test_early_stop_seeds_are_the_recorded_ones(self):
    ev = json.loads((ROOT / 'docs' / 'manyseed_evidence.json').read_text())
    self.assertEqual(sorted(earlystop.EARLY_STOP_SEEDS),
                     sorted(n for n, e in ev['documented']['flipped'].items() if 'new_early_stop' in e['events']))

  def test_candidate_selection(self):
    screens = {'fp16': {'move_decision_flip_rate': 0.01}, 'fp16_head32': {'move_decision_flip_rate': 0.02},
               'fp16_res32': {'move_decision_flip_rate': 0.005}}
    self.assertEqual(earlystop.select_candidate(screens, {'fp16': 3, 'fp16_head32': 2, 'fp16_res32': 2}), 'fp16_res32')
    # fp16_head32 is fast enough but worse than plain float16: not selected
    self.assertIsNone(earlystop.select_candidate(screens, {'fp16': 3, 'fp16_head32': 2, 'fp16_res32': 1.4}))
    screens['fp16_head32']['move_decision_flip_rate'] = 0.008
    self.assertEqual(earlystop.select_candidate(screens, {'fp16': 3, 'fp16_head32': 2, 'fp16_res32': 1.4}), 'fp16_head32')


class LayoutTest(unittest.TestCase):

  def test_all_float32_layout_is_the_authors_network(self):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    x = jnp.asarray(np.random.default_rng(2).normal(size=(1, 33, 33, 33, 2)).astype(np.float32))
    params = precision._module('fp32').init(jax.random.PRNGKey(0), x)['params']  # pylint: disable=protected-access
    ref = np.asarray(precision.apply_fn(params, 'fp32')(x))
    np.testing.assert_array_equal(np.asarray(earlystop.apply_fn(params, 'fp32_check')(x)), ref)
    for name in ('fp16_head32', 'fp16_res32'):
      y = np.asarray(earlystop.apply_fn(params, name)(x))
      self.assertEqual(y.dtype, np.float32)
      self.assertFalse(np.array_equal(y, ref), name)
      self.assertLess(np.abs(y - ref).max(), 0.5, name)
    shifted = np.asarray(earlystop.shifted(precision.apply_fn(params, 'fp16'), 0.25)(x))
    np.testing.assert_allclose(shifted, np.asarray(precision.apply_fn(params, 'fp16')(x)) + 0.25, rtol=0, atol=1e-6)


class FetchPinsTest(unittest.TestCase):
  """Re-running the fetch must not re-pin changed cached files (many-seed review finding)."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.tmp)
    shutil.copy(manyseed.cache_dir() / manyseed.COARSE_FILE, self.tmp / manyseed.COARSE_FILE)
    self.manifest = self.tmp / 'crops.json'
    self.seeds = self.tmp / 'seeds.json'
    shutil.copy(manyseed.MANIFEST, self.manifest)
    self.before = self.manifest.read_bytes()
    patches = [mock.patch.object(manyseed, 'cache_dir', lambda: self.tmp),
               mock.patch.object(manyseed, 'MANIFEST', self.manifest),
               mock.patch.object(manyseed, 'SEEDS_FILE', self.seeds),
               mock.patch.object(manyseed, '_open', side_effect=AssertionError('no network in tests'))]
    for p in patches:
      p.start()
      self.addCleanup(p.stop)

  def test_changed_coarse_level_is_refused_before_any_download(self):
    np.save(self.tmp / manyseed.COARSE_FILE, np.zeros((240, 305, 50), np.uint8))
    with self.assertRaisesRegex(ValueError, 'pinned'):
      manyseed.fetch_and_select()
    self.assertEqual(self.manifest.read_bytes(), self.before)

  def test_changed_cached_crop_is_refused_and_not_repinned(self):
    d = self.tmp / 'crop-00'
    d.mkdir()
    np.save(d / manyseed.RAW_FILE, np.zeros((2, 2, 2), np.uint8))
    np.save(d / manyseed.PROOFREAD_FILE, np.zeros((2, 2, 2), np.uint32))
    with self.assertRaisesRegex(ValueError, 'pinned'):  # before any connection (_open raises)
      manyseed.fetch_and_select()
    self.assertEqual(self.manifest.read_bytes(), self.before)
    self.assertFalse(self.seeds.exists())

  def test_changed_cached_crop_is_refused_before_downloading_a_missing_coarse_level(self):
    (self.tmp / manyseed.COARSE_FILE).unlink()
    d = self.tmp / 'crop-00'
    d.mkdir()
    np.save(d / manyseed.RAW_FILE, np.zeros((2, 2, 2), np.uint8))
    with self.assertRaisesRegex(ValueError, 'pinned'):  # _open would raise AssertionError
      manyseed.fetch_and_select()
    self.assertFalse((self.tmp / manyseed.COARSE_FILE).exists())

  def test_pinned_data_are_unchanged(self):
    self.assertEqual(data.sha256_file(manyseed.cache_dir() / manyseed.COARSE_FILE),
                     json.loads(self.before)['coarse_sha256'])


if __name__ == '__main__':
  unittest.main()
