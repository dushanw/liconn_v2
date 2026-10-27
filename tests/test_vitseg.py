"""The single-pass ViT segmentation pilot (liconn_v2/vitseg.py, docs/vitseg_design.md)."""

import unittest

import numpy as np

from liconn_v2 import benchmark
from liconn_v2 import manyseed
from liconn_v2 import vitseg


def two_slabs(shape=(24, 24, 24), gap=4):
  """Labels 1 and 2 as two slabs along x, separated by an unlabelled gap."""
  lab = np.zeros(shape, np.uint32)
  mid = shape[2] // 2
  lab[:, :, :mid - gap // 2] = 1
  lab[:, :, mid + gap // 2:] = 2
  return lab


def perfect(lab):
  """Perfect network output: the training targets (6 affinities, neuron channel), 0 where masked."""
  t, m = vitseg.training_targets(lab)
  return np.where(m > 0, t, 0.0).astype(np.float32)


def record(recall, precision, mask_voxels=1):
  return {'recall': recall, 'precision': precision, 'mask_voxels': mask_voxels}


class TargetTest(unittest.TestCase):

  def test_targets_and_masks_by_hand(self):
    lab = np.array([[[1, 1, 0, 0, 2, 2]]], np.uint32)  # (z, y, x) = (1, 1, 6)
    t, m = vitseg.affinity_targets(lab, offsets=((0, 0, 1),))
    # pairs (v, v-1) for x = 1..5: (1,1) (0,1) (0,0) (2,0) (2,2); x = 0 has no partner
    np.testing.assert_array_equal(t[0, 0, :, 0], [0, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(m[0, 0, :, 0], [0, 1, 1, 0, 1, 1])

  def test_long_range_offset_and_shape(self):
    lab = two_slabs()
    t, m = vitseg.affinity_targets(lab)
    self.assertEqual(t.shape, lab.shape + (6,))
    c = vitseg.OFFSETS.index((0, 0, 4))
    self.assertEqual(m[:, :, :4, c].sum(), 0)  # no partner inside the block
    self.assertEqual(t[0, 0, 4, c], 1)  # x = 4 and x = 0 both label 1
    self.assertEqual(t[0, 0, 17, c], 0)  # x = 17 (label 2) vs x = 13 (gap): a boundary
    self.assertEqual(m[0, 0, 17, c], 1)

  def test_training_targets_add_the_neuron_channel(self):
    lab = two_slabs()
    t, m = vitseg.training_targets(lab)
    a, am = vitseg.affinity_targets(lab)
    self.assertEqual(t.shape, lab.shape + (vitseg.CHANNELS,))
    np.testing.assert_array_equal(t[..., :6], a)
    # affinities are trained only between two neuron voxels: neuron-background pairs are masked
    x = vitseg.OFFSETS.index((0, 0, 1))
    self.assertEqual((am[0, 0, 10, x], m[0, 0, 10, x]), (1, 0))  # x = 10 (gap) vs x = 9 (label 1)
    self.assertEqual((m[0, 0, 5, x], t[0, 0, 5, x]), (1, 1))  # inside label 1
    touching = two_slabs(gap=0)
    tt, tm = vitseg.training_targets(touching)
    self.assertEqual((tm[0, 0, 12, x], tt[0, 0, 12, x]), (1, 0))  # label 2 vs label 1: trained, 0
    np.testing.assert_array_equal(t[..., vitseg.NEURON_CHANNEL], lab > 0)  # label 0 is background
    np.testing.assert_array_equal(m[..., vitseg.NEURON_CHANNEL], 1)  # every voxel has a target

  def test_balanced_loss_gives_each_class_equal_weight(self):
    t = np.zeros((1, 2, 2, 2, 1), np.float32)
    t[0, 0, 0, 0, 0] = 1.0  # 1 positive, 7 negatives
    m = np.ones_like(t)
    logits = np.zeros_like(t)
    # every voxel has BCE log 2; positives and negatives each carry weight 0.5
    self.assertAlmostEqual(float(vitseg.balanced_bce(logits, t, m)), np.log(2), places=5)
    m[0, 1] = 0.0  # masked voxels do not count
    self.assertAlmostEqual(float(vitseg.balanced_bce(logits, t, m)), np.log(2), places=5)


class TilingTest(unittest.TestCase):

  def test_every_voxel_owned_once(self):
    for n in (128, 129, 200, 500):
      own = vitseg.tile_ownership(n)
      covered = np.zeros(n, int)
      for start, lo, hi in own:
        self.assertTrue(start <= lo < hi <= start + vitseg.BLOCK)
        covered[lo:hi] += 1
      np.testing.assert_array_equal(covered, 1)
    self.assertEqual([s for s, _, _ in vitseg.tile_ownership(500)], [0, 96, 192, 288, 372])

  def test_reassembly_is_exact(self):
    rng = np.random.default_rng(0)
    image = rng.normal(size=(40, 50, 45)).astype(np.float32)
    got = vitseg.predict_affinities(lambda x: np.concatenate([x, -x], -1), image, block=16, stride=12)
    from scipy.special import expit  # pylint: disable=g-import-not-at-top
    np.testing.assert_array_equal(got[..., 0], expit(image))
    np.testing.assert_array_equal(got[..., 1], expit(-image))


class SegmentationTest(unittest.TestCase):

  def test_perfect_affinities_recover_objects(self):
    lab = two_slabs()
    aff = perfect(lab)
    frag = vitseg.fragments(aff)
    merges = vitseg.agglomerate(*vitseg.fragment_edges(frag, aff))
    seg = vitseg.segmentation_at(frag, merges, 0.5)
    recs = vitseg.seed_records(seg, lab, [{'seed_zyx': [12, 12, 3]}, {'seed_zyx': [12, 12, 20]}])
    for r in recs:
      self.assertEqual((r['recall'], r['precision'], r['unlabeled_fraction']), (1.0, 1.0, 0.0))
    self.assertNotEqual(seg[12, 12, 3], seg[12, 12, 20])
    np.testing.assert_array_equal(seg == 0, lab == 0)  # the gap stays background

  def test_boundary_map_treats_both_faces_alike(self):
    touching = two_slabs(gap=0)  # label 1 for x < 12, label 2 for x >= 12
    b = vitseg.boundary_map(perfect(touching))
    self.assertEqual(b[12, 12, 5], 0.0)  # interior
    self.assertEqual(float(b[12, 12, 11]), 1.0)  # last voxel of slab 1: its lowest edge is 0
    self.assertEqual(float(b[12, 12, 12]), 1.0)  # first voxel of slab 2
    # across a background gap the neuron channel does the work: edges to background voxels
    # do not count, so the slab faces are not boundaries (label 1 x < 10, gap 10..13)
    b = vitseg.boundary_map(perfect(two_slabs()))
    self.assertEqual((float(b[12, 12, 9]), float(b[12, 12, 14])), (0.0, 0.0))
    # without the neuron channel every voxel counts as neuron (the six affinities only)
    t, m = vitseg.affinity_targets(two_slabs())
    b = vitseg.boundary_map(np.where(m > 0, t, 0.0).astype(np.float32))
    self.assertEqual(float(b[12, 12, 9]), 1.0)

  def test_thin_rod_gets_its_own_segment(self):
    lab = np.zeros((24, 24, 24), np.uint32)
    lab[:, :, :10] = 1
    lab[:, 10:12, 12:14] = 2  # a 2 x 2 voxel rod along z, 2 voxels from the block
    aff = perfect(lab)
    frag = vitseg.fragments(aff)
    merges = vitseg.agglomerate(*vitseg.fragment_edges(frag, aff))
    seg = vitseg.segmentation_at(frag, merges, 0.5)
    recs = vitseg.seed_records(seg, lab, [{'seed_zyx': [12, 10, 12]}, {'seed_zyx': [12, 12, 4]}])
    for r in recs:
      self.assertEqual((r['recall'], r['precision']), (1.0, 1.0))

  def test_agglomeration_merges_at_the_threshold(self):
    # three fragments in a row; boundary affinities 0.8 (1-2) and 0.3 (2-3)
    frag = np.zeros((4, 4, 12), np.uint32)
    frag[..., :4], frag[..., 4:8], frag[..., 8:] = 1, 2, 3
    aff = np.ones(frag.shape + (6,), np.float32)
    aff[..., 4, 2] = 0.8  # pair (x=4, x=3): fragments 2 and 1
    aff[..., 8, 2] = 0.3  # pair (x=8, x=7): fragments 3 and 2
    a, b, s, c = vitseg.fragment_edges(frag, aff)
    self.assertEqual(list(zip(a.tolist(), b.tolist())), [(1, 2), (2, 3)])
    np.testing.assert_allclose(s / c, [0.8, 0.3])
    merges = vitseg.agglomerate(a, b, s, c)
    self.assertEqual([round(m[0], 6) for m in merges], [0.8, 0.3])
    for t, n_segments in ((0.9, 3), (0.8, 2), (0.5, 2), (0.3, 1), (0.1, 1)):
      self.assertEqual(len(np.unique(vitseg.segmentation_at(frag, merges, t))), n_segments, t)

  def test_contacts_through_background_do_not_count(self):
    frag = np.zeros((6, 6, 12), np.uint32)
    frag[..., :6], frag[..., 6:] = 1, 2
    aff = np.ones(frag.shape + (vitseg.CHANNELS,), np.float32)
    aff[..., 5:7, vitseg.NEURON_CHANNEL] = 0.0  # predicted background at the contact
    a, b, s, c = vitseg.fragment_edges(frag, aff)
    self.assertEqual(len(a), 0)
    self.assertEqual(vitseg.agglomerate(a, b, s, c), [])
    aff[..., 5:7, vitseg.NEURON_CHANNEL] = 1.0  # neuron voxels: low crossing edges still count
    aff[..., 6, 2] = 0.2
    a, b, s, c = vitseg.fragment_edges(frag, aff)
    np.testing.assert_allclose(s / c, [0.2])

  def test_touching_neurons_get_separate_fragments(self):
    lab = two_slabs(gap=0)  # no background between the two neurons
    aff = perfect(lab)
    frag = vitseg.fragments(aff)
    self.assertFalse(set(np.unique(frag[lab == 1])) & set(np.unique(frag[lab == 2])))
    merges = vitseg.agglomerate(*vitseg.fragment_edges(frag, aff))
    seg = vitseg.segmentation_at(frag, merges, 0.5)
    for r in vitseg.seed_records(seg, lab, [{'seed_zyx': [12, 12, 3]}, {'seed_zyx': [12, 12, 20]}]):
      self.assertEqual((r['recall'], r['precision']), (1.0, 1.0))

  def test_background_stays_zero_and_every_neuron_voxel_is_reached(self):
    lab = np.zeros((16, 24, 24), np.uint32)
    lab[:, 2:10, 2:10] = 1
    lab[:, 14:22, 2:22] = 2
    lab[:, 4:6, 14:16] = 3  # a thin rod
    aff = perfect(lab)
    # piece 4: neuron voxels whose own affinities are all low, so none is confident
    lab[3:5, 2:4, 18:20] = 4
    aff[3:5, 2:4, 18:20, :6] = 0.0
    aff[3:5, 2:4, 18:20, vitseg.NEURON_CHANNEL] = 1.0
    frag = vitseg.fragments(aff)
    np.testing.assert_array_equal(frag == 0, lab == 0)
    self.assertEqual(len(np.unique(frag[lab == 4])), 1)  # one fragment for the unmarked piece
    for i in (1, 2, 3):
      self.assertNotIn(int(frag[lab == i][0]), np.unique(frag[(lab != i) & (lab > 0)]).tolist())

  def test_mean_affinity_pools_after_a_merge(self):
    # 1-2 strong; 1-3 at 0.2 over 1 pair and 2-3 at 0.9 over 1 pair: after 1+2 merge,
    # the pooled 12-3 edge is (0.2 + 0.9) / 2 = 0.55
    a, b = np.array([1, 1, 2]), np.array([2, 3, 3])
    sums, counts = np.array([0.95, 0.2, 0.9]), np.array([1, 1, 1])
    merges = vitseg.agglomerate(a, b, sums, counts)
    self.assertEqual([round(m[0], 6) for m in merges], [0.95, 0.55])

  def test_seed_records_match_benchmark_score(self):
    rng = np.random.default_rng(1)
    gt = rng.integers(0, 4, size=(20, 20, 20)).astype(np.uint32)
    seg = rng.integers(1, 4, size=gt.shape)  # 0 is predicted background, tested below
    seeds = []
    for z in range(0, 20, 3):
      pos = (z, z % 7, (3 * z) % 11)
      if gt[pos]:
        seeds.append({'name': str(z), 'seed_zyx': list(pos)})
    self.assertGreater(len(seeds), 2)
    for s, r in zip(seeds, vitseg.seed_records(seg, gt, seeds)):
      want = benchmark.score(seg == seg[tuple(s['seed_zyx'])], gt, s['seed_zyx'])
      for k, v in want.items():
        self.assertEqual(r[k], v, k)
    # a seed in predicted background scores as an empty mask, not as a "background segment"
    seg[tuple(seeds[0]['seed_zyx'])] = 0
    r = vitseg.seed_records(seg, gt, seeds[:1])[0]
    want = benchmark.score(np.zeros(gt.shape, bool), gt, seeds[0]['seed_zyx'])
    for k, v in want.items():
      self.assertEqual(r[k], v, k)
    self.assertEqual((r['recall'], r['mask_voxels']), (0.0, 0))

  def test_background_scores(self):
    gt = np.array([[[0, 0, 1, 1, 2, 0]]], np.uint32)
    seg = np.array([[[0, 5, 5, 5, 0, 0]]], np.int64)
    s = vitseg.background_scores(seg, gt)
    self.assertEqual((s['precision'], s['recall']), (2 / 3, 2 / 3))
    self.assertEqual((s['label0_fraction'], s['predicted_fraction']), (0.5, 0.5))


class ModelTest(unittest.TestCase):

  def test_output_shape_and_training_lowers_the_loss(self):
    import jax  # pylint: disable=g-import-not-at-top
    tiny = {'block': 16, 'patch': 4, 'width': 16, 'depth': 1, 'heads': 2, 'encoder': (4, 8)}
    model = vitseg.make_model(**tiny)
    lab = two_slabs((16, 16, 16), gap=2)
    image = np.where(lab > 0, 1.0, -1.0).astype(np.float32)[None, ..., None]
    t, m = vitseg.training_targets(lab)
    params = model.init(jax.random.PRNGKey(0), image)['params']
    self.assertEqual(model.apply({'params': params}, image).shape, (1, 16, 16, 16, vitseg.CHANNELS))
    import optax  # pylint: disable=g-import-not-at-top
    opt = optax.adam(3e-3)  # the step's mechanics, not the pilot's schedule
    state = opt.init(params)
    step = vitseg.make_train_step(model, opt)
    losses = []
    for _ in range(100):
      params, state, loss, finite = step(params, state, image, t[None], m[None])
      self.assertTrue(bool(finite))
      losses.append(float(loss))
    self.assertLess(np.mean(losses[-5:]), 0.6 * np.mean(losses[:5]))

  def test_token_stream_is_normalised(self):
    import jax  # pylint: disable=g-import-not-at-top
    from flax import traverse_util  # pylint: disable=g-import-not-at-top
    tiny = {'block': 16, 'patch': 4, 'width': 16, 'depth': 2, 'heads': 2, 'encoder': (4, 8)}
    model = vitseg.make_model(**tiny)
    params = model.init(jax.random.PRNGKey(0), np.zeros((1, 16, 16, 16, 1), np.float32))['params']
    big = np.full((1, 16, 16, 16, 1), 1e3, np.float32)  # far outside the normalised input
    big[0, ::2] = -1e3
    _, inter = model.apply({'params': params}, big, capture_intermediates=True, mutable=['intermediates'])
    flat = traverse_util.flatten_dict(inter['intermediates'], sep='/')
    norms = [k for k in flat if k.split('/')[0].startswith('LayerNorm')]
    # 4 encoder convs, 2 decoder levels x (up-convolution + 2 convs), 2 on the token stream
    self.assertEqual(len(norms), 4 + 2 * 3 + 2)
    for k, v in flat.items():
      v = np.asarray(v[0], np.float32)
      self.assertTrue(np.isfinite(v).all(), k)
      if k in norms:
        self.assertLess(float(np.abs(v).max()), 100.0, k)

  def test_optimizer_clips_the_gradient_norm(self):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    import optax  # pylint: disable=g-import-not-at-top
    opt = vitseg.make_optimizer(1000)
    params = {'w': jnp.zeros(4)}
    state = opt.init(params)
    _, state = opt.update({'w': jnp.full(4, 50.0)}, state, params)  # norm 100
    mu = optax.tree_utils.tree_get(state, 'mu')['w']
    b1 = 0.9  # optax.adamw default
    np.testing.assert_allclose(np.asarray(mu), (1 - b1) * np.full(4, 0.5), rtol=1e-6)  # clipped to norm 1

  def test_training_stops_on_a_run_of_skips_or_a_bad_validation(self):
    self.assertIsNone(vitseg.stop_reason(vitseg.MAX_CONSECUTIVE_SKIPS - 1, 0.3))
    self.assertIn('consecutive', vitseg.stop_reason(vitseg.MAX_CONSECUTIVE_SKIPS))
    self.assertEqual(vitseg.stop_reason(0, float('nan')), 'non-finite validation loss')
    self.assertEqual(vitseg.MAX_CONSECUTIVE_SKIPS, 200)

  def test_batches_depend_only_on_step(self):
    lab = two_slabs((40, 40, 40))
    image = np.random.default_rng(0).normal(size=lab.shape).astype(np.float32)
    a = vitseg.sample_batch([image], [lab], 7, block=16)
    b = vitseg.sample_batch([image], [lab], 7, block=16)
    c = vitseg.sample_batch([image], [lab], 8, block=16)
    for x, y in zip(a, b):
      np.testing.assert_array_equal(x, y)
    self.assertFalse(np.array_equal(a[0], c[0]))
    self.assertEqual(a[0].shape, (vitseg.BATCH, 16, 16, 16, 1))


class VerdictTest(unittest.TestCase):

  def pairs(self, n=300, merges=0, avoided=0, stops=0, extensions=0, drop=0.0):
    out = []
    for i in range(n):
      ref, cand = record(0.8, 0.8), record(0.8 - drop, 0.8 - drop)
      if i < merges:
        cand = record(0.8 - drop, 0.5)
      elif i < merges + avoided:
        ref = record(0.8, 0.5)
      elif i < merges + avoided + stops:
        cand = record(0.5, 0.8 - drop)
      elif i < merges + avoided + stops + extensions:
        ref = record(0.5, 0.8)
      out.append((ref, cand))
    return out

  def test_tiers_at_their_limits(self):
    self.assertEqual((vitseg.COMPETITIVE, vitseg.PROMISING), ((0.02, 6), (0.05, 15)))
    self.assertEqual(vitseg.verdict(self.pairs(merges=6, stops=6))['verdict'], 'COMPETITIVE')
    self.assertEqual(vitseg.verdict(self.pairs(merges=7))['verdict'], 'PROMISING')
    self.assertEqual(vitseg.verdict(self.pairs(merges=15, stops=15))['verdict'], 'PROMISING')
    self.assertEqual(vitseg.verdict(self.pairs(stops=16))['verdict'], 'FAIL')

  def test_flips_are_netted(self):
    v = vitseg.verdict(self.pairs(merges=10, avoided=4, stops=9, extensions=3))
    self.assertEqual((v['net_new_merges'], v['net_new_early_stops']), (6, 6))

  def test_mean_drops(self):
    self.assertEqual(vitseg.verdict(self.pairs(drop=0.02))['verdict'], 'COMPETITIVE')
    self.assertEqual(vitseg.verdict(self.pairs(drop=0.021))['verdict'], 'PROMISING')
    self.assertEqual(vitseg.verdict(self.pairs(drop=0.051))['verdict'], 'FAIL')

  def test_events_are_the_many_seed_definition(self):
    self.assertEqual(manyseed.FLIP, 0.10)
    v = vitseg.verdict([(record(0.8, 0.8), record(0.8, 0.7))])  # exactly 10 points: no flip
    self.assertEqual(v['new_merges'], 0)

  def test_empty_subset(self):
    self.assertEqual(vitseg.verdict([]), {'seeds': 0, 'verdict': None})


class DesignTest(unittest.TestCase):

  def test_split_never_touches_test_or_benchmark_crops(self):
    self.assertEqual(vitseg.TRAIN_CROPS, tuple(range(12)))
    self.assertEqual(vitseg.TUNE_CROP, 12)
    from liconn_v2 import fresh  # pylint: disable=g-import-not-at-top
    test = {c['index'] for c in fresh.load_manifest()['crops']}
    self.assertFalse(test & (set(vitseg.TRAIN_CROPS) | {vitseg.TUNE_CROP}))


if __name__ == '__main__':
  unittest.main()
