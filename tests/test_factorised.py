"""PyTorch port of the authors' network and the factorised student (CPU, float32).

Tolerances are pre-registered in docs/factorised_design.md: maximum absolute logit difference 1e-3.
"""

import unittest

import numpy as np
import torch

from liconn_v2 import data
from liconn_v2 import factorised as fz
from liconn_v2 import reference

TOL = 1e-3


def to_torch(x_bzyxc):
  return torch.from_numpy(np.ascontiguousarray(np.transpose(x_bzyxc, (0, 4, 1, 2, 3))))


class PortTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    from clu import checkpoint  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top
    from connectomics.jax.models import convstack  # pylint: disable=g-import-not-at-top
    cls.params = checkpoint.Checkpoint('').load_state(state=None, checkpoint=data.checkpoint_path())['params']
    model = convstack.ResConvStack(convstack.ConvstackConfig(depth=reference.DEPTH, padding='same',
                                                             use_layernorm=True))
    cls.authors = staticmethod(jax.jit(lambda d: model.apply({'params': cls.params}, d)))
    cls.teacher = fz.teacher_from_params(cls.params)
    image = reference.normalize(data.load_raw_xyz())
    labels = np.transpose(data.load_proofread_xyz(), [2, 1, 0])
    centers = fz.check_centers(image.shape, 3) + [tuple(reference.SEED_XYZ[::-1])]
    cls.real = np.stack([fz.realistic_box(image, labels, c) for c in centers])  # (b, 2, z, y, x)
    cls.image = image

  def jax_logits(self, x_bczyx):
    return np.asarray(self.authors(np.transpose(x_bczyx, (0, 2, 3, 4, 1))))[..., 0]

  def torch_logits(self, model, x_bczyx):
    with torch.no_grad():
      return model(torch.from_numpy(x_bczyx)).numpy()[:, 0]

  def test_port_matches_authors_on_random_inputs(self):
    x = np.random.default_rng(0).normal(size=(2, 2, *reference.FOV_SIZE)).astype(np.float32)
    d = np.abs(self.torch_logits(self.teacher, x) - self.jax_logits(x))
    self.assertLess(d.max(), TOL)

  def test_port_matches_authors_on_real_boxes(self):
    d = np.abs(self.torch_logits(self.teacher, self.real) - self.jax_logits(self.real))
    self.assertLess(d.max(), TOL)
    self.assertTrue((self.real[:, 1] != 0).any() and (self.real[:, 1] == 0).any())  # mixed mask

  def test_port_matches_authors_on_recorded_trace_inputs(self):
    from liconn_v2 import pruned  # pylint: disable=g-import-not-at-top
    rec = pruned.InputRecorder(every=24)
    with reference.Tracer(data.checkpoint_path()) as tracer:
      trace = tracer.trace(self.image, reference.SEED_XYZ[::-1], instrument=rec.attach)
    self.assertEqual(reference.fingerprint(trace), reference.load_expected())  # the real reference trace
    x = np.transpose(np.stack(rec.inputs), (0, 4, 1, 2, 3))
    self.assertEqual(len(x), 15)
    d = np.abs(self.torch_logits(self.teacher, x) - self.jax_logits(x))
    self.assertLess(d.max(), TOL)

  def test_teacher_depth_and_names_come_from_the_checkpoint(self):
    self.assertEqual(len(self.teacher.blocks), reference.DEPTH)
    self.assertEqual(sum(p.numel() for p in self.teacher.parameters()),
                     sum(int(np.prod(np.shape(v))) for layer in self.params.values() for v in layer.values()))

  def test_student_at_initialisation_is_teacher_with_unknown_mask(self):
    s = fz.student_from_teacher(self.teacher, dilations=(1, 1, 1, 1)).eval()
    unknown = self.real.copy()
    unknown[:, 1] = 0.0
    d = np.abs(self.torch_logits(s, self.real) - self.torch_logits(self.teacher, unknown))
    self.assertLess(d.max(), TOL)  # the student ignores the mask until trained

  def test_dilated_student_runs_and_is_not_the_teacher(self):
    s = fz.student_from_teacher(self.teacher).eval()
    y = self.torch_logits(s, self.real)
    self.assertEqual(y.shape, self.real[:, 0].shape)
    self.assertTrue(np.isfinite(y).all())
    self.assertGreater(np.abs(y - self.torch_logits(self.teacher, self.real)).max(), TOL)

  def test_mask_stem_learns(self):
    s = fz.student_from_teacher(self.teacher)
    x = torch.from_numpy(self.real[:1])
    opt = torch.optim.SGD(s.decoder.parameters(), lr=1e-2)
    loss = s(x).square().mean()
    loss.backward()
    self.assertGreater(s.decoder.mask_b.weight.grad.abs().max().item(), 0)
    opt.step()
    with torch.no_grad():
      other = x.clone()
      other[:, 1] = 0.0
      self.assertGreater((s(x) - s(other)).abs().max().item(), 0)  # now the mask matters

  def test_depths_must_add_up(self):
    with self.assertRaises(ValueError):
      fz.student_from_teacher(self.teacher, encoder_depth=16, dilations=(1, 1))


class DenseEncodingTest(unittest.TestCase):
  """Tiling logic with a shallow encoder (radius 6); it depends on the depth only via the radius."""

  def setUp(self):
    torch.manual_seed(0)
    self.enc = fz.Encoder(depth=2).eval()
    self.vol = np.random.default_rng(1).normal(size=(40, 37, 45)).astype(np.float32)

  def test_radius(self):
    self.assertEqual(self.enc.radius, 6)
    self.assertEqual(fz.Encoder().radius, 34)

  def test_tiles_equal_one_pass(self):
    one = fz.encode_volume(self.enc, self.vol, tile=64)
    tiled = fz.encode_volume(self.enc, self.vol, tile=16)
    self.assertEqual(tuple(one.shape), (fz.FEATURES, *self.vol.shape))
    self.assertLess((one - tiled).abs().max().item(), TOL)

  def test_box_crop_equals_encoder_on_box_with_halo(self):
    feats = fz.encode_volume(self.enc, self.vol, tile=16)
    r, c, size = self.enc.radius, (20, 18, 22), (9, 9, 9)
    box = fz.box_features(feats, c, size)
    lo = [ci - s // 2 for ci, s in zip(c, size)]
    padded = np.pad(self.vol, r)  # zeros outside the volume, as in the definition
    sl = tuple(slice(l, l + s + 2 * r) for l, s in zip(lo, size))  # padded index l + r - r
    with torch.no_grad():
      direct = self.enc(torch.from_numpy(padded[sl])[None, None])[..., r:-r, r:-r, r:-r]
    self.assertLess((box - direct).abs().max().item(), TOL)


class CostTest(unittest.TestCase):

  def test_teacher_step_is_81_6_gflop(self):
    per_box = fz.flop_per_voxel()['teacher'] * int(np.prod(reference.FOV_SIZE))
    self.assertAlmostEqual(per_box / 1e9, 81.6, places=1)

  def test_encoder_halo_accounting(self):
    self.assertEqual(fz.encoder_input_voxels((500, 500, 500)), 568 ** 3)
    self.assertEqual(fz.encoder_input_voxels((500, 500, 500), tile=125), 64 * 193 ** 3)
    self.assertEqual(fz.encoder_input_voxels((10, 10, 10), tile=4, radius=1), (4 + 2 + 4 + 2 + 2 + 2) ** 3)

  def test_crop12_ratio(self):
    steps, shape = 137723, (500, 500, 500)  # docs/headtohead_results.md
    one = fz.crop_flop(steps, shape)
    self.assertAlmostEqual(one['teacher'] / 1e15, 11.2, places=1)
    self.assertAlmostEqual(one['decoder'] / 1e15, 2.47, places=2)
    self.assertAlmostEqual(one['encoder'] / 1e15, 0.33, places=2)  # 568^3 with the halo
    tiled = fz.crop_flop(steps, shape, tile=125)
    self.assertAlmostEqual(tiled['encoder'] / 1e15, 0.84, places=2)
    self.assertGreater(tiled['ratio'], 3.0)
    self.assertLess(tiled['ratio'], one['ratio'])

if __name__ == '__main__':
  unittest.main()
