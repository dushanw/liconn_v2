"""Lower-precision network variants (CPU)."""

import unittest

import numpy as np

from liconn_v2 import data
from liconn_v2 import precision
from liconn_v2 import reference


class PrecisionNetworkTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    from clu import checkpoint  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top
    from connectomics.jax.models import convstack  # pylint: disable=g-import-not-at-top
    cls.params = checkpoint.Checkpoint('').load_state(state=None, checkpoint=data.checkpoint_path())['params']
    model = convstack.ResConvStack(convstack.ConvstackConfig(depth=reference.DEPTH, padding='same',
                                                             use_layernorm=True))
    cls.authors = jax.jit(lambda d: model.apply({'params': cls.params}, d))
    rng = np.random.default_rng(0)
    cls.x = rng.normal(size=(1, *reference.FOV_SIZE, 2)).astype(np.float32)
    cls.ref = np.asarray(cls.authors(cls.x))

  def test_fp32_is_bitwise_the_authors_network(self):
    y = np.asarray(precision.apply_fn(self.params, 'fp32')(self.x))
    self.assertEqual(y.tobytes(), self.ref.tobytes())

  def test_low_precision_variants_run_and_stay_close(self):
    for v in ('fp16', 'fp16_ln32', 'bf16', 'bf16_ln32'):
      y = np.asarray(precision.apply_fn(self.params, v)(self.x))
      self.assertEqual(y.dtype, np.float32, v)
      self.assertEqual(y.shape, self.ref.shape, v)
      self.assertTrue(np.isfinite(y).all(), v)
      d = np.abs(y - self.ref)
      self.assertGreater(d.max(), 0, v)            # it really is a different number format
      self.assertLess(d.mean(), 0.5, v)            # but not a different network

  def test_unknown_variant_rejected(self):
    with self.assertRaises(ValueError):
      precision.apply_fn(self.params, 'int4')


class ScreenMetricsTest(unittest.TestCase):

  def test_non_finite_outputs_count_as_changed(self):
    import unittest.mock  # pylint: disable=g-import-not-at-top
    # Input i is filled with the value i, so the fake network can return output i.
    x = np.stack([np.full((2, 2, 2, 2), i, np.float32) for i in range(2)])
    outputs = {'fp32': np.array([[[[1.0], [-1.0]], [[1.0], [-1.0]]]] * 2, np.float32),
               'fp16': np.array([[[[np.inf], [-1.0]], [[1.0], [-1.0]]],
                                 [[[1.0], [-1.0]], [[1.0], [np.nan]]]], np.float32)}
    seen = []
    def fake_apply(params, v):
      def f(d):
        idx = [int(np.asarray(d)[k, 0, 0, 0, 0]) for k in range(len(d))]
        seen.extend(idx)
        return outputs[v][idx]
      return f
    with unittest.mock.patch.object(precision, 'apply_fn', fake_apply), \
         unittest.mock.patch.object(precision, 'time_per_eval', lambda f, x, r: 1.0):
      r = precision.screen(None, x, ['fp16'])['variants'][0]
    self.assertIn(1, seen)  # the NaN-containing output was actually evaluated
    self.assertAlmostEqual(r['non_finite_rate'], 2 / 8)  # one inf (input 0) + one NaN (input 1)
    self.assertEqual(r['inputs_with_non_finite'], 2)
    self.assertAlmostEqual(r['flip_rate'], 2 / 8)
    self.assertEqual(r['max_abs_logit_diff'], 0.0)
    import json  # pylint: disable=g-import-not-at-top
    json.dumps(r, allow_nan=False)


class SelectionTest(unittest.TestCase):

  def test_rule(self):
    res = [{'variant': 'fp16', 'flip_rate': 0.01}, {'variant': 'fp16_ln32', 'flip_rate': 0.001},
           {'variant': 'bf16', 'flip_rate': 0.004}, {'variant': 'bf16_ln32', 'flip_rate': 0.002}]
    self.assertEqual(precision.select_for_benchmark(res), ['fp16', 'bf16_ln32'])
    res[3]['flip_rate'] = 0.004  # tie: alphabetical
    self.assertEqual(precision.select_for_benchmark(res), ['fp16', 'bf16'])


if __name__ == '__main__':
  unittest.main()
