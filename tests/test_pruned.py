"""Pruned network, input recorder and selection rule (CPU)."""

import unittest

import numpy as np

from liconn_v2 import data
from liconn_v2 import pruned
from liconn_v2 import reference


def _authors_apply(params):
  import jax  # pylint: disable=g-import-not-at-top
  from connectomics.jax.models import convstack  # pylint: disable=g-import-not-at-top
  model = convstack.ResConvStack(convstack.ConvstackConfig(depth=reference.DEPTH, padding='same',
                                                           use_layernorm=True))
  return jax.jit(lambda d: model.apply({'params': params}, d))


class PrunedNetworkTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    from clu import checkpoint  # pylint: disable=g-import-not-at-top
    cls.params = checkpoint.Checkpoint('').load_state(state=None, checkpoint=data.checkpoint_path())['params']
    rng = np.random.default_rng(0)
    cls.x = rng.normal(size=(1, *reference.FOV_SIZE, 2)).astype(np.float32)

  def test_no_drop_is_bitwise_the_authors_network(self):
    a = np.asarray(_authors_apply(self.params)(self.x))
    b = np.asarray(pruned.apply_fn(self.params, ())(self.x))
    self.assertEqual(a.tobytes(), b.tobytes())

  def test_dropped_block_is_the_identity(self):
    # Zeroing block 7's second conv makes it add exactly 0, i.e. the identity.
    zeroed = dict(self.params)
    zeroed['res7b'] = {k: np.zeros_like(v) for k, v in self.params['res7b'].items()}
    a = np.asarray(_authors_apply(zeroed)(self.x))
    b = np.asarray(pruned.apply_fn(self.params, (7,))(self.x))
    np.testing.assert_array_equal(a, b)
    c = np.asarray(pruned.apply_fn(self.params, ())(self.x))
    self.assertGreater(np.abs(b - c).max(), 0)  # and it does change the output

  def test_gflop_accounting(self):
    full = pruned.gflop_per_eval(self.params, ())
    self.assertAlmostEqual(full, 81.600564672, places=6)
    per_block = 2 * 2 * 27 * 32 * 32 * 33 ** 3 / 1e9
    self.assertAlmostEqual(full - pruned.gflop_per_eval(self.params, (3, 11)), 2 * per_block, places=6)
    self.assertEqual(set(self.params) - set(pruned.used_params(self.params, (3,))),
                     {'res3a', 'res3b', 'LayerNorm_4'})
    # Variant runs record the pruned cost, not the full network's.
    import inspect  # pylint: disable=g-import-not-at-top
    self.assertIn('gflop_per_eval(tracer.params, skip)', inspect.getsource(pruned.run_variant))


class SelectionTest(unittest.TestCase):

  def test_sequential_greedy_uses_combined_effect(self):
    # Blocks 1 and 2 are each harmless alone but bad together; 3 is mildly harmful.
    cost = {(): 0.0, (1,): 0.0001, (2,): 0.0001, (3,): 0.0005, (0,): 0.003,
            (1, 2): 0.05, (1, 3): 0.0006, (2, 3): 0.0006, (0, 1): 0.004}
    def measure(label, skip):
      return {'label': label, 'skip': list(skip), 'flip_rate': cost.get(tuple(skip), 0.1)}
    path = pruned.sequential_greedy(measure, depth=4, max_blocks=2)
    self.assertEqual([r['skip'] for r in path], [[1], [1, 3]])  # not [1, 2]

  def test_selection_rule(self):
    singles = [{'skip': [i], 'flip_rate': f} for i, f in enumerate([0.003, 0.0005, 0.02, 0.0005])]
    sets = [(1,), (1, 3), (0, 1, 3), (0, 1, 2, 3)]
    greedy = [{'skip': list(s), 'flip_rate': f} for s, f in zip(sets, [0.0005, 0.0009, 0.004, 0.03])]
    chosen = pruned.select_variants(singles, greedy)
    self.assertEqual(chosen, [('best_single', (1,)),
                              ('greedy_flip_le_0.001', (1, 3)),
                              ('greedy_flip_le_0.005', (0, 1, 3))])
    # No 2% entry: the only set above 0.5% is at 3%, so the 0.5% set is reused and deduplicated.


class _Canvas:

  def __init__(self):
    self.calls = []

  def _get_image(self, pos):
    return np.full((33, 33, 33), float(pos[0]), np.float32)

  def predict(self, pos, logit_seed):
    self.calls.append(pos)
    return logit_seed


class RecorderTest(unittest.TestCase):

  def test_records_every_nth_input_without_changing_calls(self):
    cv = _Canvas()
    rec = pruned.InputRecorder(every=2)
    rec.attach(cv)
    for i in range(5):
      out = cv.predict((i, 0, 0), np.full((33, 33, 33), i, np.float32))
      self.assertEqual(out[0, 0, 0], i)
    self.assertEqual(rec.steps, [0, 2, 4])
    self.assertEqual(len(rec.inputs), 3)
    self.assertEqual(rec.inputs[1].shape, (33, 33, 33, 2))
    self.assertEqual((rec.inputs[1][0, 0, 0, 0], rec.inputs[1][0, 0, 0, 1]), (2.0, 2.0))  # [image, seed]
    self.assertEqual(len(cv.calls), 5)


if __name__ == '__main__':
  unittest.main()
