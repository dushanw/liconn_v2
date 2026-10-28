"""Fast tracer (liconn_v2/fasttrace.py, docs/fasttrace_design.md), on CPU."""

import unittest

import numpy as np

from liconn_v2 import data
from liconn_v2 import fasttrace
from liconn_v2 import reference


class FastTracerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.tracer = reference.Tracer(data.checkpoint_path())
    cls.fast = fasttrace.FastTracer(cls.tracer)

  @classmethod
  def tearDownClass(cls):
    cls.tracer.close()

  def _canvas(self, raw):
    from ffn.inference import inference  # pylint: disable=g-import-not-at-top
    from ffn.inference import inference_utils  # pylint: disable=g-import-not-at-top
    return inference.Canvas(self.tracer.model_info, self.tracer._exc.get_client(inference_utils.Counters()),  # pylint: disable=protected-access
                            raw, self.tracer.options, voxel_size_zyx=reference.VOXEL_SIZE_ZYX)

  def test_options_are_canvas_options(self):
    raw = np.zeros((40, 40, 40), np.float32)
    cv = self._canvas(raw)
    for attr in fasttrace.LOGIT_OPTIONS + ('disco_seed_threshold',):
      self.assertEqual(getattr(self.fast.opts, attr), getattr(cv.options, attr), attr)
    self.assertEqual(cv.options.disco_seed_threshold, 0.0)  # the consistency step the device code implements
    cv._deregister_client()  # pylint: disable=protected-access

  def test_assemble_and_update_match_canvas_update_at(self):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    assemble, update, value, _, _ = self.fast._fns  # pylint: disable=protected-access
    rng = np.random.default_rng(3)
    shape, corner = (50, 52, 54), np.array([5, 7, 9], np.int32)
    seed = rng.normal(0, 2, shape).astype(np.float32)
    seed[rng.random(shape) < 0.3] = np.nan
    image = rng.normal(0, 1, shape).astype(np.float32)
    sel = tuple(slice(c, c + 33) for c in corner)
    # input, as FFN's update_at and executor build it
    logit_seed = np.array(seed[sel])
    logit_seed[np.isnan(logit_seed)] = np.float32(self.fast.opts.pad_value)
    want_in = np.concatenate([image[sel][None, ..., None], logit_seed[None, ..., None]], axis=-1)
    np.testing.assert_array_equal(np.asarray(assemble(jnp.asarray(seed), jnp.asarray(image), corner)), want_in)
    # update, as Canvas.update_at does after prediction (disco_seed_threshold = 0)
    for scale in (2.0, -5.0):  # some voxels above the move threshold, or none
      out = (rng.normal(0, 1, (1, 33, 33, 33, 1)) * abs(scale) + (0 if scale > 0 else -10)).astype(np.float32)
      logits = np.array(out[0, ..., 0])
      old = seed[sel]
      if np.mean(logits >= self.fast.opts.move_threshold) > 0:
        with np.errstate(invalid='ignore'):
          mask = (old < 0.0) & (logits > old)
        logits[mask] = old[mask]
      want = seed.copy()
      want[sel] = logits
      new, _, _ = update(jnp.asarray(seed), jnp.asarray(out), corner)
      np.testing.assert_array_equal(np.asarray(new), want)

  def test_device_face_maxima_give_ffns_queue(self):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    from ffn.inference import movement  # pylint: disable=g-import-not-at-top
    _, update, _, _, _ = self.fast._fns  # pylint: disable=protected-access
    rng = np.random.default_rng(4)

    class View:
      def is_valid_pos(self, pos):
        return True

    view = View()
    t = self.fast.opts.move_threshold
    cases = []
    for _ in range(40):
      prob = rng.normal(t, 2, (33, 33, 33)).astype(np.float32)
      cases.append(prob)
    tie = np.full((33, 33, 33), 2.0, np.float32)  # every face ties; shared edges repeat (score, position)
    cases.append(tie)
    nan = rng.normal(t, 2, (33, 33, 33)).astype(np.float32)
    nan[8, 10, 12] = np.nan  # a NaN maximum on one face (FFN keeps it as a move)
    nan[20, 24, 9] = np.nan
    cases.append(nan)
    low = np.full((33, 33, 33), -3.0, np.float32)  # no face reaches the threshold
    cases.append(low)
    for prob in cases:
      pos = (100, 100, 100)
      a = movement.FaceMaxMovementPolicy(view, deltas=self.fast.deltas, score_threshold=t)
      b = movement.FaceMaxMovementPolicy(view, deltas=self.fast.deltas, score_threshold=t)
      a.reset_state((92, 100, 108))
      b.reset_state((92, 100, 108))
      a.update(prob, pos)
      seed = jnp.full((40, 40, 40), jnp.nan, jnp.float32)  # all NaN: the consistency step changes nothing
      _, scores, positions = update(seed, jnp.asarray(prob[None, ..., None]), np.zeros(3, np.int32))
      fasttrace.queue_moves(b, np.asarray(scores), np.asarray(positions).astype(np.int64), pos, self.fast.deltas)
      key = lambda q: [(None if np.isnan(sc) else float(sc), [int(v) for v in c]) for sc, c in q.scored_coords]
      self.assertEqual(key(a), key(b))
      self.assertEqual(a.done_rounded_coords, b.done_rounded_coords)

  def test_first_max_is_numpy_argmax_with_nans_and_ties(self):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    first_max = jax.jit(self.fast._fns[4])  # pylint: disable=protected-access
    rng = np.random.default_rng(6)
    for trial in range(600):
      f = rng.normal(size=289).astype(np.float32)
      if trial % 3 == 0:
        f[rng.choice(289, rng.integers(2, 6), replace=False)] = np.nan
      elif trial % 3 == 1:
        f[rng.choice(289, rng.integers(2, 6), replace=False)] = f.max() + 1
      self.assertEqual(int(first_max(jnp.asarray(f))), int(np.argmax(f)), trial)
    rows = rng.normal(size=(6, 289)).astype(np.float32)  # all six faces at once
    rows[0, [3, 9, 200]] = np.nan
    rows[2, [5, 6]] = rows[2].max() + 1
    np.testing.assert_array_equal(np.asarray(first_max(jnp.asarray(rows))), np.argmax(rows, axis=1))

  def test_valid_positions_match_canvas(self):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    rng = np.random.default_rng(5)
    raw = np.zeros((60, 62, 64), np.float32)
    cv = self._canvas(raw)
    values = rng.normal(0, 1, raw.shape).astype(np.float32)
    values[rng.random(raw.shape) < 0.2] = np.nan
    cv.seed[...] = values
    self.fast._put_image(raw)  # pylint: disable=protected-access
    self.fast._seed = jnp.asarray(values)  # pylint: disable=protected-access
    view = fasttrace._CanvasView(self.fast)  # pylint: disable=protected-access
    for _ in range(300):
      pos = tuple(int(v) for v in rng.integers(0, raw.shape))
      self.assertEqual(view.is_valid_pos(pos), cv.is_valid_pos(pos), pos)
    cv._deregister_client()  # pylint: disable=protected-access

  def test_notebook_seed_is_bitwise_identical_on_cpu(self):
    raw = reference.normalize(data.load_raw_xyz())
    t = self.fast.trace(raw, reference.SEED_XYZ[::-1])
    self.assertEqual(reference.fingerprint(t), {k: v for k, v in reference.load_expected().items()
                                                if k in reference.fingerprint(t)})


if __name__ == '__main__':
  unittest.main()
