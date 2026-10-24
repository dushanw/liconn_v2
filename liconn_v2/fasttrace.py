"""Fast single-object tracer: FFN's trace without the overhead around the network
(docs/fasttrace_design.md).

The image and the canvas (float32 logits, NaN where unpredicted) stay on the device. The
network is the same jitted function the reference path calls through FFN's executor;
here it is called directly. Input assembly, FFN's seed-consistency update, the canvas
write and the six face maxima (FFN's get_scored_move_offsets) are small jitted
functions; only the six maxima and their positions come back to the host. The queue is
FFN's own FaceMaxMovementPolicy and MovementRestrictor on the host, fed by queue_moves,
which does exactly what FaceMaxMovementPolicy.update does with those maxima, and a
stand-in for the canvas that answers is_valid_pos as Canvas does. The output must be
bitwise identical to reference.Tracer.trace.
"""

import functools
import time

import numpy as np

from liconn_v2 import reference

LOGIT_OPTIONS = ('init_activation', 'pad_value', 'move_threshold', 'segment_threshold')  # as in Canvas.__init__


def canvas_options(options):
  """FFN Canvas's copy of the inference options, with thresholds converted to logits."""
  from ffn.inference import inference  # pylint: disable=g-import-not-at-top
  from ffn.inference import inference_pb2  # pylint: disable=g-import-not-at-top
  opts = inference_pb2.InferenceOptions()
  opts.CopyFrom(options)
  for attr in LOGIT_OPTIONS:
    setattr(opts, attr, inference.logit(getattr(opts, attr)))
  return opts


class _CanvasView:
  """What FaceMaxMovementPolicy asks of its canvas: Canvas.is_valid_pos."""

  def __init__(self, fast):
    self._fast = fast

  def is_valid_pos(self, pos, ignore_move_threshold=False):
    f = self._fast
    if not ignore_move_threshold and f.value(pos) < f.opts.move_threshold:
      return False
    np_pos = np.array(pos)
    if np.any(np_pos - f.margin < 0) or np.any(np_pos + f.margin >= f.shape):
      return False
    return True  # a fresh Canvas has no segmentation yet (Canvas.segmentation is all 0)


@functools.lru_cache(maxsize=None)
def _device_fns(size, pad_value, move_threshold):
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax import lax  # pylint: disable=g-import-not-at-top
  size = tuple(size)
  half = np.array(size) // 2
  pad = np.float32(pad_value)
  t_move = np.float32(move_threshold)

  @jax.jit
  def assemble(seed, image, corner):
    """Executor input for one box: image and logit seed (NaN -> pad), as (1, z, y, x, 2)."""
    s = lax.dynamic_slice(seed, corner, size)
    s = jnp.where(jnp.isnan(s), pad, s)
    im = lax.dynamic_slice(image, corner, size)
    return jnp.concatenate([im[None, ..., None], s[None, ..., None]], axis=-1)

  def first_max(faces):
    """np.argmax along the last axis: the first NaN if there is one, otherwise the first
    maximum. On a GPU, jnp.argmax does not reliably pick the first of several NaNs, so
    NaNs are located explicitly (argmax of a boolean mask breaks ties by the lowest index)."""
    nan = jnp.isnan(faces)
    return jnp.where(jnp.any(nan, axis=-1), jnp.argmax(nan, axis=-1), jnp.argmax(faces, axis=-1))

  @functools.partial(jax.jit, donate_argnums=0)
  def update(seed, out, corner):
    """Canvas.update_at after prediction (disco_seed_threshold = 0); returns the new canvas
    and, in FFN's face order, the six face maxima of the prediction and their positions
    within each 17 x 17 face (get_scored_move_offsets with deltas 8: argmax in C order,
    first maximum wins, the first NaN counts as the maximum, as in NumPy)."""
    logits = out[0, ..., 0]
    old = lax.dynamic_slice(seed, corner, size)
    active = jnp.mean(logits >= t_move) > 0
    mask = active & (old < np.float32(0.0)) & (logits > old)  # th_max = logit(0.5) = 0
    logits = jnp.where(mask, old, logits)
    seed = lax.dynamic_update_slice(seed, logits, corner)
    sub = [slice(int(h) - 8, int(h) + 9) for h in half]
    faces = []
    for axis in range(3):
      for offset in (-8, 8):
        sel = list(sub)
        sel[axis] = int(half[axis]) + offset
        faces.append(logits[tuple(sel)].reshape(-1))  # a 17 x 17 face, C order
    faces = jnp.stack(faces)  # (6, 289), FFN's face order
    flat = first_max(faces)  # one reduction for all six faces
    scores = jnp.take_along_axis(faces, flat[:, None], axis=1)[:, 0]
    return seed, scores, jnp.stack([flat // 17, flat % 17], axis=1)

  @jax.jit
  def value(seed, pos):
    return seed[pos[0], pos[1], pos[2]]

  @jax.jit
  def initial(shape_holder, start, init):
    seed = jnp.full(shape_holder.shape, jnp.nan, jnp.float32)
    return seed.at[start[0], start[1], start[2]].set(init)

  return assemble, update, value, initial, first_max


def queue_moves(policy, scores, positions, position, deltas) -> None:
  """FaceMaxMovementPolicy.update, given the six face maxima computed on the device.

  Same steps as FFN: mark the position's lattice cell done; for each face in FFN's order,
  skip it if `score < threshold` (so a NaN score is kept, as in FFN); de-duplicate equal
  (score, relative position) pairs; sort descending; append score and absolute position.
  """
  policy.done_rounded_coords.add(policy.quantize_pos(position))
  done, moves, k = set(), [], 0
  for axis, d in enumerate(deltas):
    if d == 0:
      continue
    for offset in (-d, d):
      score, face_pos = scores[k], positions[k]
      k += 1
      if score < policy.score_threshold:
        continue
      rel = [face_pos[0] - 8, face_pos[1] - 8]
      rel.insert(axis, offset)
      ret = (score, tuple(rel))
      if ret not in done:
        done.add(ret)
        moves.append(ret)
  for score, rel in sorted(moves, reverse=True):
    policy.scored_coords.append((score, [rel[i] + position[i] for i in range(3)]))


class FastTracer:
  """Traces like reference.Tracer, with the tracer's (possibly replaced) network."""

  def __init__(self, tracer):
    self.tracer = tracer
    self.net = tracer._exc._apply_fn  # pylint: disable=protected-access
    self.opts = canvas_options(tracer.options)
    mi = tracer.model_info
    self.size = tuple(np.array(mi.input_seed_size[::-1]))
    assert tuple(np.array(mi.input_image_size[::-1])) == self.size == tuple(np.array(mi.pred_mask_size[::-1]))
    self.margin = np.array(self.size) // 2
    self.deltas = mi.deltas[::-1]
    self._fns = _device_fns(self.size, float(self.opts.pad_value), float(self.opts.move_threshold))
    self._image = None
    self.shape = None
    self._seed = None

  def value(self, pos) -> np.float32:
    return np.float32(self._fns[2](self._seed, np.asarray(pos, np.int32)))

  def _put_image(self, raw_zyx):
    """Copies the image to the device for every trace, as reference.Tracer reads it on
    every call (no caching: a cache keyed by the array could reuse a stale image)."""
    import jax  # pylint: disable=g-import-not-at-top
    self._image = jax.device_put(np.asarray(raw_zyx, np.float32))
    self.shape = np.array(raw_zyx.shape)

  def trace(self, raw_zyx, seed_zyx) -> reference.Trace:
    import jax  # pylint: disable=g-import-not-at-top
    from ffn.inference import movement  # pylint: disable=g-import-not-at-top
    assemble, update, _, initial, _ = self._fns
    self._put_image(raw_zyx)
    start = tuple(int(v) for v in seed_zyx)
    self._seed = initial(self._image, np.asarray(start, np.int32), np.float32(self.opts.init_activation))
    self._image.block_until_ready()  # setup is finished before the timer starts
    self._seed.block_until_ready()
    self._view = _CanvasView(self)  # FFN's policy keeps only a weak reference to its canvas
    policy = movement.FaceMaxMovementPolicy(self._view, deltas=self.deltas,
                                            score_threshold=self.opts.move_threshold)
    restrictor = movement.MovementRestrictor()
    policy.reset_state(start)
    if not policy:
      policy.append((policy.score_threshold * 2, start))
    history, iterations = [], 0
    t0 = time.perf_counter()
    for pos in policy:  # Canvas.segment_at
      if self.value(start) < self.opts.move_threshold:
        break
      if not restrictor.is_valid_pos(pos):
        continue
      corner = np.asarray(np.array(pos) - self.margin, np.int32)
      out = self.net(assemble(self._seed, self._image, corner))
      self._seed, scores, positions = update(self._seed, out, corner)
      iterations += 1
      scores, positions = jax.device_get((scores, positions))  # one transfer
      queue_moves(policy, scores, positions.astype(np.int64), pos, self.deltas)
      history.append(tuple(int(v) for v in pos))
    seed = np.asarray(self._seed)
    trace_seconds = time.perf_counter() - t0
    return reference.Trace(seed, history, iterations, self.tracer.compile_seconds, trace_seconds, {},
                           self.tracer.backend, self.tracer.device_kind)
