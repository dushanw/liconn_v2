"""The authors' LICONN FFN inference example as a CPU script, plus its fingerprint.

Follows google/ffn notebooks/jax_ffn_inference_liconn.ipynb at commit 12d680e
(see docs/method_source.md for the verbatim notebook code and the disclosed
deviations: no GPU assert, no DynamicImage visualization, cached inputs).
"""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
# The reference is defined on CPU (owner decision). Set before anything can
# initialize a JAX backend; run_trace re-checks the active backend.
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np  # pylint: disable=g-import-not-at-top
from scipy.ndimage import label
from scipy.special import logit

from liconn_v2 import data

# Settings copied from the notebook.
DEPTH = 20
FOV_SIZE = (33, 33, 33)
DELTAS = (8, 8, 8)
VOXEL_SIZE_ZYX = (24, 18, 18)
INIT_ACTIVATION = 0.95
PAD_VALUE = 0.5
MOVE_THRESHOLD = 0.6
SEGMENT_THRESHOLD = 0.6
SEED_XYZ = (123, 171, 225)

EXPECTED_FILE = Path(__file__).with_name('expected_reference.json')


@dataclasses.dataclass
class Trace:
  seed: np.ndarray  # float32 FFN canvas (z, y, x), NaN where never predicted.
  history: list  # FOV centers (z, y, x) in visit order.
  iterations: int
  compile_seconds: float
  trace_seconds: float
  counters: dict
  backend: str


def normalize(raw_xyz: np.ndarray) -> np.ndarray:
  raw = np.transpose(raw_xyz, [2, 1, 0])
  return (raw.astype(np.float32) - 128.0) / 33.  # normalize data for inference


class Tracer:
  """The notebook's model and executor, loaded once, tracing one object per call.

  Each call uses a fresh Canvas, so traces are independent, as in the notebook.
  """

  def __init__(self, ckpt_path: str):
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    tf.config.set_visible_devices([], 'GPU')
    from clu import checkpoint  # pylint: disable=g-import-not-at-top
    from connectomics.jax.models import convstack  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    # TF device settings do not constrain JAX; fail if a backend other than CPU was
    # initialized before this module set JAX_PLATFORMS.
    jax.config.update('jax_platforms', 'cpu')
    if jax.default_backend() != 'cpu':
      raise RuntimeError(f'JAX backend is {jax.default_backend()}, expected cpu')
    from ffn.inference import executor  # pylint: disable=g-import-not-at-top
    from ffn.inference import inference  # pylint: disable=g-import-not-at-top
    from ffn.inference import inference_pb2  # pylint: disable=g-import-not-at-top
    from ffn.inference import inference_utils  # pylint: disable=g-import-not-at-top
    from ffn.training import model as ffn_model  # pylint: disable=g-import-not-at-top
    self._inference = inference
    self._inference_utils = inference_utils
    self.backend = jax.default_backend()

    ckpt = checkpoint.Checkpoint('')
    state = ckpt.load_state(state=None, checkpoint=ckpt_path)
    self.params = state['params']

    model = convstack.ResConvStack(convstack.ConvstackConfig(depth=DEPTH, padding='same', use_layernorm=True))
    self.model_info = ffn_model.ModelInfo(deltas=DELTAS, pred_mask_size=FOV_SIZE, input_seed_size=FOV_SIZE, input_image_size=FOV_SIZE)

    @jax.jit
    def _apply_fn(data_):
      return model.apply({'params': state['params']}, data_)

    iface = executor.ExecutorInterface()
    self.counters = inference_utils.Counters()
    # The executor normally exits once `expected_clients` canvases have come and gone;
    # keep it serving one canvas per trace until close().
    self._exc = executor.JAXExecutor(iface, self.model_info, _apply_fn, self.counters, 1,
                                     expected_clients=2**31)

    # Compile once up front so the step counters measure steady-state inference.
    # The model is a pure function; this call does not affect the trace.
    t0 = time.perf_counter()
    np.asarray(self._exc._apply_fn(jnp.zeros((1, *FOV_SIZE, 2), jnp.float32)))  # pylint: disable=protected-access
    self.compile_seconds = time.perf_counter() - t0

    self.options = inference_pb2.InferenceOptions(
        init_activation=INIT_ACTIVATION, pad_value=PAD_VALUE,
        move_threshold=MOVE_THRESHOLD, segment_threshold=SEGMENT_THRESHOLD)
    self._exc.start_server()

  def trace(self, raw_zyx: np.ndarray, seed_zyx, instrument=None) -> Trace:
    """`instrument(canvas)`, if given, is called before tracing (read-only hooks)."""
    before = json.loads(self.counters.dumps())
    trace_counters = self._inference_utils.Counters()
    # keep_history only appends visited positions and counters only record timings;
    # neither alters the canvas.
    cv = self._inference.Canvas(self.model_info, self._exc.get_client(trace_counters), raw_zyx,
                                self.options, voxel_size_zyx=VOXEL_SIZE_ZYX,
                                counters=trace_counters, keep_history=True)
    if instrument is not None:
      instrument(cv)
    t0 = time.perf_counter()
    iterations = cv.segment_at(tuple(int(v) for v in seed_zyx))
    trace_seconds = time.perf_counter() - t0
    seed = np.array(cv.seed, dtype=np.float32)
    history = [tuple(int(v) for v in p) for p in cv.history]
    cv._deregister_client()  # pylint: disable=protected-access
    after = json.loads(self.counters.dumps())
    counters = {k: v - before.get(k, 0) for k, v in after.items()}
    counters.update(json.loads(trace_counters.dumps()))
    return Trace(seed, history, int(iterations), self.compile_seconds, trace_seconds,
                 counters, self.backend)

  def close(self):
    self._exc.stop_server()

  def __enter__(self):
    return self

  def __exit__(self, *exc_info):
    self.close()


def run_trace(raw_zyx: np.ndarray, ckpt_path: str) -> Trace:
  """Traces the notebook's single neurite with the notebook's settings."""
  with Tracer(ckpt_path) as tracer:
    return tracer.trace(raw_zyx, SEED_XYZ[::-1])


def gflop_per_eval(params, fov_size=FOV_SIZE) -> float:
  """Dense-convolution count: 2 x prod(kernel) x Cin x Cout x FOV voxels per conv."""
  import jax  # pylint: disable=g-import-not-at-top
  voxels = int(np.prod(fov_size))
  flops = sum(2 * int(np.prod(x.shape[:3])) * x.shape[3] * x.shape[4] * voxels
              for x in jax.tree_util.tree_leaves(params) if x.ndim == 5)
  return flops / 1e9


def notebook_mask(seed: np.ndarray) -> np.ndarray:
  """The notebook's mask: label(cv.seed > 0)[0] == 1."""
  return label(seed > 0)[0] == 1


def seed_component(seed: np.ndarray) -> np.ndarray:
  """Component containing the seed at FFN's segment_threshold (diagnostic)."""
  labels = label(seed >= np.float32(logit(SEGMENT_THRESHOLD)))[0]
  z, y, x = SEED_XYZ[::-1]
  return labels == labels[z, y, x] if labels[z, y, x] else np.zeros_like(labels, bool)


def save_mask_npz(path, mask: np.ndarray) -> None:
  """Stores a full-volume mask cropped to its bounding box."""
  zz, yy, xx = np.nonzero(mask)
  lo = np.array([zz.min(), yy.min(), xx.min()])
  hi = np.array([zz.max(), yy.max(), xx.max()]) + 1
  np.savez_compressed(path, mask=mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]],
                      offset_zyx=lo, shape_zyx=np.array(mask.shape))


def load_mask_npz(path) -> np.ndarray:
  """Inverse of save_mask_npz."""
  with np.load(path) as f:
    full = np.zeros(tuple(f['shape_zyx']), bool)
    lo = f['offset_zyx']
    crop = f['mask']
    full[lo[0]:lo[0] + crop.shape[0], lo[1]:lo[1] + crop.shape[1], lo[2]:lo[2] + crop.shape[2]] = crop
  return full


def _sha(arr: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def fingerprint(trace: Trace) -> dict:
  """Values a faster implementation must reproduce exactly."""
  seed = np.ascontiguousarray(trace.seed, dtype='<f4')
  canonical = np.where(np.isnan(seed), np.float32(np.nan), seed + np.float32(0.0)).astype('<f4')
  nb = notebook_mask(seed)
  comp = seed_component(seed)
  z, y, x = SEED_XYZ[::-1]
  return {
      'seed_sha256': _sha(seed),
      'seed_canonical_sha256': _sha(canonical),
      'predicted_voxels': int((~np.isnan(seed)).sum()),
      'predicted_mask_sha256': _sha(~np.isnan(seed)),
      'iterations': trace.iterations,
      'history_sha256': hashlib.sha256(json.dumps(trace.history).encode()).hexdigest(),
      'history_first': list(trace.history[0]) if trace.history else None,
      'notebook_mask_sha256': _sha(nb),
      'notebook_mask_voxels': int(nb.sum()),
      'notebook_mask_contains_seed': bool(nb[z, y, x]),
      'seed_component_sha256': _sha(comp),
      'seed_component_voxels': int(comp.sum()),
  }


def load_expected() -> dict:
  return json.loads(EXPECTED_FILE.read_text())


def run_reference() -> tuple[Trace, dict]:
  raw = normalize(data.load_raw_xyz())
  trace = run_trace(raw, data.checkpoint_path())
  return trace, fingerprint(trace)
