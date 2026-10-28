"""The authors' FFN network evaluated in lower precision (docs/precision_design.md).

Same forward pass and layer names as connectomics' ResConvStack (depth 20, 32
features, LayerNorm, 'same' padding), with a number format per layer type.
Parameters stay float32 and are cast at use; input and output stay float32.
In float32 it computes exactly the authors' network.
"""

import time

import numpy as np

from liconn_v2 import reference

FEATURES = 32
KERNEL = (3, 3, 3)
# variant name -> (convolution dtype, LayerNorm dtype)
VARIANTS = {
    'fp32': ('float32', 'float32'),
    'fp16': ('float16', 'float16'),
    'fp16_ln32': ('float16', 'float32'),
    'bf16': ('bfloat16', 'bfloat16'),
    'bf16_ln32': ('bfloat16', 'float32'),
}


def _module(variant, depth=reference.DEPTH):
  import flax.linen as nn  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  if variant not in VARIANTS:
    raise ValueError(f'unknown variant {variant!r}; known: {sorted(VARIANTS)}')
  conv_dt, ln_dt = (getattr(jnp, n) for n in VARIANTS[variant])

  class PrecisionResConvStack(nn.Module):

    @nn.compact
    def __call__(self, x):
      conv = lambda name, f=FEATURES, k=KERNEL: nn.Conv(f, k, padding='same', dtype=conv_dt, name=name)
      norm = lambda name: nn.LayerNorm(dtype=ln_dt, name=name)
      x = conv('pre_a')(x)
      x = norm('LayerNorm_0')(x)
      x = nn.relu(x)
      x = conv('pre_b')(x)
      for i in range(depth):
        mod_input = x
        x = norm(f'LayerNorm_{i + 1}')(x)
        x = nn.relu(x)
        x = conv(f'res{i}a')(x)
        x = nn.relu(x)
        x = conv(f'res{i}b')(x)
        x += mod_input
      x = norm(f'LayerNorm_{depth + 1}')(x)
      x = nn.relu(x)
      return conv('output', 1, (1, 1, 1))(x).astype(jnp.float32)

  return PrecisionResConvStack()


def apply_fn(params, variant):
  """A jitted function data -> float32 logits in the given number format."""
  import jax  # pylint: disable=g-import-not-at-top
  model = _module(variant)

  @jax.jit
  def f(data):
    return model.apply({'params': params}, data)

  return f


def install(tracer, variant):
  """Makes an existing reference.Tracer evaluate the network in `variant`."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  if variant == 'fp32':
    return
  tracer._exc._apply_fn = apply_fn(tracer.params, variant)  # pylint: disable=protected-access
  np.asarray(tracer._exc._apply_fn(jnp.zeros((1, *reference.FOV_SIZE, 2), jnp.float32)))  # pylint: disable=protected-access


def time_per_eval(f, x, repeats=50) -> float:
  """Median seconds per call at batch 1, after one warm-up call."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  xb = jnp.asarray(x[:1])
  np.asarray(f(xb))
  times = []
  for _ in range(repeats):
    t = time.perf_counter()
    np.asarray(f(xb))
    times.append(time.perf_counter() - t)
  return float(np.median(times))


def screen(params, inputs, variants, batch=1, timing_repeats=50) -> dict:
  """Flip rate, logit differences and time per evaluation against float32.

  batch=1 by default, as FFN evaluates the network during tracing; GPU convolution
  results can depend on the batch size.
  """
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  x = np.asarray(inputs, np.float32)
  t = np.float32(np.log(reference.SEGMENT_THRESHOLD / (1 - reference.SEGMENT_THRESHOLD)))

  def run(f):
    return np.concatenate([np.asarray(f(jnp.asarray(x[i:i + batch])))[..., 0]
                           for i in range(0, len(x), batch)])

  f32 = apply_fn(params, 'fp32')
  full = run(f32)
  full_fg = full >= t
  base_time = time_per_eval(f32, x, timing_repeats)
  results = []
  for v in variants:
    f = apply_fn(params, v)
    y = run(f)
    # Overflow can make 16-bit outputs non-finite. Such a voxel has no decision, so it
    # counts as changed; logit differences are taken over finite outputs only.
    finite = np.isfinite(y)
    fg = (y >= t) & finite
    changed = (fg != full_fg) | ~finite
    d = (y - full)[finite]
    sec = time_per_eval(f, x, timing_repeats)
    results.append({
        'variant': v,
        'non_finite_rate': float((~finite).mean()),
        'inputs_with_non_finite': int((~finite.reshape(len(x), -1).all(axis=1)).sum()),
        'flip_rate': float(changed.mean()),
        'lost_rate': float((full_fg & ~fg).mean()),
        'gained_rate': float((~full_fg & fg).mean()),
        'mean_abs_logit_diff': float(np.abs(d).mean()) if d.size else None,
        'max_abs_logit_diff': float(np.abs(d).max()) if d.size else None,
        'mean_logit_shift': float(d.mean()) if d.size else None,
        'seconds_per_eval': sec,
        'speedup_vs_fp32': base_time / sec if sec else None,
    })
  return {'fp32_seconds_per_eval': base_time, 'batch': batch, 'variants': results}


def select_for_benchmark(results) -> list:
  """Rule fixed before the screen: fp16, and the bfloat16 variant with the lower flip rate."""
  by = {r['variant']: r for r in results}
  bf = min((by[v] for v in ('bf16', 'bf16_ln32') if v in by), key=lambda r: (r['flip_rate'], r['variant']))
  return ['fp16', bf['variant']]


def run_variant(seeds, raw_zyx, gt_zyx, ckpt_path, variant, on_record=None):
  """Like benchmark.run, with the network in `variant`. Returns (records, masks)."""
  from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
  records, masks = [], {}
  with reference.Tracer(ckpt_path) as tracer:
    install(tracer, variant)
    gflop = reference.gflop_per_eval(tracer.params)  # FLOPs do not change with precision
    for s in seeds:
      trace = tracer.trace(raw_zyx, s['seed_zyx'])
      mask = benchmark.object_mask(trace.seed)
      rec = {'name': s['name'], 'class': s['class'], 'seed_zyx': list(s['seed_zyx'])}
      rec.update(benchmark.score(mask, gt_zyx, s['seed_zyx']))
      rec.update(benchmark.cost(trace.iterations, rec['correct_voxels'], gflop))
      rec.update({'trace_seconds': trace.trace_seconds, 'gflop_per_eval': gflop,
                  'number_format': variant, 'backend': trace.backend,
                  'device_kind': trace.device_kind,
                  'history_sha256': reference.history_sha256(trace.history),
                  'canvas_sha256': reference.canvas_sha256(trace.seed),
                  # NaN marks never-predicted voxels; overflow would show as +-inf.
                  'canvas_inf_voxels': int(np.isinf(trace.seed).sum())})
      records.append(rec)
      masks[s['name']] = mask
      if on_record is not None:
        on_record(rec)
  return records, masks
