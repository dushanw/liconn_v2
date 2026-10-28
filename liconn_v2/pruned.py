"""The authors' FFN network with chosen residual blocks dropped (docs/block_drop_design.md).

PrunedResConvStack repeats connectomics' ResConvStack forward pass (depth 20,
32 features, LayerNorm, 'same' padding) with every layer named explicitly as in
the checkpoint, so it loads the authors' weights unchanged. With no blocks dropped
it computes exactly the same function; a dropped block becomes the identity.
"""

import numpy as np

from liconn_v2 import reference

DEPTH = reference.DEPTH
FEATURES = 32
KERNEL = (3, 3, 3)
T = np.float32(np.log(reference.SEGMENT_THRESHOLD / (1 - reference.SEGMENT_THRESHOLD)))


def _module(depth=DEPTH, skip=()):
  import flax.linen as nn  # pylint: disable=g-import-not-at-top

  class PrunedResConvStack(nn.Module):
    """ResConvStack (num_convs=2, use_layernorm=True) with blocks in `skip` dropped."""

    @nn.compact
    def __call__(self, x):
      x = nn.Conv(FEATURES, KERNEL, padding='same', name='pre_a')(x)
      x = nn.LayerNorm(name='LayerNorm_0')(x)
      x = nn.relu(x)
      x = nn.Conv(FEATURES, KERNEL, padding='same', name='pre_b')(x)
      for i in range(depth):
        if i in skip:
          continue  # the block becomes the identity
        mod_input = x
        x = nn.LayerNorm(name=f'LayerNorm_{i + 1}')(x)
        x = nn.relu(x)
        x = nn.Conv(FEATURES, KERNEL, padding='same', name=f'res{i}a')(x)
        x = nn.relu(x)
        x = nn.Conv(FEATURES, KERNEL, padding='same', name=f'res{i}b')(x)
        x += mod_input
      x = nn.LayerNorm(name=f'LayerNorm_{depth + 1}')(x)
      x = nn.relu(x)
      return nn.Conv(1, (1, 1, 1), name='output')(x)

  return PrunedResConvStack()


def used_params(params, skip=(), depth=DEPTH):
  """The subset of checkpoint parameters the pruned network uses."""
  drop = set()
  for i in skip:
    drop |= {f'res{i}a', f'res{i}b', f'LayerNorm_{i + 1}'}
  return {k: v for k, v in params.items() if k not in drop}


def apply_fn(params, skip=(), depth=DEPTH):
  """A jitted function data -> logits, like the notebook's _apply_fn."""
  import jax  # pylint: disable=g-import-not-at-top
  model = _module(depth, tuple(sorted(skip)))
  p = used_params(params, skip, depth)

  @jax.jit
  def f(data):
    return model.apply({'params': p}, data)

  return f


def gflop_per_eval(params, skip=()) -> float:
  return reference.gflop_per_eval(used_params(params, skip))


def install(tracer, skip):
  """Makes an existing reference.Tracer evaluate the pruned network.

  Replaces the executor's compiled network function after the Tracer is built, so
  reference.py stays unchanged; compiles once before tracing.
  """
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  if not skip:
    return
  tracer._exc._apply_fn = apply_fn(tracer.params, skip)  # pylint: disable=protected-access
  np.asarray(tracer._exc._apply_fn(jnp.zeros((1, *reference.FOV_SIZE, 2), jnp.float32)))  # pylint: disable=protected-access


class InputRecorder:
  """Records the network input of every `every`-th step (read-only).

  Wraps Canvas.predict(pos, logit_seed); the input FFN builds from these is
  [image box, seed box] along the channel axis, as in ffn's JAXExecutor.
  """

  def __init__(self, every=8):
    self.every = every
    self.inputs = []
    self.steps = []
    self._n = 0

  def attach(self, cv):
    orig = cv.predict

    def predict(pos, logit_seed):
      if self._n % self.every == 0:
        img = np.array(cv._get_image(pos), np.float32)  # pylint: disable=protected-access
        self.inputs.append(np.stack([img, np.array(logit_seed, np.float32)], axis=-1))
        self.steps.append(self._n)
      self._n += 1
      return orig(pos, logit_seed)

    cv.predict = predict


class Screen:
  """Compares pruned variants with the full network on recorded inputs."""

  def __init__(self, params, inputs, batch=8):
    self.params = params
    self.x = np.asarray(inputs, np.float32)
    self.batch = batch
    self.full = self._run(())
    self.full_fg = self.full >= T

  def _run(self, skip):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    f = apply_fn(self.params, skip)
    return np.concatenate([np.asarray(f(jnp.asarray(self.x[i:i + self.batch])))[..., 0]
                           for i in range(0, len(self.x), self.batch)])

  def measure(self, label, skip) -> dict:
    y = self._run(skip)
    fg = y >= T
    d = y - self.full
    return {
        'label': label,
        'skip': sorted(int(b) for b in skip),
        'flip_rate': float((fg != self.full_fg).mean()),
        'lost_rate': float((self.full_fg & ~fg).mean()),     # object -> background
        'gained_rate': float((~self.full_fg & fg).mean()),   # background -> object
        'mean_abs_logit_diff': float(np.abs(d).mean()),
        'mean_logit_shift': float(d.mean()),
        'gflop_per_eval': gflop_per_eval(self.params, skip),
    }


def screen(params, inputs, variants, batch=8) -> list:
  """Measures each (label, skip) variant against the full network."""
  s = Screen(params, inputs, batch)
  return [s.measure(label, skip) for label, skip in variants]


def sequential_greedy(measure, depth=DEPTH, max_blocks=DEPTH) -> list:
  """Drops one more block at a time: the one with the lowest combined flip rate.

  `measure(label, skip)` returns a screen result. Ties go to the lower block index.
  Returns the chosen set after each step, with its measured result.
  """
  dropped, path = [], []
  for k in range(1, max_blocks + 1):
    candidates = [b for b in range(depth) if b not in dropped]
    if not candidates:
      break
    trials = [measure(f'greedy_{k}', tuple(sorted(dropped + [b]))) for b in candidates]
    best = min(zip(trials, candidates), key=lambda tc: (tc[0]['flip_rate'], tc[1]))
    dropped.append(best[1])
    path.append(best[0])
  return path


def select_variants(single_results, greedy_results, limits=(0.001, 0.005, 0.02)) -> list:
  # greedy_results: the sequential-greedy path, one result per number of dropped blocks.
  """The pre-registered rule: best single drop, then the largest greedy set per limit."""
  chosen = []
  best = min(single_results, key=lambda r: (r['flip_rate'], r['skip'][0]))
  chosen.append(('best_single', tuple(best['skip'])))
  by_k = sorted(greedy_results, key=lambda r: len(r['skip']))
  for lim in limits:
    ok = [r for r in by_k if r['flip_rate'] <= lim]
    if ok:
      chosen.append((f'greedy_flip_le_{lim}', tuple(ok[-1]['skip'])))
  seen, unique = set(), []
  for label, skip in chosen:
    if tuple(sorted(skip)) not in seen:
      seen.add(tuple(sorted(skip)))
      unique.append((label, tuple(sorted(skip))))
  return unique


def run_variant(seeds, raw_zyx, gt_zyx, ckpt_path, skip, on_record=None):
  """Like benchmark.run, with the pruned network. Returns (records, masks)."""
  from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
  records, masks = [], {}
  with reference.Tracer(ckpt_path) as tracer:
    install(tracer, skip)
    gflop = gflop_per_eval(tracer.params, skip)
    for s in seeds:
      trace = tracer.trace(raw_zyx, s['seed_zyx'])
      mask = benchmark.object_mask(trace.seed)
      rec = {'name': s['name'], 'class': s['class'], 'seed_zyx': list(s['seed_zyx'])}
      rec.update(benchmark.score(mask, gt_zyx, s['seed_zyx']))
      rec.update(benchmark.cost(trace.iterations, rec['correct_voxels'], gflop))
      rec.update({'trace_seconds': trace.trace_seconds, 'gflop_per_eval': gflop,
                  'number_format': 'float32', 'backend': trace.backend,
                  'device_kind': trace.device_kind,
                  'history_sha256': reference.history_sha256(trace.history),
                  'canvas_sha256': reference.canvas_sha256(trace.seed),
                  'skip_blocks': sorted(int(b) for b in skip)})
      records.append(rec)
      masks[s['name']] = mask
      if on_record is not None:
        on_record(rec)
  return records, masks
