"""Why float16 stops traces early (docs/earlystop_design.md).

Records FFN's move decisions step by step (closed loop), re-evaluates the network
inputs of float32 traces in float16 and candidate variants (open loop), and analyses
where float16 traces begin to differ and whether float16 is biased against moving.
"""

import json
from pathlib import Path

import numpy as np

from liconn_v2 import reference

# The 10 new early stops of docs/manyseed_results.md.
EARLY_STOP_SEEDS = ('m1_axon_175567', 'm1_axon_173408', 'm2_dendrite_104095', 'm2_axon_92043',
                    'm3_dendrite_142095', 'm5_dendrite_226719', 'm5_axon_232100', 'm6_axon_216332',
                    'm7_axon_320743', 'm10_axon_145819')
DELTAS = (8, 8, 8)  # FaceMaxMovementPolicy deltas (z, y, x) for this model
NEAR = 0.05  # "near threshold": float32 face maximum within this many logits of logit(0.6)
THRESHOLD_SUPPORT = 7  # early stops that must begin at a threshold step to support A
BOOTSTRAP = 10_000
RNG_SEED = 0
MIN_SPEEDUP = 1.5
CANDIDATES = ('fp16', 'fp16_head32', 'fp16_res32')
EVENTS = ('new_merge', 'new_early_stop', 'avoided_merge', 'extension')


def move_threshold() -> np.float32:
  p = reference.MOVE_THRESHOLD
  return np.float32(np.log(p / (1 - p)))


def segment_threshold() -> np.float32:
  p = reference.SEGMENT_THRESHOLD
  return np.float32(np.log(p / (1 - p)))


def quantize(pos, start, deltas=DELTAS) -> tuple:
  """FFN's FaceMaxMovementPolicy.quantize_pos: the 8-voxel lattice cell of a position."""
  d = np.array(deltas)
  return tuple(int(v) for v in (np.array(pos) - np.array(start) + d // 2) // np.maximum(d, 1))


# ---------------------------------------------------------------- seeds

def select_controls(verdict: dict, ref_records: dict) -> list:
  """For each early stop: the event-free seed of the same class, same crop if possible,
  with the closest float32 step count (ties by name); no seed twice."""
  flagged = set(verdict['flipped_seeds'])
  used, out = set(), []
  for name in EARLY_STOP_SEEDS:
    me = ref_records[name]
    pool = [r for n, r in ref_records.items() if n not in flagged and n not in used and r['class'] == me['class']]
    same = [r for r in pool if r['crop'] == me['crop']]
    best = min(same or pool, key=lambda r: (abs(r['steps'] - me['steps']), r['name']))
    used.add(best['name'])
    out.append(best['name'])
  return out


# ---------------------------------------------------------------- faces

def faces(prob_map: np.ndarray, threshold, deltas=DELTAS) -> list:
  """The six face maxima as FFN's get_scored_move_offsets sees them.

  Returns, in FFN's face order, (axis, offset, score, relative position, moves), where
  relative position is FFN's offset of the queued move from the box center.
  """
  center = np.array(prob_map.shape) // 2
  subvol = [slice(c - d, c + d + 1) for c, d in zip(center, deltas)]
  out = []
  for axis, d in enumerate(deltas):
    for offset in (-d, d):
      sel = subvol[:]
      sel[axis] = offset + center[axis]
      face = prob_map[tuple(sel)]
      pos = np.unravel_index(face.argmax(), face.shape)
      score = face[pos]
      rel = [int(pos[0] - face.shape[0] // 2), int(pos[1] - face.shape[1] // 2)]
      rel.insert(axis, offset)
      out.append((axis, offset, float(score), tuple(rel), bool(score >= threshold)))
  return out


def queued(face_list) -> list:
  """The queued moves in queue order (FFN sorts a step's moves by score, highest first)."""
  moves = sorted({(f[2], tuple(f[3])) for f in face_list if f[4]}, reverse=True)
  return [rel for _, rel in moves]


def seed_consistency(logits: np.ndarray, old_seed: np.ndarray, threshold) -> np.ndarray:
  """FFN Canvas.update_at's step after prediction (disco_seed_threshold = 0)."""
  out = np.array(logits, dtype=np.float32)
  if np.mean(out >= threshold) > 0:
    with np.errstate(invalid='ignore'):
      mask = (old_seed < np.float32(0.0)) & (out > old_seed)  # logit(0.5) = 0
    out[mask] = old_seed[mask]
  return out


def voxel_flips(y: np.ndarray, ref: np.ndarray) -> int:
  """Voxels whose segment decision differs from float32's; a non-finite output has no
  decision and counts as changed (as in precision.screen)."""
  ts = segment_threshold()
  return int((((y >= ts) != (ref >= ts)) | ~np.isfinite(y)).sum())


# ---------------------------------------------------------------- recorder

class Recorder:
  """Read-only hooks on a Canvas: the faces seen by the movement policy at every step.

  With `open_loop` (a dict name -> jitted network function), the network input of
  every step is also evaluated by each function; its output goes through FFN's
  seed-consistency step against the same old seed and its faces are recorded.
  """

  def __init__(self, tracer, open_loop=None):
    self.tracer = tracer
    self.open_loop = open_loop or {}
    self.steps = []
    self._last_input = None
    self._pending = None
    if self.open_loop:
      orig = tracer._exc._apply_fn  # pylint: disable=protected-access

      def apply_and_keep(data):
        self._last_input = data
        return orig(data)

      tracer._exc._apply_fn = apply_and_keep  # pylint: disable=protected-access
      self._orig_apply = orig

  def close(self):
    if self.open_loop:
      self.tracer._exc._apply_fn = self._orig_apply  # pylint: disable=protected-access

  def __call__(self, cv):
    t = move_threshold()
    policy = cv.movement_policy
    orig_update = policy.update
    orig_predict = cv.predict

    def predict(pos, logit_seed):
      logits = orig_predict(pos, logit_seed)
      if self.open_loop:
        off = np.array(reference.FOV_SIZE) // 2
        sel = tuple(slice(p - o, p - o + n) for p, o, n in zip(pos, off, reference.FOV_SIZE))
        old = np.array(cv.seed[sel])
        variants = {}
        for name, f in self.open_loop.items():
          y = np.asarray(f(self._last_input))[0, ..., 0].astype(np.float32)
          variants[name] = (faces(seed_consistency(y, old, t), t), voxel_flips(y, logits))
        self._pending = {'own': faces(seed_consistency(logits, old, t), t), 'variants': variants}
      return logits

    def update(prob_map, position):
      step = {'pos': [int(v) for v in position], 'faces': faces(prob_map, t)}
      if self._pending is not None:
        step['emulation_matches'] = step['faces'] == self._pending['own']
        step['open_loop'] = {k: {'faces': v[0], 'voxel_flips': v[1]} for k, v in self._pending['variants'].items()}
        self._pending = None
      self.steps.append(step)
      return orig_update(prob_map, position)

    cv.predict = predict
    policy.update = update


# ---------------------------------------------------------------- analysis A

def first_difference(ref_steps: list, cand_steps: list):
  """First step, in float32 visit order, at a center visited in both runs whose queued
  moves differ; classified as threshold, argmax or order. None if no such step."""
  t = float(move_threshold())
  cand_at = {}
  for s in cand_steps:
    cand_at.setdefault(tuple(s['pos']), s)
  for i, s in enumerate(ref_steps):
    c = cand_at.get(tuple(s['pos']))
    if c is None:
      continue
    a, b = s['faces'], c['faces']
    qa, qb = queued(a), queued(b)
    if qa == qb:
      continue
    moved_a = {(f[0], f[1]) for f in a if f[4]}
    moved_b = {(f[0], f[1]) for f in b if f[4]}
    if moved_a != moved_b:
      kind = 'threshold'
      face = sorted(moved_a ^ moved_b)[0]
    elif set(qa) != set(qb):
      kind = 'argmax'
      face = sorted((f[0], f[1]) for f, g in zip(a, b) if f[4] and f[3] != g[3])[0]
    else:
      kind = 'order'
      face = None
    out = {'step': i, 'pos': s['pos'], 'kind': kind}
    if face is not None:
      fa = next(f for f in a if (f[0], f[1]) == face)
      fb = next(f for f in b if (f[0], f[1]) == face)
      out.update(face=list(face), ref_score=fa[2], cand_score=fb[2],
                 margin=abs(fa[2] - t), ref_moves=fa[4], cand_moves=fb[4])
    return out
  return None


def frontier(ref_steps: list, cand_steps: list):
  """Where float16's trace stops (primary analysis A).

  The first float32-visited lattice cell, in float32 visit order, that float16 never
  visits although it visited the cell of the float32 step that queued it. At float16's
  step in that parent cell, the face that queued the lost move is classified as
  threshold (no move), argmax (moves into another cell) or queued_not_evaluated (moves
  into the same cell, but float16 never evaluates a box there; the recorder does not
  see why). None if float16 visits every float32 cell.
  """
  if not ref_steps or not cand_steps:
    return None
  start = ref_steps[0]['pos']
  t = float(move_threshold())
  q = lambda p: quantize(p, start)
  cand_cell = {}
  for s in cand_steps:
    cand_cell.setdefault(q(s['pos']), s)
  queued_by = {}
  for i, s in enumerate(ref_steps):
    for f in s['faces']:
      if f[4]:
        coord = tuple(int(p + r) for p, r in zip(s['pos'], f[3]))
        queued_by.setdefault(coord, (i, (f[0], f[1])))
  for j, s in enumerate(ref_steps[1:], 1):
    cell = q(s['pos'])
    if cell in cand_cell or tuple(s['pos']) not in queued_by:
      continue
    i, face = queued_by[tuple(s['pos'])]
    parent = ref_steps[i]
    pc = cand_cell.get(q(parent['pos']))
    if pc is None:
      continue  # an ancestor was already lost; not the frontier
    rf = next(f for f in parent['faces'] if (f[0], f[1]) == face)
    cf = next(f for f in pc['faces'] if (f[0], f[1]) == face)
    if not cf[4]:
      kind = 'threshold'
    elif q([p + r for p, r in zip(pc['pos'], cf[3])]) != cell:
      kind = 'argmax'
    else:
      kind = 'queued_not_evaluated'
    return {'step': j, 'parent_step': i, 'cell': list(cell), 'kind': kind, 'face': list(face),
            'same_center': list(pc['pos']) == list(parent['pos']),
            'ref_score': rf[2], 'cand_score': cf[2], 'margin': rf[2] - t}
  return None


# ---------------------------------------------------------------- analysis B

def face_errors(steps: list, variant: str) -> list:
  """(float32 face maximum, variant minus float32) for every face of every open-loop step.

  Faces whose variant score is not finite (float16 overflow gives NaN) are left out
  here and counted by non_finite_faces; FFN never moves through such a face.
  """
  out = []
  for s in steps:
    for f, g in zip(s['faces'], s['open_loop'][variant]['faces']):
      if np.isfinite(g[2]) and np.isfinite(f[2]):
        out.append((f[2], g[2] - f[2]))
  return out


def non_finite_faces(steps: list, variant: str) -> dict:
  bad = [(f, g) for s in steps for f, g in zip(s['faces'], s['open_loop'][variant]['faces'])
         if not np.isfinite(g[2])]
  return {'faces': sum(len(s['faces']) for s in steps), 'non_finite': len(bad),
          'non_finite_where_float32_moves': sum(1 for f, _ in bad if f[4])}


def near_threshold_errors(errors) -> np.ndarray:
  t = float(move_threshold())
  return np.array([e for score, e in errors if abs(score - t) <= NEAR], dtype=np.float64)


def bootstrap_mean(per_seed: dict, n=BOOTSTRAP, seed=RNG_SEED) -> dict:
  """Pooled mean over seeds, with a 95% interval from resampling whole seeds."""
  names = sorted(per_seed)
  arrays = [np.asarray(per_seed[k], np.float64) for k in names]
  pooled = np.concatenate(arrays) if arrays else np.array([])
  rng = np.random.default_rng(seed)
  sums = np.array([a.sum() for a in arrays])
  counts = np.array([a.size for a in arrays])
  idx = rng.integers(0, len(names), size=(n, len(names)))
  with np.errstate(invalid='ignore', divide='ignore'):
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
  means = means[np.isfinite(means)]
  return {'mean': float(pooled.mean()) if pooled.size else None, 'n': int(pooled.size),
          'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))] if means.size else None}


def losses_gains(steps: list, variant: str) -> dict:
  lost = sum(1 for s in steps for f, g in zip(s['faces'], s['open_loop'][variant]['faces']) if f[4] and not g[4])
  gained = sum(1 for s in steps for f, g in zip(s['faces'], s['open_loop'][variant]['faces']) if g[4] and not f[4])
  return {'lost': lost, 'gained': gained}


# ---------------------------------------------------------------- exploratory

PLATEAU = 2.9  # float32 face maxima above this are on the network's saturated plateau


def plateau_stats(steps_by_seed: dict, variants) -> dict:
  """Exploratory (not pre-registered, added after the runs): how many moving float32
  faces sit on the saturated plateau, and how often each variant moves the same face to
  a different voxel (argmax shift), on and off the plateau."""
  scores, shifts = [], {v: {'on_plateau': 0, 'below': 0} for v in variants}
  for steps in steps_by_seed.values():
    for s in steps:
      for k, f in enumerate(s['faces']):
        if not f[4]:
          continue
        scores.append(f[2])
        for v in variants:
          g = s['open_loop'][v]['faces'][k]
          if g[4] and list(g[3]) != list(f[3]):
            shifts[v]['on_plateau' if f[2] > PLATEAU else 'below'] += 1
  sc = np.array(scores)
  return {'moving_faces': int(sc.size), 'on_plateau_fraction': float((sc > PLATEAU).mean()) if sc.size else None,
          'median_score': float(np.median(sc)) if sc.size else None, 'argmax_shifts': shifts}


# ---------------------------------------------------------------- analysis C

def screen(steps_by_seed: dict, variant: str) -> dict:
  steps = [s for v in steps_by_seed.values() for s in v]
  differ = sum(1 for s in steps if queued(s['faces']) != queued(s['open_loop'][variant]['faces']))
  voxels = len(steps) * int(np.prod(reference.FOV_SIZE))
  return {'steps': len(steps), 'move_decision_flip_rate': differ / len(steps) if steps else None,
          **losses_gains(steps, variant),
          'voxel_flip_rate': sum(s['open_loop'][variant]['voxel_flips'] for s in steps) / voxels if steps else None}


def select_candidate(screens: dict, speedups: dict) -> str | None:
  """Pre-registered: among fixes (not plain float16) at least 1.5x faster per evaluation
  than float32 and with a lower move-decision flip rate than plain float16, the one with
  the lowest flip rate; None if none qualifies."""
  base = screens['fp16']['move_decision_flip_rate']
  ok = [v for v in screens if v != 'fp16' and speedups.get(v, 0) >= MIN_SPEEDUP
        and screens[v]['move_decision_flip_rate'] < base]
  return min(ok, key=lambda v: (screens[v]['move_decision_flip_rate'], v)) if ok else None


# ---------------------------------------------------------------- candidate networks

# name -> (convolution dtype, LayerNorm dtype, residual stream dtype, head dtype)
LAYOUTS = {
    'fp32_check': ('float32', 'float32', 'float32', 'float32'),
    'fp16_head32': ('float16', 'float16', 'float16', 'float32'),
    'fp16_res32': ('float16', 'float16', 'float32', 'float16'),
}


def _module(layout, depth=reference.DEPTH):
  import flax.linen as nn  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from liconn_v2 import precision  # pylint: disable=g-import-not-at-top
  conv_dt, ln_dt, stream_dt, head_dt = (getattr(jnp, n) for n in LAYOUTS[layout])

  class LayoutResConvStack(nn.Module):

    @nn.compact
    def __call__(self, x):
      conv = lambda name, dt=conv_dt, f=precision.FEATURES, k=precision.KERNEL: nn.Conv(
          f, k, padding='same', dtype=dt, name=name)
      x = conv('pre_a')(x)
      x = nn.LayerNorm(dtype=ln_dt, name='LayerNorm_0')(x)
      x = nn.relu(x)
      x = conv('pre_b')(x).astype(stream_dt)
      for i in range(depth):
        mod_input = x
        y = nn.LayerNorm(dtype=ln_dt, name=f'LayerNorm_{i + 1}')(x)
        y = nn.relu(y)
        y = conv(f'res{i}a')(y)
        y = nn.relu(y)
        y = conv(f'res{i}b')(y)
        x = mod_input + y.astype(stream_dt)
      x = nn.LayerNorm(dtype=head_dt, name=f'LayerNorm_{depth + 1}')(x)  # stream dtype in, head dtype out
      x = nn.relu(x)
      return conv('output', head_dt, 1, (1, 1, 1))(x).astype(jnp.float32)

  return LayoutResConvStack()


def shifted(f, c: float):
  """fp16_shift: a network function with a constant logit offset added (jitted)."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  return jax.jit(lambda data: f(data) + jnp.float32(c))


def apply_fn(params, name):
  """Jitted data -> float32 logits for a candidate layout or a precision.VARIANTS name."""
  import jax  # pylint: disable=g-import-not-at-top
  from liconn_v2 import precision  # pylint: disable=g-import-not-at-top
  if name not in LAYOUTS:
    return precision.apply_fn(params, name)
  model = _module(name)

  @jax.jit
  def f(data):
    return model.apply({'params': params}, data)

  return f


def install(tracer, name):
  """Makes a reference.Tracer evaluate the network as a candidate layout."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  if name not in LAYOUTS:
    raise ValueError(f'unknown layout {name!r}; known: {sorted(LAYOUTS)}')
  tracer._exc._apply_fn = apply_fn(tracer.params, name)  # pylint: disable=protected-access
  np.asarray(tracer._exc._apply_fn(jnp.zeros((1, *reference.FOV_SIZE, 2), jnp.float32)))  # pylint: disable=protected-access


def run_layout(seeds, raw_zyx, gt_zyx, ckpt_path, name, on_record=None):
  """precision.run_variant for a candidate layout: the same records, the network from install()."""
  from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
  records, masks = [], {}
  with reference.Tracer(ckpt_path) as tracer:
    install(tracer, name)
    gflop = reference.gflop_per_eval(tracer.params)
    for s in seeds:
      trace = tracer.trace(raw_zyx, s['seed_zyx'])
      mask = benchmark.object_mask(trace.seed)
      rec = {'name': s['name'], 'class': s['class'], 'seed_zyx': list(s['seed_zyx'])}
      rec.update(benchmark.score(mask, gt_zyx, s['seed_zyx']))
      rec.update(benchmark.cost(trace.iterations, rec['correct_voxels'], gflop))
      rec.update({'trace_seconds': trace.trace_seconds, 'gflop_per_eval': gflop,
                  'number_format': name, 'backend': trace.backend, 'device_kind': trace.device_kind,
                  'history_sha256': reference.history_sha256(trace.history),
                  'canvas_sha256': reference.canvas_sha256(trace.seed),
                  'canvas_inf_voxels': int(np.isinf(trace.seed).sum())})
      records.append(rec)
      masks[s['name']] = mask
      if on_record is not None:
        on_record(rec)
  return records, masks


# ---------------------------------------------------------------- io

def save_steps(path: Path, steps: list) -> None:
  path.write_text(json.dumps(steps) + '\n')


def load_steps(path: Path) -> list:
  return json.loads(Path(path).read_text())
