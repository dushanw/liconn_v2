"""Single-pass ViT segmentation pilot (docs/vitseg_design.md).

A UNETR-style network predicts, for a whole 128^3 block in one pass, six affinities and
the probability that each voxel belongs to a neuron (label 0 is background). Blocks are
tiled over a 500^3 crop; background stays 0, a watershed of the neuron voxels on the
boundary map gives fragments, and mean-affinity agglomeration (as in waterz) joins them. Each fresh seed is scored with
benchmark.score on the segment containing it, and compared with the recorded
fp16_head32 FFN trace of the same seed (manyseed.events).
"""

import heapq
import time

import numpy as np

from liconn_v2 import manyseed
from liconn_v2 import reference

# Model (fixed in the design).
BLOCK = 128
PATCH = 8
WIDTH = 384
DEPTH = 8
HEADS = 6
ENCODER = (16, 32, 64)  # conv channels at full, 1/2 and 1/4 resolution; tokens at 1/8
OFFSETS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 0, 0), (0, 4, 0), (0, 0, 4))  # (z, y, x)
NEAREST = 3  # the first three offsets are the nearest neighbours
NEURON_CHANNEL = len(OFFSETS)  # the 7th output: P(voxel belongs to a neuron, label > 0)
CHANNELS = len(OFFSETS) + 1

# Training.
BATCH = 2
PEAK_LR = 3e-4
WEIGHT_DECAY = 0.05
WARMUP = 1_000
LOSS_SCALE = 1024.0
MAX_SKIPPED_FRACTION = 0.01
CLIP_NORM = 1.0  # global gradient norm (revision 2, after run 1 diverged)
MAX_CONSECUTIVE_SKIPS = 200  # the trainer stops after this many non-finite steps in a row
VALIDATION_BLOCKS = 8  # fixed crop-12 blocks; the checkpoint with the lowest loss on them is used
VALIDATION_STREAM = 1_000_000_007  # random stream of the validation blocks (training uses step numbers)
TRAIN_CROPS = tuple(range(12))
TUNE_CROP = 12
RNG_SEED = 0
TRAIN_STEPS = 170_000  # ~35 h at 0.74 s/step (steps 101-400 of the timing job, Slurm 7433735)

# Segmentation.
STRIDE = 96
NEURON = 0.5  # neuron probability >= 0.5: a neuron voxel; below it, background (label 0)
FOREGROUND = 0.5  # b < 0.5: a confident neuron voxel (decision point of the balanced affinities)
PEAK_SIZE = 5  # distance-transform peaks are maxima within a 5^3 neighbourhood
VOXEL_SIZE_ZYX = reference.VOXEL_SIZE_ZYX
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))

# Verdict (owner-confirmed 2026-10-24, before training): (max mean drop, max net flips).
COMPETITIVE = (0.02, 6)
PROMISING = (0.05, 15)


def normalize_zyx(raw_xyz: np.ndarray) -> np.ndarray:
  """FFN's input normalisation, (raw - 128) / 33, in (z, y, x) order."""
  return (np.transpose(raw_xyz, [2, 1, 0]).astype(np.float32) - 128.0) / 33.0


# ---------------------------------------------------------------- targets and loss

def affinity_targets(labels: np.ndarray, offsets=OFFSETS):
  """(target, mask), each (z, y, x, len(offsets)) float32.

  For voxel v and offset o, the pair is (v, v - o): target 1 if both have the same
  nonzero label, 0 if the labels differ and at least one is nonzero. The mask is 0 where
  both are unlabelled or v - o lies outside the block.
  """
  shape = labels.shape
  target = np.zeros(shape + (len(offsets),), np.float32)
  mask = np.zeros(shape + (len(offsets),), np.float32)
  for c, o in enumerate(offsets):
    here = tuple(slice(d, None) for d in o)
    there = tuple(slice(0, n - d) for d, n in zip(o, shape))
    a, b = labels[here], labels[there]
    target[here + (c,)] = (a == b) & (a != 0)
    mask[here + (c,)] = (a != 0) | (b != 0)
  return target, mask


def training_targets(labels: np.ndarray, offsets=OFFSETS):
  """(target, mask), each (z, y, x, len(offsets) + 1) float32: the affinity targets, then
  the neuron target (label > 0) with mask 1 everywhere, since label 0 is background.
  Affinities are trained only on pairs of two neuron voxels (same neuron 1, different
  neurons 0): background is the neuron channel's job, and segmentation reads affinities
  only between neuron voxels. Otherwise neuron-background pairs, 98.9% of crop 12's
  boundary pairs, would take nearly all of the negative weight."""
  t, m = affinity_targets(labels, offsets)
  for c, o in enumerate(offsets):
    here = tuple(slice(d, None) for d in o)
    there = tuple(slice(0, n - d) for d, n in zip(o, labels.shape))
    m[here + (c,)] *= (labels[here] != 0) & (labels[there] != 0)
  fg = (labels > 0).astype(np.float32)[..., None]
  return np.concatenate([t, fg], -1), np.concatenate([m, np.ones_like(fg)], -1)


def balanced_bce(logits, target, mask):
  """Masked binary cross-entropy; within each channel, positives and negatives get equal
  total weight (each half of one channel's unit), averaged over channels."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top
  axes = tuple(range(logits.ndim - 1))
  pos = target * mask
  neg = (1.0 - target) * mask
  w = 0.5 * pos / jnp.maximum(pos.sum(axes), 1.0) + 0.5 * neg / jnp.maximum(neg.sum(axes), 1.0)
  return jnp.sum(w * optax.sigmoid_binary_cross_entropy(logits, target)) / logits.shape[-1]


# ---------------------------------------------------------------- model

def make_model(block=BLOCK, patch=PATCH, width=WIDTH, depth=DEPTH, heads=HEADS, encoder=ENCODER,
               channels=CHANNELS):
  """The hybrid UNETR/TransUNet-style network: (b, z, y, x, 1) image -> (b, z, y, x, channels)
  logits. Convolutional encoder levels at full, 1/2, 1/4 ... resolution (`encoder`
  channels) feed skip connections; the last level is cut into patch tokens for a ViT with
  global attention over the whole block; a convolutional decoder upsamples back through
  the skips. Every convolution is followed by a LayerNorm over channels, and the token
  stream is layer-normalised after the patch embedding (before the position embedding is
  added) and before the decoder (revision 3: in run 1 activations grew to the float16
  limit)."""
  import flax.linen as nn  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  reduce = patch // 2 ** (len(encoder) - 1)  # token stride on the last encoder level
  if block % patch or reduce < 1 or reduce * 2 ** (len(encoder) - 1) != patch:
    raise ValueError('block must be a multiple of patch, and patch = r * 2 ** (len(encoder) - 1)')
  dt = jnp.float16
  grid = block // patch

  def conv(ch, y, stride=1):  # convolution, LayerNorm over channels, ReLU (bounded activations)
    return nn.relu(nn.LayerNorm(dtype=dt)(nn.Conv(ch, (3, 3, 3), strides=(stride,) * 3, dtype=dt)(y)))

  class Encoder(nn.Module):

    @nn.compact
    def __call__(self, t):
      y = nn.LayerNorm(dtype=dt)(t)
      t = t + nn.MultiHeadDotProductAttention(num_heads=heads, dtype=dt)(y, y)
      y = nn.LayerNorm(dtype=dt)(t)
      y = nn.Dense(width, dtype=dt)(nn.gelu(nn.Dense(4 * width, dtype=dt)(y)))
      return t + y

  class ViTSeg(nn.Module):

    @nn.compact
    def __call__(self, x):
      skips, y = [], x
      for i, ch in enumerate(encoder):
        y = conv(ch, conv(ch, y, stride=2 if i else 1))
        skips.append(y)
      t = nn.Conv(width, (reduce,) * 3, strides=(reduce,) * 3, padding='VALID', dtype=dt)(y)
      b = t.shape[0]
      t = t.reshape(b, grid ** 3, width)
      t = nn.LayerNorm(dtype=dt)(t) + self.param('pos', nn.initializers.normal(0.02), (1, grid ** 3, width)).astype(dt)
      for _ in range(depth):
        t = Encoder()(t)
      t = nn.LayerNorm(dtype=dt)(t)
      y = t.reshape(b, grid, grid, grid, width)
      factor = reduce
      for ch, skip in zip(reversed(encoder), reversed(skips)):
        y = nn.relu(nn.LayerNorm(dtype=dt)(nn.ConvTranspose(ch, (factor,) * 3, strides=(factor,) * 3, dtype=dt)(y)))
        y = conv(ch, conv(ch, jnp.concatenate([y, skip.astype(dt)], -1)))
        factor = 2
      return nn.Conv(channels, (1, 1, 1), dtype=jnp.float32)(y).astype(jnp.float32)

  return ViTSeg()


# ---------------------------------------------------------------- training

def augment(img: np.ndarray, lab: np.ndarray, rng: np.random.Generator):
  """Flips in z, y, x; y-x transposition; intensity scale and shift (image only)."""
  for axis in range(3):
    if rng.random() < 0.5:
      img, lab = np.flip(img, axis), np.flip(lab, axis)
  if rng.random() < 0.5:
    img, lab = np.swapaxes(img, 1, 2), np.swapaxes(lab, 1, 2)
  img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
  return np.ascontiguousarray(img, np.float32), np.ascontiguousarray(lab)


def sample_batch(images: list, labels: list, step: int, block=BLOCK, batch=BATCH, seed=RNG_SEED):
  """Batch `step`: random blocks from random training crops. Depends only on (seed, step),
  so a resumed run draws the same batches."""
  rng = np.random.default_rng([seed, step])
  xs, ts, ms = [], [], []
  for _ in range(batch):
    k = int(rng.integers(len(images)))
    corner = [int(rng.integers(n - block + 1)) for n in images[k].shape]
    sl = tuple(slice(c, c + block) for c in corner)
    img, lab = augment(images[k][sl], labels[k][sl], rng)
    t, m = training_targets(lab)
    xs.append(img[..., None])
    ts.append(t)
    ms.append(m)
  return np.stack(xs), np.stack(ts), np.stack(ms)


def validation_batch(image: np.ndarray, labels: np.ndarray, n=VALIDATION_BLOCKS, block=BLOCK, seed=RNG_SEED):
  """n fixed, unaugmented blocks of the tuning crop for checkpoint selection."""
  rng = np.random.default_rng([seed, VALIDATION_STREAM])
  xs, ts, ms = [], [], []
  for _ in range(n):
    sl = tuple(slice(c, c + block) for c in (int(rng.integers(s - block + 1)) for s in image.shape))
    t, m = training_targets(labels[sl])
    xs.append(np.ascontiguousarray(image[sl], np.float32)[..., None])
    ts.append(t)
    ms.append(m)
  return np.stack(xs), np.stack(ts), np.stack(ms)


def make_eval_loss(model):
  import jax  # pylint: disable=g-import-not-at-top
  return jax.jit(lambda params, x, t, m: balanced_bce(model.apply({'params': params}, x), t, m))


def make_optimizer(total_steps: int):
  import optax  # pylint: disable=g-import-not-at-top
  warmup = min(WARMUP, max(1, total_steps // 10))  # WARMUP for real runs; shorter for tiny ones
  schedule = optax.warmup_cosine_decay_schedule(0.0, PEAK_LR, warmup, total_steps, 0.0)
  return optax.chain(optax.clip_by_global_norm(CLIP_NORM), optax.adamw(schedule, weight_decay=WEIGHT_DECAY))


def make_train_step(model, optimizer, loss_scale=LOSS_SCALE):
  """Jitted (params, opt_state, x, t, m) -> (params, opt_state, loss, finite). A step with a
  non-finite gradient leaves params and optimiser state unchanged."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top

  @jax.jit
  def step(params, opt_state, x, t, m):
    def scaled(p):
      return balanced_bce(model.apply({'params': p}, x), t, m) * loss_scale
    loss, grads = jax.value_and_grad(scaled)(params)
    grads = jax.tree_util.tree_map(lambda g: g / loss_scale, grads)
    finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
    updates, new_opt = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(finite, a, b), new, old)
    return keep(new_params, params), keep(new_opt, opt_state), loss / loss_scale, finite

  return step


def stop_reason(consecutive_skips: int, validation_loss=None):
  """Why training must stop now, or None: MAX_CONSECUTIVE_SKIPS non-finite steps in a row,
  or a non-finite validation loss (run 1 kept skipping every step for 25,000 steps)."""
  if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
    return f'{consecutive_skips} consecutive non-finite steps'
  if validation_loss is not None and not np.isfinite(validation_loss):
    return 'non-finite validation loss'
  return None


# ---------------------------------------------------------------- inference

def tile_starts(n: int, block=BLOCK, stride=STRIDE) -> list:
  """Block starts along one axis: every `stride`, and a last block ending at the face."""
  if n < block:
    raise ValueError(f'axis of {n} voxels is smaller than the block ({block})')
  starts = list(range(0, n - block + 1, stride))
  if starts[-1] != n - block:
    starts.append(n - block)
  return starts


def tile_ownership(n: int, block=BLOCK, stride=STRIDE) -> list:
  """(start, own_lo, own_hi) per block: each voxel is owned by exactly one block, split at
  the middle of each overlap."""
  starts = tile_starts(n, block, stride)
  cuts = [0] + [(a + block + b) // 2 for a, b in zip(starts, starts[1:])] + [n]
  return [(s, cuts[i], cuts[i + 1]) for i, s in enumerate(starts)]


def predict_affinities(apply_fn, image_zyx: np.ndarray, block=BLOCK, stride=STRIDE) -> np.ndarray:
  """Sigmoid affinities (z, y, x, C) float32 for a whole crop, one block per call.
  apply_fn maps a (1, block, block, block, 1) float32 array to logits."""
  tiles = [tile_ownership(n, block, stride) for n in image_zyx.shape]
  out = None
  for z0, zl, zh in tiles[0]:
    for y0, yl, yh in tiles[1]:
      for x0, xl, xh in tiles[2]:
        x = image_zyx[z0:z0 + block, y0:y0 + block, x0:x0 + block][None, ..., None]
        logits = np.asarray(apply_fn(np.ascontiguousarray(x, np.float32)))[0]
        if out is None:
          out = np.empty(image_zyx.shape + (logits.shape[-1],), np.float32)
        out[zl:zh, yl:yh, xl:xh] = logits[zl - z0:zh - z0, yl - y0:yh - y0, xl - x0:xh - x0]
  from scipy.special import expit  # pylint: disable=g-import-not-at-top
  return expit(out)


# ---------------------------------------------------------------- segmentation

def neuron_mask(aff: np.ndarray) -> np.ndarray:
  """Voxels predicted to belong to a neuron (neuron channel >= NEURON). Input without the
  neuron channel (the six affinities only) counts every voxel as neuron."""
  if aff.shape[-1] <= NEURON_CHANNEL:
    return np.ones(aff.shape[:3], bool)
  return aff[..., NEURON_CHANNEL] >= NEURON


def boundary_map(aff: np.ndarray) -> np.ndarray:
  """b(v) = 1 - the lowest affinity among the (up to six) nearest-neighbour edges incident
  to v whose two voxels are both neuron voxels: the three stored at v (pairs v, v - o) and
  the three stored at v + o (pairs v + o, v). Background is the neuron channel's job, so
  edges to background voxels do not count. The lowest edge, not the mean: where two
  neurons touch directly, a face voxel has one cross edge among six, and a mean would call
  it confident (b = 1/6) and join both neurons in one fragment. b = 0 where a voxel has
  no such edge."""
  neuron = neuron_mask(aff)
  low = np.ones(aff.shape[:3], np.float32)
  for c, o in enumerate(OFFSETS[:NEAREST]):
    here = tuple(slice(d, None) for d in o)
    there = tuple(slice(0, n - d) for d, n in zip(o, aff.shape[:3]))
    v = np.where(neuron[here] & neuron[there], aff[here + (c,)], 1.0)
    np.minimum(low[here], v, out=low[here])
    np.minimum(low[there], v, out=low[there])
  return (1.0 - low).astype(np.float32)


def confident_mask(aff: np.ndarray) -> np.ndarray:
  """Neuron voxels away from neuron-neuron boundaries: neuron and b < FOREGROUND."""
  return neuron_mask(aff) & (boundary_map(aff) < FOREGROUND)


def fragments(aff: np.ndarray) -> np.ndarray:
  """Fragments of the neuron voxels; background stays 0. Markers are the connected local
  maxima (within PEAK_SIZE) of the anisotropic distance transform of the confident voxels;
  a connected piece of neuron voxels without such a marker becomes one marker itself, so
  every neuron voxel is reached. The watershed of the boundary map is restricted to the
  neuron voxels (labels 1..n, uint32; 0 = background)."""
  from scipy import ndimage  # pylint: disable=g-import-not-at-top
  from skimage.segmentation import watershed  # pylint: disable=g-import-not-at-top
  neuron = neuron_mask(aff)
  out = np.zeros(aff.shape[:3], np.uint32)
  if not neuron.any():
    return out
  b = boundary_map(aff)
  conf = neuron & (b < FOREGROUND)
  dt = ndimage.distance_transform_edt(conf, sampling=VOXEL_SIZE_ZYX).astype(np.float32)
  peaks = conf & (dt == ndimage.maximum_filter(dt, size=PEAK_SIZE))
  markers, n = ndimage.label(peaks, structure=np.ones((3, 3, 3)))
  pieces, _ = ndimage.label(neuron)  # 6-connected, as the watershed floods
  unmarked = np.setdiff1d(np.unique(pieces[neuron]), np.unique(pieces[markers > 0]))
  if len(unmarked):
    lut = np.zeros(int(pieces.max()) + 1, np.int32)
    lut[unmarked] = n + 1 + np.arange(len(unmarked))
    extra = lut[pieces]
    markers = np.where(extra > 0, extra, markers)
  out[...] = watershed(b, markers, mask=neuron)
  return out


def fragment_edges(frag: np.ndarray, aff: np.ndarray):
  """Touching fragment pairs (a < b) with the sum and count of the nearest-neighbour
  affinities over the voxel pairs across their boundary in which both voxels are neuron
  voxels. Contacts through background do not count: on crop 12 with oracle affinities
  they diluted the scores of true continuations (docs/vitseg_design.md). Confidence is not
  required here: b is the lowest incident edge, so requiring it would drop every low
  crossing edge with its own voxels and bias contact scores upwards."""
  fg = neuron_mask(aff) & (frag > 0)
  keys, vals = [], []
  n = int(frag.max()) + 1
  for c, o in enumerate(OFFSETS[:NEAREST]):
    here = tuple(slice(d, None) for d in o)
    there = tuple(slice(0, s - d) for d, s in zip(o, frag.shape))
    a, b = frag[here].ravel().astype(np.int64), frag[there].ravel().astype(np.int64)
    diff = (a != b) & fg[here].ravel() & fg[there].ravel()
    lo, hi = np.minimum(a[diff], b[diff]), np.maximum(a[diff], b[diff])
    keys.append(lo * n + hi)
    vals.append(aff[here + (c,)].ravel()[diff].astype(np.float64))
  keys, vals = np.concatenate(keys), np.concatenate(vals)
  uniq, inv = np.unique(keys, return_inverse=True)
  sums = np.bincount(inv, weights=vals)
  counts = np.bincount(inv)
  return uniq // n, uniq % n, sums, counts


def agglomerate(a, b, sums, counts, min_score=min(THRESHOLDS)) -> list:
  """Mean-affinity agglomeration: repeatedly merge the touching pair with the highest
  mean boundary affinity, pooling sums and counts, until the best is below min_score.
  Returns the merges [(score, u, v)] in order; the result at any threshold t >= min_score
  is the prefix before the first merge with score < t."""
  adj = {}
  for u, v, s, c in zip(a.tolist(), b.tolist(), sums.tolist(), counts.tolist()):
    adj.setdefault(u, {})[v] = [s, c]
    adj.setdefault(v, {})[u] = [s, c]
  heap = [(-s / c, u, v) for u, v, s, c in zip(a.tolist(), b.tolist(), sums.tolist(), counts.tolist())]
  heapq.heapify(heap)
  merges = []
  while heap:
    neg, u, v = heapq.heappop(heap)
    if u not in adj or v not in adj.get(u, {}):
      continue  # stale: one side was merged away
    s, c = adj[u][v]
    if -neg != s / c:
      continue  # stale: the edge was re-scored
    if s / c < min_score:
      break
    merges.append((s / c, u, v))
    if len(adj[u]) < len(adj[v]):
      u, v = v, u  # keep the larger adjacency
    del adj[u][v]
    for w, (ws, wc) in adj.pop(v).items():
      if w == u:
        continue
      del adj[w][v]
      e = adj[u].get(w)
      e = [ws + e[0], wc + e[1]] if e else [ws, wc]
      adj[u][w] = e
      adj[w][u] = e
      heapq.heappush(heap, (-e[0] / e[1], min(u, w), max(u, w)))
  return merges


def segmentation_at(frag: np.ndarray, merges: list, threshold: float) -> np.ndarray:
  """The segmentation after the merges before the first one scoring below threshold."""
  n = int(frag.max()) + 1
  parent = np.arange(n)

  def find(x):
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  for score, u, v in merges:
    if score < threshold:
      break
    parent[find(v)] = find(u)
  roots = parent.copy()
  while True:  # resolve every chain to its root, vectorised
    nxt = roots[roots]
    if np.array_equal(nxt, roots):
      break
    roots = nxt
  return roots[frag]


# ---------------------------------------------------------------- scoring

def overlap_table(seg: np.ndarray, gt: np.ndarray):
  """(segment id, gt id) -> voxel count, as parallel arrays, plus segment and gt sizes."""
  s = seg.ravel().astype(np.int64)
  g = gt.ravel().astype(np.int64)
  ng = int(g.max()) + 1
  pairs, counts = np.unique(s * ng + g, return_counts=True)
  return pairs // ng, pairs % ng, counts


def seed_records(seg: np.ndarray, gt: np.ndarray, seeds: list) -> list:
  """benchmark.score of the segment containing each seed, computed from one overlap table
  (identical results, much faster than one full scan per seed). Segment 0 is predicted
  background, not a segment: a seed there gets benchmark.score of an empty mask (recall
  and precision 0)."""
  from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
  sid, gid, cnt = overlap_table(seg, gt)
  gt_size = {}
  for g, c in zip(gid.tolist(), cnt.tolist()):
    gt_size[g] = gt_size.get(g, 0) + c
  by_seg = {}
  for s, g, c in zip(sid.tolist(), gid.tolist(), cnt.tolist()):
    by_seg.setdefault(s, {})[g] = c
  out = []
  for seed in seeds:
    z = tuple(int(v) for v in seed['seed_zyx'])
    g_id = int(gt[z])
    if g_id == 0:
      raise ValueError(f'seed {z} is on an unlabeled voxel')
    overlap = by_seg[int(seg[z])] if seg[z] else {}
    m = sum(overlap.values())
    tp = overlap.get(g_id, 0)
    unlabeled = overlap.get(0, 0)
    labeled = m - unlabeled
    others = {k: v for k, v in overlap.items() if k not in (0, g_id)}
    merges = sorted(([int(k), int(v)] for k, v in others.items() if v >= benchmark.MERGE_MIN_VOXELS),
                    key=lambda kv: -kv[1])
    out.append({'name': seed.get('name'), 'crop': seed.get('crop'), 'class': seed.get('class'),
                'seed_zyx': list(z), 'object_id': g_id, 'mask_voxels': m, 'correct_voxels': tp,
                'object_voxels': gt_size[g_id], 'recall': tp / gt_size[g_id],
                'precision': tp / labeled if labeled else 0.0,
                'unlabeled_fraction': unlabeled / m if m else 0.0,
                'merged_voxels': int(sum(others.values())), 'merges': merges})
  return out


def background_scores(seg: np.ndarray, gt: np.ndarray) -> dict:
  """Voxel-level agreement of predicted background (segment 0) with label 0."""
  pred, true = seg == 0, gt == 0
  both = int((pred & true).sum())
  return {'voxels': int(gt.size), 'label0_fraction': float(true.mean()), 'predicted_fraction': float(pred.mean()),
          'precision': both / int(pred.sum()) if pred.any() else 0.0,
          'recall': both / int(true.sum()) if true.any() else 0.0}


def large_objects(gt: np.ndarray) -> list:
  """Objects that pass the seed size rule of benchmark.candidates, of any class: at least
  MIN_INTERIOR_VOXELS inside the margin and at most MAX_CROP_VOXELS in the crop (sorted IDs)."""
  from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
  m = benchmark.MARGIN
  ids, total = np.unique(gt, return_counts=True)
  ui, inner = np.unique(gt[m:-m, m:-m, m:-m], return_counts=True)
  total = dict(zip(ids.tolist(), total.tolist()))
  return [int(i) for i, c in zip(ui.tolist(), inner.tolist())
          if i != 0 and c >= benchmark.MIN_INTERIOR_VOXELS and total[i] <= benchmark.MAX_CROP_VOXELS]


def tuning_objects(gt: np.ndarray) -> list:
  """The large objects of the tuning crop, as seeds at their benchmark.seed_point."""
  from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
  return [{'name': f't{i}', 'object_id': i, 'seed_zyx': list(benchmark.seed_point(gt, i))}
          for i in large_objects(gt)]


def mean_f1(records: list) -> float:
  f = [2 * r['precision'] * r['recall'] / (r['precision'] + r['recall']) if r['precision'] + r['recall'] else 0.0
       for r in records]
  return float(np.mean(f))


def choose_threshold(frag, merges, gt, objects, thresholds=THRESHOLDS):
  """(best threshold, table): the highest mean F1 over the tuning objects; ties to the
  higher threshold (fewer merges)."""
  table = []
  for t in thresholds:
    recs = seed_records(segmentation_at(frag, merges, t), gt, objects)
    table.append({'threshold': t, 'mean_f1': mean_f1(recs),
                  'mean_recall': float(np.mean([r['recall'] for r in recs])),
                  'mean_precision': float(np.mean([r['precision'] for r in recs]))})
  best = max(table, key=lambda r: (r['mean_f1'], r['threshold']))
  return best['threshold'], table


# ---------------------------------------------------------------- verdict

def verdict(pairs: list) -> dict:
  """Tiered verdict over (FFN record, ViT record) pairs (docs/vitseg_design.md)."""
  n = len(pairs)
  if n == 0:
    return {'seeds': 0, 'verdict': None}
  ev = [manyseed.events(r, c) for r, c in pairs]
  d_rec = float(np.mean([c['recall'] for _, c in pairs]) - np.mean([r['recall'] for r, _ in pairs]))
  d_prec = float(np.mean([c['precision'] for _, c in pairs]) - np.mean([r['precision'] for r, _ in pairs]))
  count = lambda k: int(sum(e[k] for e in ev))
  net_merges = count('new_merge') - count('avoided_merge')
  net_early = count('new_early_stop') - count('extension')

  def meets(tier):
    drop, net = tier
    return (d_rec >= -drop - manyseed.EPS and d_prec >= -drop - manyseed.EPS
            and net_merges <= net and net_early <= net)

  result = 'COMPETITIVE' if meets(COMPETITIVE) else 'PROMISING' if meets(PROMISING) else 'FAIL'
  return {'seeds': n, 'mean_recall_change': d_rec, 'mean_precision_change': d_prec,
          'new_merges': count('new_merge'), 'avoided_merges': count('avoided_merge'),
          'new_early_stops': count('new_early_stop'), 'extensions': count('extension'),
          'net_new_merges': net_merges, 'net_new_early_stops': net_early, 'verdict': result}


def segment_crop(apply_fn, raw_xyz: np.ndarray):
  """(fragments, merges, timings) for one crop."""
  timings = {}
  t = time.perf_counter()
  aff = predict_affinities(apply_fn, normalize_zyx(raw_xyz))
  timings['network_seconds'] = time.perf_counter() - t
  t = time.perf_counter()
  frag = fragments(aff)
  timings['watershed_seconds'] = time.perf_counter() - t
  t = time.perf_counter()
  merges = agglomerate(*fragment_edges(frag, aff))
  timings['agglomeration_seconds'] = time.perf_counter() - t
  timings['fragments'] = int(frag.max())
  return frag, merges, timings
