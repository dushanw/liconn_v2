"""SegNeuron fine-tuned on LICONN crops 0-11 (docs/segneuron_finetune_design.md): the parts that
need only numpy, so they are tested without torch.

The training loop (examples/train_segneuron.py) runs in its own torch environment and imports
the authors' network, targets, loss and augmentations from the pinned SegNeuron checkout. It
reads the crops written here by `write_crops` as plain .npy files (z, y, x).
"""

import hashlib
import json
from pathlib import Path

import numpy as np

# No liconn_v2 imports: this module also runs in the torch training environment.
TRAIN_CROPS = tuple(range(12))  # 0-11, as vitseg.TRAIN_CROPS (the ViT's training crops)
VALID_CROP = 12  # vitseg.TUNE_CROP: checkpoint choice, then threshold and beta, then scoring
PATCH = (20, 128, 128)  # z, y, x; the authors' training patch
VALID_PATCHES = 32
VALID_SEED = 20261028
BASE_LR = 1e-3
END_LR = 1e-5
POWER = 1.5  # the authors' polynomial decay exponent
VALID_EVERY = 1000
RAW = 'crop-{:02d}_raw.npy'
LABELS = 'crop-{:02d}_labels.npy'


def learning_rate(step: int, total: int, base=BASE_LR, end=END_LR, power=POWER) -> float:
  """The authors' calculate_lr (supervised_train.py) with no warm-up and decay over the whole
  run: (base - end) * (1 - step / total) ** power + end, and end after the last step."""
  if step >= total:
    return end
  return (base - end) * (1.0 - step / total) ** power + end


def relabel(labels: np.ndarray) -> np.ndarray:
  """Labels as consecutive uint32 ids with 0 kept as 0; the same partition of the voxels."""
  ids, inverse = np.unique(labels, return_inverse=True)
  if ids[0] != 0:
    inverse = inverse + 1
  if inverse.max() >= 2**32:
    raise ValueError('too many objects for uint32')
  return inverse.reshape(labels.shape).astype(np.uint32)


def compact_patch(labels: np.ndarray) -> np.ndarray:
  """A patch's labels renumbered 0, 1, ..., K (0 kept) before augmentation: the authors'
  elastic augmentation writes labels as uint16 (utils/augmentation.py), so larger ids would wrap
  and could merge two neurons."""
  r = relabel(labels)
  if r.max() >= 2**16:
    raise ValueError('more than 65535 objects in one patch')
  return r


def boundary_errors(pred: np.ndarray, labels: np.ndarray) -> dict:
  """Pair counts on predicted affinities (3, z, y, x), pairs (v, v - o) for o = z, y, x as in the
  authors' targets, pairs leaving the patch not counted: same-neuron pairs (one nonzero label)
  below 0.5 and different-neuron pairs (two different nonzero labels) at or above 0.5. A
  diagnostic of the boundaries between touching neurons; it does not choose the checkpoint."""
  out = {'same': 0, 'same_below_0.5': 0, 'diff': 0, 'diff_at_or_above_0.5': 0}
  for c in range(3):
    here = [slice(None)] * 3
    there = [slice(None)] * 3
    here[c], there[c] = slice(1, None), slice(None, -1)
    a, b, p = labels[tuple(here)], labels[tuple(there)], pred[c][tuple(here)]
    same = (a == b) & (a > 0)
    diff = (a != b) & (a > 0) & (b > 0)
    out['same'] += int(same.sum())
    out['same_below_0.5'] += int((same & (p < 0.5)).sum())
    out['diff'] += int(diff.sum())
    out['diff_at_or_above_0.5'] += int((diff & (p >= 0.5)).sum())
  return out


def write_crops(out: Path, load_raw_zyx, load_labels_zyx, crops=TRAIN_CROPS + (VALID_CROP,)) -> dict:
  """Writes each crop's raw image (uint8) and relabelled proofread labels (uint32), both
  (z, y, x), and returns a manifest with their sha256."""
  out.mkdir(parents=True, exist_ok=True)
  manifest = {'sha256': {}, 'crops': {}}
  for k in crops:
    raw, lab = load_raw_zyx(k), relabel(load_labels_zyx(k))
    if raw.dtype != np.uint8 or raw.shape != lab.shape:
      raise ValueError(f'crop {k}: raw {raw.dtype} {raw.shape}, labels {lab.shape}')
    manifest['crops'][str(k)] = {'shape_zyx': list(lab.shape), 'objects': int(lab.max()),
                                 'labelled_fraction': float((lab > 0).mean())}
    for name, arr in ((RAW.format(k), raw), (LABELS.format(k), lab)):
      np.save(out / name, np.ascontiguousarray(arr), allow_pickle=False)
      manifest['sha256'][name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
  (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
  return manifest


def corner(rng: np.random.Generator, shape, patch=PATCH) -> tuple:
  """A uniformly random patch corner inside a volume."""
  if any(p > s for p, s in zip(patch, shape)):
    raise ValueError(f'patch {patch} larger than volume {shape}')
  return tuple(int(rng.integers(0, s - p + 1)) for s, p in zip(shape, patch))


def train_draw(rng: np.random.Generator, shapes: dict, patch=PATCH) -> tuple:
  """(crop, corner): a training crop chosen uniformly, then a patch corner in it."""
  crops = sorted(shapes)
  if VALID_CROP in crops:
    raise ValueError('the validation crop is not a training crop')
  k = crops[int(rng.integers(len(crops)))]
  return k, corner(rng, shapes[k], patch)


def valid_corners(shape, n=VALID_PATCHES, seed=VALID_SEED, patch=PATCH) -> list:
  """Fixed patch corners in the validation crop, the same for every run."""
  rng = np.random.default_rng(seed)
  return [corner(rng, shape, patch) for _ in range(n)]


def crop_gap(a_xyz, b_xyz, size=500) -> int:
  """Voxels between two size^3 crops given by their corners (negative: they overlap, 0: they
  touch)."""
  return max(max(b - (a + size), a - (b + size)) for a, b in zip(a_xyz, b_xyz))


def best_step(log: list) -> dict:
  """The validation record with the lowest finite loss (ties: the earlier step)."""
  finite = [r for r in log if np.isfinite(r['valid_loss'])]
  if not finite:
    raise ValueError('no finite validation loss')
  return min(finite, key=lambda r: (r['valid_loss'], r['step']))
