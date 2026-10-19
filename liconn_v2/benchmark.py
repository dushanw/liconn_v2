"""FFN inference benchmark against the authors' proofread segmentation.

See docs/benchmark_design.md for the definitions used here.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.special import logit

from liconn_v2 import reference

CLASSES_FILE = Path(__file__).with_name('proofread_classes.json')
MARGIN = 16  # FFN FOV half-size: seeds must be this far from every crop face.
MIN_INTERIOR_VOXELS = 20_000
MAX_CROP_VOXELS = 350_000
SEEDS_PER_CLASS = 6
RNG_SEED = 0
MERGE_MIN_VOXELS = 1_000
NOTEBOOK_SEED_NAME = 'notebook'


def load_classes() -> dict:
  c = json.loads(CLASSES_FILE.read_text())
  return {'axons': c['axons'], 'dendrites': c['dendrites']}


def candidates(gt_zyx: np.ndarray, ids) -> list:
  """Class members that fit the size rules (sorted by ID)."""
  ids = np.array(sorted(ids), dtype=gt_zyx.dtype)
  u, total = np.unique(gt_zyx, return_counts=True)
  interior = gt_zyx[MARGIN:-MARGIN, MARGIN:-MARGIN, MARGIN:-MARGIN]
  ui, inner = np.unique(interior, return_counts=True)
  total = dict(zip(u.tolist(), total.tolist()))
  inner = dict(zip(ui.tolist(), inner.tolist()))
  return [int(i) for i in ids.tolist()
          if total.get(i, 0) <= MAX_CROP_VOXELS and inner.get(i, 0) >= MIN_INTERIOR_VOXELS]


def seed_point(gt_zyx: np.ndarray, obj_id: int) -> tuple:
  """Deepest interior voxel of the object (anisotropic EDT), inside the FOV margin."""
  mask = gt_zyx == obj_id
  if not mask[MARGIN:-MARGIN, MARGIN:-MARGIN, MARGIN:-MARGIN].any():
    raise ValueError(f'object {obj_id} has no voxels inside the FOV margin')
  zz, yy, xx = np.nonzero(mask)
  blo = np.array([zz.min(), yy.min(), xx.min()])
  bhi = np.array([zz.max(), yy.max(), xx.max()]) + 1
  # Bounding box plus one zero voxel on every side: exact inside the crop, and the
  # crop faces count as object boundary.
  box = np.zeros(tuple(bhi - blo + 2), bool)
  box[1:-1, 1:-1, 1:-1] = mask[blo[0]:bhi[0], blo[1]:bhi[1], blo[2]:bhi[2]]
  dist = distance_transform_edt(box, sampling=reference.VOXEL_SIZE_ZYX)[1:-1, 1:-1, 1:-1]
  # Restrict to the FOV margin.
  lo = np.maximum(blo, MARGIN) - blo
  hi = np.minimum(bhi, np.array(mask.shape) - MARGIN) - blo
  inner = dist[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
  # argmax returns the first maximum in C order, i.e. the lowest (z, y, x).
  pos = np.array(np.unravel_index(np.argmax(inner), inner.shape)) + lo + blo
  assert mask[tuple(pos)], (obj_id, pos)
  return tuple(int(v) for v in pos)


def select_seeds(gt_zyx: np.ndarray, classes: dict | None = None) -> list:
  """Deterministic seeds: up to 6 axons, up to 6 dendrites, then the notebook seed."""
  classes = load_classes() if classes is None else classes
  rng = np.random.default_rng(RNG_SEED)
  seeds = []
  for cls in ('axons', 'dendrites'):
    pool = candidates(gt_zyx, classes[cls])
    # Only 5 dendrites qualify in this crop; then all of them are used.
    chosen = rng.choice(pool, size=min(SEEDS_PER_CLASS, len(pool)), replace=False)
    for obj_id in chosen.tolist():
      seeds.append({'name': f'{cls[:-1]}_{obj_id}', 'class': cls[:-1], 'object_id': int(obj_id),
                    'seed_zyx': seed_point(gt_zyx, int(obj_id)), 'candidates': len(pool)})
  z, y, x = reference.SEED_XYZ[::-1]
  seeds.append({'name': NOTEBOOK_SEED_NAME, 'class': 'unclassified',
                'object_id': int(gt_zyx[z, y, x]), 'seed_zyx': (z, y, x), 'candidates': None})
  return seeds


def object_mask(canvas: np.ndarray) -> np.ndarray:
  """FFN's segment definition (Canvas.segment_all): canvas >= logit(segment_threshold)."""
  return canvas >= np.float32(logit(reference.SEGMENT_THRESHOLD))


def score(mask: np.ndarray, gt_zyx: np.ndarray, seed_zyx) -> dict:
  """Accuracy of one FFN object mask against the proofread labels."""
  g_id = int(gt_zyx[tuple(seed_zyx)])
  if g_id == 0:
    raise ValueError(f'seed {seed_zyx} is on an unlabeled voxel')
  labels = gt_zyx[mask]
  ids, counts = np.unique(labels, return_counts=True)
  overlap = dict(zip(ids.tolist(), counts.tolist()))
  m = int(mask.sum())
  tp = overlap.get(g_id, 0)
  unlabeled = overlap.get(0, 0)
  labeled = m - unlabeled
  g_size = int((gt_zyx == g_id).sum())
  others = {k: v for k, v in overlap.items() if k not in (0, g_id)}
  merges = sorted(([int(k), int(v)] for k, v in others.items() if v >= MERGE_MIN_VOXELS),
                  key=lambda kv: -kv[1])
  return {
      'object_id': g_id,
      'mask_voxels': m,
      'correct_voxels': tp,
      'object_voxels': g_size,
      'recall': tp / g_size,
      'precision': tp / labeled if labeled else 0.0,
      'unlabeled_fraction': unlabeled / m if m else 0.0,
      'merged_voxels': int(sum(others.values())),
      'merges': merges,
  }


def cost(steps: int, correct_voxels: int, gflop_per_eval: float) -> dict:
  return {
      'steps': steps,
      'network_gflop': steps * gflop_per_eval,
      'evals_per_1000_correct_voxels': 1000 * steps / correct_voxels if correct_voxels else None,
      'gflop_per_correct_voxel': steps * gflop_per_eval / correct_voxels if correct_voxels else None,
  }


def summarize(records: list) -> dict:
  def block(rs):
    if not rs:
      return None
    steps = sum(r['steps'] for r in rs)
    correct = sum(r['correct_voxels'] for r in rs)
    return {
        'seeds': len(rs),
        'mean_recall': float(np.mean([r['recall'] for r in rs])),
        'mean_precision': float(np.mean([r['precision'] for r in rs])),
        'mean_unlabeled_fraction': float(np.mean([r['unlabeled_fraction'] for r in rs])),
        'total_merges': sum(len(r['merges']) for r in rs),
        'total_merged_voxels': sum(r['merged_voxels'] for r in rs),
        'total_steps': steps,
        'total_network_gflop': sum(r['network_gflop'] for r in rs),
        'total_correct_voxels': correct,
        'evals_per_1000_correct_voxels': 1000 * steps / correct if correct else None,
        'gflop_per_correct_voxel': sum(r['network_gflop'] for r in rs) / correct if correct else None,
        'wall_seconds': sum(r['trace_seconds'] for r in rs),
    }
  out = {'all': block(records)}
  for cls in ('axon', 'dendrite', 'unclassified'):
    out[cls] = block([r for r in records if r['class'] == cls])
  return out


def run(seeds: list, raw_zyx: np.ndarray, gt_zyx: np.ndarray, ckpt_path: str, on_record=None):
  """Traces each seed with the authors' settings and scores it. Returns (records, masks)."""
  records, masks = [], {}
  with reference.Tracer(ckpt_path) as tracer:
    gflop = reference.gflop_per_eval(tracer.params)
    for s in seeds:
      trace = tracer.trace(raw_zyx, s['seed_zyx'])
      mask = object_mask(trace.seed)
      rec = {'name': s['name'], 'class': s['class'], 'seed_zyx': list(s['seed_zyx'])}
      rec.update(score(mask, gt_zyx, s['seed_zyx']))
      rec.update(cost(trace.iterations, rec['correct_voxels'], gflop))
      rec.update({'trace_seconds': trace.trace_seconds, 'gflop_per_eval': gflop,
                  'number_format': 'float32', 'backend': trace.backend})
      records.append(rec)
      masks[s['name']] = mask
      if on_record is not None:
        on_record(rec)
  return records, masks
