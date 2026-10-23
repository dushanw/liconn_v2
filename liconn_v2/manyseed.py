"""Many-seed tolerance: crops, seeds and verdict (docs/manyseed_design.md).

Ten further 500^3 crops of ExPID82_1, chosen by a fixed grid + coverage rule, give 300
fresh seeds. A speed variant is judged by how often it flips a seed's outcome (a new
merge or a new early stop) instead of by per-seed 1% rules.

Fetch the coarse coverage level and the crops with:  .venv/bin/python -m liconn_v2.manyseed
"""

import json
import os
from pathlib import Path

import numpy as np

from liconn_v2 import benchmark
from liconn_v2 import data
from liconn_v2 import heldout

# Crop rule (fixed 2026-10-21 before any new crop was read).
VOLUME_XYZ = (3825, 4870, 795)  # proofread segmentation at 18 x 18 x 24 nm
CROP_SIZE = 500
GRID_X0, GRID_Y0, GRID_STEP, Z0 = 100, 83, 500, 111  # grid through crop A
CROP_A_XYZ = data.CROP_XYZ
CROP_B_XYZ = heldout.CROP_XYZ
COARSE_SCALE_INDEX = 4  # 288 x 288 x 384 nm
COARSE_FACTOR = 16  # coarse voxel edge in 18 x 18 x 24 nm voxels, in x, y and z
MIN_COVERAGE = 0.5
N_CROPS = 10
RNG_SEED = 0

# Seed rule.
SEEDS_PER_CROP = 30
MAX_DENDRITES_PER_CROP = 15
TARGET_SEEDS = 300
# Objects seeded in crop B (docs/heldout_results.md); never seeded again.
CROP_B_SEED_OBJECTS = frozenset({151289, 229311, 197450, 176723, 14447, 242804, 283334,
                                 211590, 217911, 283771, 245099, 250842})

# Verdict (owner-confirmed 2026-10-21).
FLIP = 0.10  # a per-seed change of more than 10 points
MAX_NEW_MERGES = 3
MAX_NEW_EARLY_STOPS = 3
MAX_MEAN_RECALL_DROP = 0.01
MAX_MEAN_PRECISION_DROP = 0.01
CONFIDENCE = 0.95
EPS = 1e-9  # float slack at the limits, as in liconn_v2/compare.py
ALREADY_MERGED_PRECISION = 0.2  # below this the precision rule can barely see a further merge

COARSE_FILE = 'coarse_labeled_xyz.npy'
RAW_FILE = 'raw_xyz_uint8.npy'
PROOFREAD_FILE = '231030_agg_240123_crop_xyz_uint32.npy'
MANIFEST = Path(__file__).resolve().parent.parent / 'docs' / 'manyseed_crops.json'
SEEDS_FILE = Path(__file__).resolve().parent.parent / 'docs' / 'manyseed_seeds.json'


# ---------------------------------------------------------------- crops

def candidate_positions() -> list:
  """Grid positions (x0, y0, z0) of 500^3 crops fully inside the volume, sorted."""
  out = []
  x0 = GRID_X0
  while x0 + CROP_SIZE <= VOLUME_XYZ[0]:
    y0 = GRID_Y0
    while y0 + CROP_SIZE <= VOLUME_XYZ[1]:
      out.append((x0, y0, Z0))
      y0 += GRID_STEP
    x0 += GRID_STEP
  return sorted(out)


def overlaps(a, b, size=CROP_SIZE) -> bool:
  return all(a[i] < b[i] + size and b[i] < a[i] + size for i in range(3))


def coverage(labeled_coarse_xyz: np.ndarray, pos) -> float:
  """Labeled fraction over every coarse cell the crop touches."""
  lo = [p // COARSE_FACTOR for p in pos]
  hi = [-(-(p + CROP_SIZE) // COARSE_FACTOR) for p in pos]
  block = labeled_coarse_xyz[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
  return float(block.mean())


def select_crops(labeled_coarse_xyz: np.ndarray, n: int | None = N_CROPS) -> list:
  """The pre-registered crop choice: qualifying grid crops in rng(0) order, first n (all if None)."""
  qualifying = []
  for pos in candidate_positions():
    if overlaps(pos, CROP_A_XYZ) or overlaps(pos, CROP_B_XYZ):
      continue
    c = coverage(labeled_coarse_xyz, pos)
    if c >= MIN_COVERAGE:
      qualifying.append({'xyz': list(pos), 'coverage': round(c, 4)})
  order = np.random.default_rng(RNG_SEED).permutation(len(qualifying))
  chosen = [qualifying[i] for i in (order if n is None else order[:n])]
  return [{'index': k, **c} for k, c in enumerate(chosen)]


# ---------------------------------------------------------------- cache

def cache_dir() -> Path:
  return data.cache_dir() / 'manyseed'


def crop_dir(k: int) -> Path:
  return cache_dir() / f'crop-{k:02d}'


def load_manifest() -> dict:
  return json.loads(MANIFEST.read_text())


def verify_crop(k: int, manifest: dict | None = None) -> None:
  manifest = load_manifest() if manifest is None else manifest
  for rel, expected in manifest['crops'][k]['sha256'].items():
    path = crop_dir(k) / rel
    if not path.is_file():
      raise FileNotFoundError(f'{path} is missing; run: python -m liconn_v2.manyseed')
    actual = data.sha256_file(path)
    if actual != expected:
      raise ValueError(f'{path} has sha256 {actual}, expected {expected}')


def load_raw_xyz(k: int) -> np.ndarray:
  verify_crop(k)
  return np.load(crop_dir(k) / RAW_FILE)


def load_proofread_zyx(k: int) -> np.ndarray:
  verify_crop(k)
  return np.ascontiguousarray(np.transpose(np.load(crop_dir(k) / PROOFREAD_FILE), [2, 1, 0]))


# ---------------------------------------------------------------- seeds

def excluded_objects() -> set:
  return set(heldout.BENCHMARK_SEED_OBJECTS) | set(CROP_B_SEED_OBJECTS)


def select_seeds_for_crop(gt_zyx: np.ndarray, k: int, excluded, classes: dict | None = None) -> list:
  """Up to 15 dendrites, then axons to 30, by the benchmark rule; rng(0) per crop."""
  classes = benchmark.load_classes() if classes is None else classes
  if max(max(v) for v in classes.values()) >= 2 ** 32 and gt_zyx.dtype == np.uint32:
    raise ValueError('class IDs do not fit the uint32 labels')
  rng = np.random.default_rng(RNG_SEED)
  excluded = set(excluded)
  seeds = []
  for cls, limit in (('dendrites', MAX_DENDRITES_PER_CROP), ('axons', None)):
    pool = [i for i in benchmark.candidates(gt_zyx, classes[cls]) if i not in excluded]
    n = min(limit if limit is not None else SEEDS_PER_CROP - len(seeds), len(pool))
    chosen = rng.choice(pool, size=n, replace=False) if n > 0 else []
    for obj_id in (chosen.tolist() if n > 0 else []):
      seeds.append({'name': f'm{k}_{cls[:-1]}_{obj_id}', 'crop': k, 'class': cls[:-1],
                    'object_id': int(obj_id), 'seed_zyx': [int(v) for v in benchmark.seed_point(gt_zyx, int(obj_id))],
                    'candidates': len(pool)})
  return seeds


def recorded_seeds() -> list:
  return json.loads(SEEDS_FILE.read_text())['seeds']


def seeds_for_crop(k: int, gt_zyx: np.ndarray | None = None) -> list:
  """Crop k's recorded seeds, re-derived from the crop to check they follow the rule."""
  recorded = recorded_seeds()
  mine = [s for s in recorded if s['crop'] == k]
  if gt_zyx is not None:
    earlier = {s['object_id'] for s in recorded if s['crop'] < k}
    again = select_seeds_for_crop(gt_zyx, k, excluded_objects() | earlier)
    last = k == max(s['crop'] for s in recorded)  # the last crop is cut to reach exactly 300
    n_earlier = sum(s['crop'] < k for s in recorded)
    if last and len(mine) != min(len(again), TARGET_SEEDS - n_earlier):
      raise ValueError(f'crop {k}: last crop has {len(mine)} seeds, expected {TARGET_SEEDS - n_earlier}')
    if (again[:len(mine)] if last else again) != mine:
      raise ValueError(f'crop {k}: recorded seeds differ from the seed rule')
  return mine


# ---------------------------------------------------------------- verdict

def upper_bound(k: int, n: int, confidence: float = CONFIDENCE) -> float:
  """One-sided Clopper-Pearson upper bound on a binomial rate."""
  from scipy.stats import beta  # pylint: disable=g-import-not-at-top
  return 1.0 if k >= n else float(beta.ppf(confidence, k + 1, n - k))


def events(ref: dict, cand: dict) -> dict:
  """Per-seed changes (candidate minus reference) and flip events."""
  d_rec = cand['recall'] - ref['recall']
  d_prec = cand['precision'] - ref['precision']
  return {
      'recall_change': d_rec,
      'precision_change': d_prec,
      'new_merge': d_prec < -FLIP - EPS,
      'new_early_stop': d_rec < -FLIP - EPS,
      'avoided_merge': d_prec > FLIP + EPS,
      'extension': d_rec > FLIP + EPS,
      'reference_already_merged': ref['precision'] < ALREADY_MERGED_PRECISION,
      'candidate_empty': cand.get('mask_voxels', 1) == 0,
  }


def verdict(ref_runs: list, cand_runs: list, expected_seeds=None, expected_inputs=None) -> dict:
  """Judges a candidate against the reference over all seeds of the given runs.

  Each run is a loaded benchmark.json (dict with 'seeds', 'records', 'inputs_sha256').
  Runs are paired by position (one per crop). expected_seeds, if given, is the full
  pre-registered seed list (dicts with name, crop, object_id, seed_zyx): every seed must
  be traced, in the run pair of its crop, from its registered point and object, in both
  runs. expected_inputs, if given, is the pinned inputs_sha256 per pair.
  """
  problems = []
  if len(ref_runs) != len(cand_runs):
    problems.append(f'{len(ref_runs)} reference runs but {len(cand_runs)} candidate runs')
  ref_rec, cand_rec, ref_seed, cand_seed, pair_of = {}, {}, {}, {}, {}
  for i, (r, c) in enumerate(zip(ref_runs, cand_runs)):
    if not r.get('inputs_sha256') or r.get('inputs_sha256') != c.get('inputs_sha256'):
      problems.append(f'pair {i}: missing or different inputs')
    elif expected_inputs is not None and r['inputs_sha256'] != expected_inputs[i]:
      problems.append(f'pair {i}: inputs differ from the pinned crop and checkpoint')
    for run, recs, seeds in ((r, ref_rec, ref_seed), (c, cand_rec, cand_seed)):
      for s in run['seeds']:
        seeds[s['name']] = s
      for rec in run['records']:
        if rec['name'] in recs:
          problems.append(f"seed {rec['name']} appears twice")
        recs[rec['name']] = rec
        pair_of.setdefault(rec['name'], set()).add(i)
  registered = {s['name']: s for s in expected_seeds} if expected_seeds is not None else None
  names = list(registered) if registered is not None else sorted(ref_seed)
  for n in (names if registered is not None else []):
    want = registered[n]
    if pair_of.get(n, set()) - {want['crop']}:
      problems.append(f"seed {n}: traced outside the run pair of crop {want['crop']}")
    for label, recs, seeds in (('reference', ref_rec, ref_seed), ('candidate', cand_rec, cand_seed)):
      for what, d in (('record', recs.get(n)), ('seed', seeds.get(n))):
        if d is not None and (d.get('object_id') != want['object_id']
                              or list(d.get('seed_zyx', [])) != list(want['seed_zyx'])):
          problems.append(f'seed {n}: {label} {what} differs from the registered object or point')
  for n in names:
    if n not in ref_rec or n not in cand_rec:
      problems.append(f'seed {n} missing from {"reference" if n not in ref_rec else "candidate"}')
    elif (list(ref_seed[n]['seed_zyx']) != list(cand_seed[n]['seed_zyx'])
          or ref_seed[n]['object_id'] != cand_seed[n]['object_id']):
      problems.append(f'seed {n}: different seed point or object')
  extra = sorted((set(ref_rec) | set(cand_rec)) - set(names))
  if extra:
    problems.append(f'unexpected seeds: {extra}')
  paired = [n for n in names if n in ref_rec and n in cand_rec]
  per_seed = {n: {**events(ref_rec[n], cand_rec[n]), 'class': ref_seed[n].get('class'),
                  'crop': ref_seed[n].get('crop')} for n in paired}
  kinds = ('new_merge', 'new_early_stop', 'avoided_merge', 'extension')

  def count(key, cls=None):
    return sum(1 for e in per_seed.values() if e[key] and (cls is None or e['class'] == cls))

  def mean(key, cls=None):
    v = [e[key] for e in per_seed.values() if cls is None or e['class'] == cls]
    return float(np.mean(v)) if v else 0.0

  def spread(key):
    v = [abs(e[key]) for e in per_seed.values() if abs(e[key]) <= FLIP + EPS]
    return {'median': float(np.median(v)) * 100 if v else None,
            'p95': float(np.percentile(v, 95)) * 100 if v else None}

  n = len(paired)
  n_dend = sum(1 for e in per_seed.values() if e['class'] == 'dendrite')
  mean_rec, mean_prec = mean('recall_change'), mean('precision_change')
  counts = {k: count(k) for k in kinds}
  dend_counts = {k: count(k, 'dendrite') for k in kinds}
  by_crop = {}
  for e in per_seed.values():
    c = by_crop.setdefault(str(e['crop']), {'seeds': 0, **{k: 0 for k in kinds}})
    c['seeds'] += 1
    for k in kinds:
      c[k] += int(e[k])
  rules = {
      'complete_same_inputs': not problems,
      'all_300_seeds': n >= TARGET_SEEDS,  # the limits are counts out of 300
      'new_merges': counts['new_merge'] <= MAX_NEW_MERGES,
      'new_early_stops': counts['new_early_stop'] <= MAX_NEW_EARLY_STOPS,
      'mean_recall_drop': mean_rec >= -MAX_MEAN_RECALL_DROP - EPS,
      'mean_precision_drop': mean_prec >= -MAX_MEAN_PRECISION_DROP - EPS,
  }
  return {
      'design': 'docs/manyseed_design.md',
      'seeds': n,
      'dendrite_seeds': n_dend,
      'counts': counts,
      'dendrite_counts': dend_counts,
      'upper_bound_95': {
          'new_merge_rate': upper_bound(counts['new_merge'], n) if n else None,
          'new_early_stop_rate': upper_bound(counts['new_early_stop'], n) if n else None,
          'dendrite_new_merge_rate': upper_bound(dend_counts['new_merge'], n_dend) if n_dend else None,
          'note': 'assumes independent seeds; seeds within a crop are not, so these are optimistic',
      },
      'mean_recall_change': mean_rec,
      'mean_precision_change': mean_prec,
      'dendrite_mean_recall_change': mean('recall_change', 'dendrite'),
      'dendrite_mean_precision_change': mean('precision_change', 'dendrite'),
      'small_change_points': {'recall': spread('recall_change'), 'precision': spread('precision_change')},
      'by_crop': by_crop,
      'reference_already_merged': sorted(k for k, e in per_seed.items() if e['reference_already_merged']),
      'candidate_empty': sorted(k for k, e in per_seed.items() if e['candidate_empty']),
      'rules': rules,
      'passes': all(rules.values()),
      'problems': problems,
      'flipped_seeds': {name: e for name, e in per_seed.items()
                        if e['new_merge'] or e['new_early_stop'] or e['avoided_merge'] or e['extension']},
  }


# ---------------------------------------------------------------- fetch

def _open(path, scale_index):
  os.environ.setdefault('GCE_METADATA_ROOT', 'metadata.google.internal.invalid')
  import tensorstore as ts  # pylint: disable=g-import-not-at-top
  return ts.open({'driver': 'neuroglancer_precomputed', 'kvstore': {'driver': 'gcs', 'bucket': data.BUCKET},
                  'path': path, 'scale_index': scale_index}, read=True).result()[ts.d['channel'][0]]


def fetch_crop(c: dict, image, seg) -> None:
  d = crop_dir(c['index'])
  d.mkdir(parents=True, exist_ok=True)
  x0, y0, z0 = c['xyz']
  sl = (slice(x0, x0 + CROP_SIZE), slice(y0, y0 + CROP_SIZE), slice(z0, z0 + CROP_SIZE))
  if not (d / RAW_FILE).is_file():
    np.save(d / RAW_FILE, image[sl].read().result())
  if not (d / PROOFREAD_FILE).is_file():
    lab = seg[sl].read().result()
    if int(lab.max()) >= 2 ** 32:
      raise ValueError(f'crop {c["index"]}: label ID {int(lab.max())} does not fit uint32')
    np.save(d / PROOFREAD_FILE, lab.astype(np.uint32))
  c['sha256'] = {name: data.sha256_file(d / name) for name in (RAW_FILE, PROOFREAD_FILE)}


def fetch_and_select() -> list:
  """Fetches crops in the pre-registered order until 300 seeds, then records crops and seeds.

  The first 10 crops are always fetched; further crops (owner decision 2026-10-21) only
  while fewer than 300 seeds have been selected. The last crop's seeds are cut, in their
  selection order, to a total of exactly 300.
  """
  root = cache_dir()
  root.mkdir(parents=True, exist_ok=True)
  coarse_path = root / COARSE_FILE
  # Existing pins are checked first and never replaced (review finding, 2026-10-21).
  pinned = load_manifest() if MANIFEST.is_file() else None
  if pinned is not None and coarse_path.is_file() and data.sha256_file(coarse_path) != pinned['coarse_sha256']:
    raise ValueError(f'{coarse_path} differs from its pinned sha256 in {MANIFEST}')
  for c in (pinned or {}).get('crops', []):  # cached crops, before any download
    for name, expected in c['sha256'].items():
      path = crop_dir(c['index']) / name
      if path.is_file() and data.sha256_file(path) != expected:
        raise ValueError(f'{path} differs from its pinned sha256 in {MANIFEST}')
  if not coarse_path.is_file():
    labels = _open(data.PROOFREAD_PATH, COARSE_SCALE_INDEX).read().result()
    np.save(coarse_path, (labels > 0).astype(np.uint8))
  if pinned is not None and data.sha256_file(coarse_path) != pinned['coarse_sha256']:
    raise ValueError(f'{coarse_path} differs from its pinned sha256 in {MANIFEST}')
  order = select_crops(np.load(coarse_path), None)
  image = _open(data.IMAGE_PATH, data.IMAGE_SCALE_INDEX)
  seg = _open(data.PROOFREAD_PATH, 0)
  crops, seeds, excluded = [], [], excluded_objects()
  for c in order:
    if len(crops) >= N_CROPS and len(seeds) >= TARGET_SEEDS:
      break
    fetch_crop(c, image, seg)
    if pinned is not None and c['index'] < len(pinned['crops']) and c != pinned['crops'][c['index']]:
      raise ValueError(f"crop {c['index']} differs from its pinned entry in {MANIFEST}")
    crops.append(c)
    gt = np.ascontiguousarray(np.transpose(np.load(crop_dir(c['index']) / PROOFREAD_FILE), [2, 1, 0]))
    s = select_seeds_for_crop(gt, c['index'], excluded)[:max(0, TARGET_SEEDS - len(seeds))]
    excluded |= {x['object_id'] for x in s}
    seeds += s
    print(f"crop {c['index']} at {c['xyz']} coverage {c['coverage']:.3f}: "
          f"{sum(x['class'] == 'dendrite' for x in s)} dendrites, {sum(x['class'] == 'axon' for x in s)} axons; "
          f"total {len(seeds)}", flush=True)
  manifest = {'design': 'docs/manyseed_design.md', 'coarse_sha256': data.sha256_file(coarse_path), 'crops': crops}
  if pinned is not None and manifest != pinned:
    raise ValueError(f'the crops differ from the pinned {MANIFEST}; nothing was written')
  if SEEDS_FILE.is_file() and json.loads(SEEDS_FILE.read_text())['seeds'] != seeds:
    raise ValueError(f'the seeds differ from the recorded {SEEDS_FILE}; nothing was written')
  MANIFEST.write_text(json.dumps(manifest, indent=1) + '\n')
  SEEDS_FILE.write_text(seeds_text(seeds))
  return seeds


def seeds_text(seeds: list) -> str:
  """The seeds file, one seed per line."""
  return '{"design": %s, "total": %d, "target": %d, "seeds": [\n%s\n]}\n' % (
      json.dumps('docs/manyseed_design.md'), len(seeds), TARGET_SEEDS, ',\n'.join(json.dumps(s) for s in seeds))


if __name__ == '__main__':
  s = fetch_and_select()
  print(f'{len(s)} seeds recorded in {SEEDS_FILE} (target {TARGET_SEEDS})')
