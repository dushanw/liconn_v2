"""The never-read crops 41-50: crops, frozen methods and the pre-stated analysis (docs/unseen_design.md).

Owner decisions (2026-11-04): ten crops; the ten positions of the fixed crop order that no earlier
study fetched (41-50); every method, FFN fast mode against float32 included; crops 13-40 stay in
reserve. Nothing here is chosen on these crops: the methods, checkpoints, thresholds, rules and
margins below are fixed in the design before any of them is read.

Fetch with:  .venv/bin/python -m liconn_v2.unseen   (only after the owner's go-ahead)
"""

import json
from pathlib import Path

import numpy as np

from liconn_v2 import data
from liconn_v2 import headtohead as hh
from liconn_v2 import manyseed

CROPS = tuple(range(41, 51))
# Positions 41-50 of manyseed.select_crops(coarse, None), as listed in the design.
CROP_XYZ = {41: (100, 4083, 111), 42: (2600, 1583, 111), 43: (3100, 4083, 111), 44: (100, 3083, 111),
            45: (600, 3583, 111), 46: (1600, 3583, 111), 47: (2600, 3583, 111), 48: (2100, 1583, 111),
            49: (600, 4083, 111), 50: (2100, 583, 111)}
USED_BEFORE = 41  # crops 0-40: docs/manyseed_crops.json (0-12) and docs/fresh_crops.json (13-40)

DOCS = Path(__file__).resolve().parent.parent / 'docs'
DESIGN = DOCS / 'unseen_design.md'
MANIFEST = DOCS / 'unseen_crops.json'

# Frozen methods. B200 runs are those of examples/time_crop11.py (job E, Slurm 27770944).
B200_RUNS = ('F1', 'F5', 'S10', 'P2', 'Q2')
COUNTED_NONFINITE = ('F5', 'S10', 'P2', 'Q2')  # float16 runs; time_crop11 counts non-finite outputs only there
RUN_CHECKPOINT = {'S10': 'student', 'P2': 'p8', 'Q2': 'p4'}
# run: (network, format, objects at once, copy back only the visited box, residual stream scale),
# as time_crop11.RUNS and SCALE; the judge refuses a result file recorded with other settings
RUN_SETTINGS = {
    'F1': ('ffn', 'fp32', 1, True, 1.0),
    'F5': ('ffn', 'fp16_head32', 32, True, 1 / 16),
    'S10': ('student', 'fp16_head32', 32, True, 1 / 16),
    'P2': ('p8', 'fp16_head32', 32, True, 1 / 16),
    'Q2': ('p4', 'fp16_head32', 32, True, 1 / 16),
}
REFERENCE = 'F1'  # FFN float32, one object at a time
FFN_FAST = 'F5'  # FFN fp16_head32, 32 objects at once, stream x 1/16
SMALLER = {'S10': 'student', 'P2': 'P8', 'Q2': 'P4'}
CHECKPOINTS = {  # relative to the MIT run folder distil-20261031; the files job E used
    'student': ('train-dilated-r1/best.pt', '1cef5c3dd75030265621e97b7bf4dbe6c82bf9a4cd0a426a5bc8f53026177ae8'),
    'p8': ('train-p8-r1/best.pt', '01eac60fc0b87b5685f69fa164abe32055f98eff95ebeacb186b58e3303cf055'),
    'p4': ('train-p4-r1/last.pt', 'fc88a63b89c9bacf79edd8872fd4ded5fdeab770cf79c2320504515ae13c4533'),
}
TEACHER_SHA256 = '123c8e9289f702deff0b842933bc6af3cfddd282749b6eb954dc55fd91d52389'  # export/teacher.pt, job E
SEGNEURON_CKPT_SHA256 = 'e726bb43201aefbcf1e19b6feb8be555656f635241c07e3d7803329dbf55f228'
CLASSES_SHA256 = '7e5d3c5e310bd0fdaf042075c7b3fe3d75393e3fc749fa046dc0ae241fe4313b'  # liconn_v2/proofread_classes.json
SEGNEURON_THRESHOLD = 0.6  # chosen on crop 12 (docs/segneuron_finetune_results.md)
V100_FFN_VARIANT = 'fp16_head32'  # run V: the crop-12 head-to-head's FFN

# Question 1: the owner's 2% rule (docs/tolerance2_results.md) over every object.
FLIP = manyseed.FLIP  # a per-object fall of more than 10 points
MAX_FLIP_FRACTION = 0.02  # new merges, and new early stops, each at most 2% of the objects
MAX_MEAN_DROP = 0.01  # mean recall and mean precision each fall by at most 1 point
EPS = manyseed.EPS

# Questions 2 and 3.
CONFIDENCE = 0.95
MARGIN = 0.02  # F1; non-inferiority of the smaller FFNs against FFN
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261104
RUNAWAY = 3  # descriptive: a trace of at least 3x the reference's steps on the same object


# ---------------------------------------------------------------- crops

def check_crop(k: int) -> None:
  if k not in CROPS:
    raise ValueError(f'crop {k} is not one of the unseen crops {CROPS[0]}-{CROPS[-1]}')


def ordered_crops(labeled_coarse_xyz: np.ndarray) -> list:
  """The unseen crops from the fixed order; they must be exactly the design's positions 41-50 and
  the order's first 41 must be the crops used before."""
  order = manyseed.select_crops(labeled_coarse_xyz, None)
  used = [c['xyz'] for c in manyseed.load_manifest()['crops']] + \
         [c['xyz'] for c in json.loads((DOCS / 'fresh_crops.json').read_text())['crops']]
  if [c['xyz'] for c in order[:USED_BEFORE]] != used:
    raise ValueError('the crop order no longer matches crops 0-40')
  rest = order[USED_BEFORE:]
  if [c['index'] for c in rest] != list(CROPS) or [tuple(c['xyz']) for c in rest] != [CROP_XYZ[k] for k in CROPS]:
    raise ValueError(f'positions after crop 40 are {[(c["index"], c["xyz"]) for c in rest]}, not the design\'s')
  return rest


def load_manifest() -> dict:
  return json.loads(MANIFEST.read_text())


def _entry(k: int, manifest: dict | None = None) -> dict:
  check_crop(k)
  manifest = load_manifest() if manifest is None else manifest
  for c in manifest['crops']:
    if c['index'] == k:
      return c
  raise KeyError(f'crop {k} is not pinned in {MANIFEST}')


def verify_crop(k: int, manifest: dict | None = None) -> None:
  for rel, expected in _entry(k, manifest)['sha256'].items():
    path = manyseed.crop_dir(k) / rel
    if not path.is_file():
      raise FileNotFoundError(f'{path} is missing')
    actual = data.sha256_file(path)
    if actual != expected:
      raise ValueError(f'{path} has sha256 {actual}, expected {expected}')


def load_raw_xyz(k: int) -> np.ndarray:
  verify_crop(k)
  return np.load(manyseed.crop_dir(k) / manyseed.RAW_FILE)


def load_proofread_zyx(k: int) -> np.ndarray:
  verify_crop(k)
  return np.ascontiguousarray(np.transpose(np.load(manyseed.crop_dir(k) / manyseed.PROOFREAD_FILE), [2, 1, 0]))


def design_sha256() -> str:
  return data.sha256_file(DESIGN)


def fetch() -> dict:
  """Fetches crops 41-50 and pins their files. Writes files only; computes nothing from them."""
  coarse = manyseed.cache_dir() / manyseed.COARSE_FILE
  if data.sha256_file(coarse) != manyseed.load_manifest()['coarse_sha256']:
    raise ValueError(f'{coarse} differs from its pinned sha256')
  crops = ordered_crops(np.load(coarse))
  pinned = load_manifest() if MANIFEST.is_file() else None
  if pinned is not None and pinned['design_sha256'] != design_sha256():
    raise ValueError(f'the design changed since the crops were pinned in {MANIFEST}')
  image = manyseed._open(data.IMAGE_PATH, data.IMAGE_SCALE_INDEX)  # pylint: disable=protected-access
  seg = manyseed._open(data.PROOFREAD_PATH, 0)  # pylint: disable=protected-access
  for c in crops:
    manyseed.fetch_crop(c, image, seg)  # skips files already cached
    if pinned is not None and c != _entry(c['index'], pinned):
      raise ValueError(f"crop {c['index']} differs from its pinned entry in {MANIFEST}")
  if pinned is not None:  # the pins are never rewritten
    return pinned
  out = {'design': 'docs/unseen_design.md', 'design_sha256': design_sha256(), 'crops': crops}
  MANIFEST.write_text(json.dumps(out, indent=1) + '\n')
  return out


# ---------------------------------------------------------------- analysis (numpy only)

def _paired(ref: list, cand: list, what: str) -> None:
  if [r['object_id'] for r in ref] != [r['object_id'] for r in cand]:
    raise ValueError(f'{what}: the two runs scored different objects')


def class_of(classes: dict) -> dict:
  """object ID -> class name, from benchmark.load_classes() (class -> IDs)."""
  return {int(i): name for name, ids in classes.items() for i in ids}


def _flips(ref: list, cand: list) -> dict:
  p = np.array([c['precision'] - r['precision'] for r, c in zip(ref, cand)])
  r_ = np.array([c['recall'] - r['recall'] for r, c in zip(ref, cand)])
  return {'objects': len(p), 'new_merges': int((p < -FLIP - EPS).sum()), 'new_early_stops': int((r_ < -FLIP - EPS).sum()),
          'avoided_merges': int((p > FLIP + EPS).sum()), 'extensions': int((r_ > FLIP + EPS).sum()),
          'mean_precision_change': float(p.mean()) if len(p) else None,
          'mean_recall_change': float(r_.mean()) if len(p) else None}


def fast_mode_rule(ref_by_crop: dict, cand_by_crop: dict, classes: dict | None = None) -> dict:
  """Question 1: F5 against F1 on every object of every crop, pooled. classes (object ID -> class)
  only split the counts; they do not enter the rule."""
  if sorted(ref_by_crop) != list(CROPS) or sorted(cand_by_crop) != list(CROPS):
    raise ValueError('question 1 needs all ten crops')
  for k in CROPS:
    _paired(ref_by_crop[k], cand_by_crop[k], f'crop {k}')
  ref = [r for k in CROPS for r in ref_by_crop[k]]
  cand = [r for k in CROPS for r in cand_by_crop[k]]
  counts = _flips(ref, cand)
  limit = MAX_FLIP_FRACTION * counts['objects']
  rules = {'new_merges': counts['new_merges'] <= limit + EPS,
           'new_early_stops': counts['new_early_stops'] <= limit + EPS,
           'mean_recall_drop': counts['mean_recall_change'] >= -MAX_MEAN_DROP - EPS,
           'mean_precision_drop': counts['mean_precision_change'] >= -MAX_MEAN_DROP - EPS}
  by_class = {}
  if classes is not None:
    names = [classes.get(r['object_id'], 'unclassified') for r in ref]
    for name in sorted(set(names)):
      idx = [i for i, x in enumerate(names) if x == name]
      by_class[name] = _flips([ref[i] for i in idx], [cand[i] for i in idx])
  already = [i for i, r in enumerate(ref) if r['precision'] < manyseed.ALREADY_MERGED_PRECISION]
  return {'rule': '2% of objects: new merges and new early stops (a fall of more than 10 points) each at most '
                  '2% of the objects; mean recall and mean precision each fall by at most 1 point',
          'limit_objects': limit, 'counts': counts, 'rules': rules, 'passes': all(rules.values()),
          'per_crop': {k: _flips(ref_by_crop[k], cand_by_crop[k]) for k in CROPS}, 'by_class': by_class,
          'reference_already_merged': {'objects': len(already),
                                       'new_merges_among_them': _flips([ref[i] for i in already],
                                                                       [cand[i] for i in already])['new_merges']}}


def crop_mean_f1(records: list) -> float:
  return float(np.mean([r['f1'] for r in records]))


def t_quantile(df: int, confidence: float = CONFIDENCE) -> float:
  from scipy import stats  # pylint: disable=g-import-not-at-top
  return float(stats.t.ppf(0.5 + confidence / 2, df))


def paired_crops(a: dict, b: dict) -> dict:
  """a - b over crops, from per-crop mean F1 (crop -> value): mean, t-interval (decides), percentile
  bootstrap interval (a check)."""
  if sorted(a) != list(CROPS) or sorted(b) != list(CROPS):
    raise ValueError('a comparison needs all ten crops for both methods')
  d = np.array([a[k] - b[k] for k in CROPS])
  n = len(d)
  half = t_quantile(n - 1) * d.std(ddof=1) / np.sqrt(n)
  rng = np.random.default_rng(BOOTSTRAP_SEED)
  boot = d[rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))].mean(axis=1)
  lo_q, hi_q = (1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2
  return {'crops': n, 'per_crop': {k: float(v) for k, v in zip(CROPS, d)}, 'mean': float(d.mean()),
          't_interval': [float(d.mean() - half), float(d.mean() + half)],
          'bootstrap_interval': [float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q))],
          'crops_a_higher': int((d > 0).sum()), 'crops_b_higher': int((d < 0).sum())}


def margin_verdict(diff: dict) -> dict:
  """Question 3 for (smaller FFN minus FFN), from the t-interval [L, U] and the 0.02 margin."""
  lo, hi = diff['t_interval']
  if lo > -MARGIN:
    label = 'not worse than FFN by more than 0.02' if hi >= 0 else 'worse than FFN, by less than 0.02'
  else:
    label = 'worse than FFN' if hi < 0 else 'inconclusive'
  return {'verdict': label, 'non_inferior': lo > -MARGIN, 'worse_than_ffn': hi < 0, 'better_than_ffn': lo > 0}


def difference_claimed(diff: dict) -> bool:
  lo, hi = diff['t_interval']
  return lo > 0 or hi < 0


def object_counts(records: list) -> dict:
  merges = [r['merges'] if isinstance(r['merges'], int) else len(r['merges']) for r in records]
  return {'objects': len(records), 'merges': int(sum(merges)), 'objects_with_merge': int(sum(m > 0 for m in merges)),
          'merged_voxels': int(sum(r['merged_voxels'] for r in records))}


def runaways(records: list, reference: list) -> dict:
  """Traces with at least 3x the reference's steps on the same object (descriptive)."""
  _paired(reference, records, 'runaways')
  hit = [r['object_id'] for r, f in zip(records, reference) if r['steps'] >= RUNAWAY * f['steps']]
  return {'runaway_traces': len(hit), 'object_ids': hit,
          'reference_stopped_at_first_step': [f['object_id'] for f in reference if f['steps'] == 1]}


def _merge_lists(records: list) -> list:
  """Records whose merge count is an int (examples/time_crop11.py) with merges as a list of that
  length, as headtohead.paired expects."""
  return [dict(r, merges=[None] * r['merges']) if isinstance(r['merges'], int) else r for r in records]


def pooled_means(records: list) -> dict:
  return {k: float(np.mean([r[k] for r in records])) for k in ('recall', 'precision', 'f1')}


def repeated_objects(runs_by_crop: dict) -> int:
  """Object IDs scored in more than one crop (a neuron can be large in two crops)."""
  seen = {}
  for k in CROPS:
    for r in runs_by_crop[k]:
      seen[r['object_id']] = seen.get(r['object_id'], 0) + 1
  return int(sum(c > 1 for c in seen.values()))


METHODS = B200_RUNS + ('V', 'segneuron')


def analysis(results: dict, failures: dict | None = None, classes: dict | None = None) -> dict:
  """The pre-stated analysis. results: method -> crop -> records, for methods in METHODS (B200 runs
  from examples/run_unseen.py; 'V' and 'segneuron' from examples/run_unseen_wahab.py), every list
  in the same object order. failures: method -> {crop: reason} for runs that stopped or produced
  non-finite outputs; such a run is a result for that method (the design), and every comparison
  that needs it is reported as not judged, with the reasons. classes: object ID -> class, for the
  class split of question 1 only."""
  failures = {m: dict(v) for m, v in (failures or {}).items() if v}
  ok = [m for m in METHODS if m not in failures]
  for m in ok:
    if sorted(results[m]) != list(CROPS):
      raise ValueError(f'{m} has crops {sorted(results[m])}; a run without a result must be listed as a failure')
  for k in CROPS:
    for m in ok:
      _paired(results[ok[0]][k], results[m][k], f'crop {k} {m}')
  pool = {m: [r for k in CROPS for r in results[m][k]] for m in ok}
  means = {m: {k: crop_mean_f1(results[m][k]) for k in CROPS} for m in ok}

  def not_judged(*methods):
    return {'not_judged': {m: failures[m] for m in methods if m in failures}}

  if REFERENCE in failures:
    q1 = {'passes': None, **not_judged(REFERENCE, FFN_FAST)}
  elif FFN_FAST in failures:
    q1 = {'passes': False, 'rules': {'complete': False}, **not_judged(FFN_FAST)}
  else:
    q1 = fast_mode_rule(results[REFERENCE], results[FFN_FAST], classes)
    q1['rules']['complete'] = True

  q2 = {}
  for ffn in (FFN_FAST, 'V'):
    if ffn in failures or 'segneuron' in failures:
      q2[f'{ffn}_minus_segneuron'] = not_judged(ffn, 'segneuron')
      continue
    d = paired_crops(means[ffn], means['segneuron'])
    q2[f'{ffn}_minus_segneuron'] = {**d, 'difference_claimed': difference_claimed(d),
                                    'objects': hh.paired(_merge_lists(pool[ffn]), pool['segneuron'])}
  if 'V' in pool and FFN_FAST in pool:
    f1_diff = np.array([v['f1'] - f['f1'] for v, f in zip(pool['V'], pool[FFN_FAST])])
    q2['V_vs_F5_agreement'] = {'objects': len(f1_diff), 'mean_f1_difference': float(f1_diff.mean()),
                               'max_abs_f1_difference': float(np.abs(f1_diff).max()),
                               'objects_abs_difference_gt_0.1': int((np.abs(f1_diff) > 0.1).sum())}
  if all('difference_claimed' in q2[f'{f}_minus_segneuron'] for f in (FFN_FAST, 'V')):
    q2['claims_agree'] = q2[f'{FFN_FAST}_minus_segneuron']['difference_claimed'] == \
        q2['V_minus_segneuron']['difference_claimed']

  q3 = {}
  for name in SMALLER:
    if name in failures or FFN_FAST in failures:
      q3[f'{SMALLER[name]}_minus_ffn'] = not_judged(name, FFN_FAST)
    else:
      d = paired_crops(means[name], means[FFN_FAST])
      q3[f'{SMALLER[name]}_minus_ffn'] = {**d, **margin_verdict(d)}
  if 'P2' in failures or 'S10' in failures:
    p8_student = not_judged('P2', 'S10')
  else:
    d = paired_crops(means['P2'], means['S10'])
    p8_student = {**d, 'difference_claimed': difference_claimed(d)}

  pairs = [(name, REFERENCE) for name in (*SMALLER, FFN_FAST)] + [(name, FFN_FAST) for name in SMALLER]
  run_away = {f'{name}_vs_{ref}': runaways(pool[name], pool[ref]) for name, ref in pairs
              if name in pool and ref in pool}
  any_ok = results[ok[0]] if ok else {}
  return {'design': 'docs/unseen_design.md', 'crops': list(CROPS), 'failures': failures,
          'objects_per_crop': {k: len(any_ok[k]) for k in CROPS} if ok else None,
          'object_ids_in_more_than_one_crop': repeated_objects(any_ok) if ok else None,
          'q1_fast_mode': q1, 'q2_ffn_vs_segneuron': q2, 'q3_smaller_ffns': q3, 'q3_p8_minus_student': p8_student,
          'crop_mean_f1': means, 'pooled': {m: pooled_means(v) for m, v in pool.items()},
          'counts': {m: object_counts(v) for m, v in pool.items()}, 'runaways': run_away}

if __name__ == '__main__':
  print(json.dumps({'crops': [c['index'] for c in fetch()['crops']], 'manifest': str(MANIFEST)}))
