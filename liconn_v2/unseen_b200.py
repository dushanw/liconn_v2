"""SegNeuron on the B200 against the unseen-crop test's B200 runs (docs/unseen_b200_segneuron_design.md).

Numpy only: reads the B200 SegNeuron result files (crop-K/segneuron.json from examples/run_unseen_wahab.py,
run by cluster/mit/unseen_segneuron.sbatch), the test's V100 SegNeuron result files and the test's
judged B200 crop times (docs/unseen_evidence.json), and computes the addendum's measures.
"""

from pathlib import Path

import math

import numpy as np

from liconn_v2 import data
from liconn_v2 import unseen

DESIGN = Path(__file__).resolve().parent.parent / 'docs' / 'unseen_b200_segneuron_design.md'
PRIMARY = 'F5'  # batched fast FFN
RUNS = unseen.B200_RUNS  # F1, F5, S10, P2, Q2: the test's B200 runs
NAMES = {'F1': 'float32 FFN one object at a time', 'F5': 'batched fast FFN', 'S10': 'the student', 'P2': 'P8',
         'Q2': 'P4'}
WAYS = ('end_to_end', 'without_startup')  # end to end as on the V100 (primary); loop plus post-processing
INFER_PINS = ('torch==2.7.1+cu128', 'numpy==2.5.4', 'tifffile==2026.10.15')  # the design's inference versions
AGREEMENT_F1 = 0.1  # objects whose F1 differs by more than this between the B200 and V100 runs are counted


def design_sha256() -> str:
  return data.sha256_file(DESIGN)


def check_staging(sources: dict, envs: dict, pinned: dict) -> None:
  """Every judged crop ran from the same staged source (sources {k: source_sha256.txt text}), holding
  the test's code unchanged (pinned {path: sha256}, docs/unseen_evidence.json source_sha256), with the
  design's inference versions (envs {k: env-infer.txt text}), or ValueError."""
  if len(set(sources.values())) > 1:
    raise ValueError(f'crops ran from different staged sources: {sorted(sources)}')
  for k, text in sources.items():
    staged = {path: sha for sha, path in (line.split(None, 1) for line in text.splitlines() if line.strip())}
    changed = sorted(path for path, sha in pinned.items() if staged.get(path) != sha)
    if changed:
      raise ValueError(f"B200 SegNeuron crop {k}: staged source differs from the test's code in {changed}")
  for k, env in envs.items():
    missing = [p for p in INFER_PINS if p not in env.split()]
    if missing:
      raise ValueError(f'B200 SegNeuron crop {k}: inference environment lacks {missing}')


def check_device(gpu: str, k: int) -> None:
  """The job's GPU (the name nvidia-smi reported) is a B200, or ValueError."""
  if not gpu.startswith('NVIDIA B200'):
    raise ValueError(f'B200 SegNeuron crop {k} ran on {gpu!r}')


def check_file(x: dict, k: int, v100: dict, objects: int) -> None:
  """A B200 SegNeuron file of crop k is the test's method on the test's objects (all objects of the
  crop, as many as the test's FFN runs timed), or ValueError."""
  where = f'B200 SegNeuron crop {k}'
  if not len(x['object_ids']) == x['object_count'] == len(v100['object_ids']) == objects:
    raise ValueError(f"{where}: {len(x['object_ids'])} objects, the test has {objects}")
  if x['crop'] != k or x['smoke_objects'] is not None:
    raise ValueError(f"{where}: file is for crop {x['crop']}, smoke {x['smoke_objects']}")
  if x['design_sha256'] != unseen.design_sha256():
    raise ValueError(f'{where}: written under another version of docs/unseen_design.md')
  if x['checkpoint_sha256'] != unseen.SEGNEURON_CKPT_SHA256 or x['threshold'] != unseen.SEGNEURON_THRESHOLD:
    raise ValueError(f"{where}: checkpoint {x['checkpoint_sha256']} or threshold {x['threshold']} is not the test's")
  if x['crop_sha256'] != v100['crop_sha256'] or x['object_ids'] != v100['object_ids']:
    raise ValueError(f'{where}: crop file or objects differ from the V100 run')
  if [o['object_id'] for o in x['objects']] != x['object_ids']:
    raise ValueError(f'{where}: records are not every object of the header, in order')


def valid_seconds(v) -> bool:
  return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0


def times(x: dict) -> dict:
  """SegNeuron's crop time both ways: end to end as headtohead.speed (inference.py's process wall time
  plus post-processing) and without start-up (inference.py's own loop time plus post-processing)."""
  loop = x['inference']['inference_seconds']
  for v in (x['inference_wall_seconds'], loop, x['post_seconds']):
    if not valid_seconds(v):
      raise ValueError(f'invalid SegNeuron time {v!r}')
  return {'end_to_end': x['inference_wall_seconds'] + x['post_seconds'],
          'without_startup': loop + x['post_seconds'],
          'inference_wall_seconds': x['inference_wall_seconds'], 'inference_seconds': loop,
          'post_seconds': x['post_seconds']}


def statement(cheaper: int, dearer: int, judged: int, name: str) -> str:
  """The design's wording, from the number of crops on which SegNeuron was cheaper and dearer (the
  rest of the judged crops are ties)."""
  n = len(unseen.CROPS)
  ties = judged - cheaper - dearer
  tied = f', equal on {ties}' if ties else ''
  if judged == 0:
    return 'not judged'
  if judged < n:
    return f'SegNeuron was cheaper on {cheaper} of {judged} judged crops{tied} ({n - judged} not judged)'
  if cheaper == n:
    return f'SegNeuron was cheaper than {name} on every test crop'
  if dearer == n:
    return f'{name} was cheaper than SegNeuron on every test crop'
  return f'SegNeuron was cheaper on {cheaper} of {n} crops{tied}'


def compare(sn: dict, run: dict, objects: dict, name: str) -> dict:
  """SegNeuron's crop times sn {k: seconds, or None if not timed} against a B200 run's crop times run
  {k: seconds} on crops with objects {k: n}: crops cheaper, break-even objects per crop and from totals."""
  for k in unseen.CROPS:
    if not (valid_seconds(run[k]) and run[k] > 0 and objects[k] > 0):
      raise ValueError(f'crop {k}: {name} took {run[k]!r} s for {objects[k]} objects')
  judged = [k for k in unseen.CROPS if sn[k] is not None]
  cheaper = [k for k in judged if sn[k] < run[k]]
  dearer = [k for k in judged if sn[k] > run[k]]
  per_crop = {k: sn[k] / (run[k] / objects[k]) for k in judged}
  out = {'crops_judged': len(judged), 'crops_segneuron_cheaper': len(cheaper), 'cheaper_on': cheaper,
         'crops_segneuron_dearer': len(dearer),
         'segneuron_over_run': {k: sn[k] / run[k] for k in judged},
         'break_even_per_crop': per_crop,
         'statement': statement(len(cheaper), len(dearer), len(judged), name)}
  if judged:
    v = list(per_crop.values())
    out['break_even'] = {'median': float(np.median(v)), 'min': float(min(v)), 'max': float(max(v))}
  if len(judged) == len(unseen.CROPS):
    n = sum(objects[k] for k in unseen.CROPS)
    out['totals'] = {'segneuron_seconds': float(sum(sn.values())), 'run_seconds': float(sum(run.values())),
                     'break_even_per_crop': (sum(sn.values()) / len(unseen.CROPS)) / (sum(run.values()) / n)}
  return out


def agreement(b200: list, v100: list) -> dict:
  """Per-object F1 of the B200 run minus the V100 run's, for the same objects in the same order."""
  if [o['object_id'] for o in b200] != [o['object_id'] for o in v100]:
    raise ValueError('the B200 and V100 runs scored different objects')
  d = np.array([a['f1'] - b['f1'] for a, b in zip(b200, v100)])
  return {'objects': len(d), 'mean_f1_difference': float(d.mean()), 'max_abs_f1_difference': float(abs(d).max()),
          f'objects_abs_difference_gt_{AGREEMENT_F1}': int((abs(d) > AGREEMENT_F1).sum()),
          'identical_f1': int((d == 0).sum()),
          'mean_f1_b200': float(np.mean([o['f1'] for o in b200])),
          'mean_f1_v100': float(np.mean([o['f1'] for o in v100]))}


def judge(b200: dict, v100: dict, evidence: dict) -> dict:
  """b200 {k: file or None (missing)}, v100 {k: the test's V100 SegNeuron file}, evidence: the test's
  docs/unseen_evidence.json. Every measure of the design, with failures as 'not judged'."""
  j = evidence['judged']
  objects = {k: j['objects_per_crop'][str(k)] for k in unseen.CROPS}
  failures, sn = {}, {}
  for k in unseen.CROPS:
    r = v100[k]
    if r['crop'] != k or r['smoke_objects'] is not None or r['design_sha256'] != unseen.design_sha256() or \
        r['checkpoint_sha256'] != unseen.SEGNEURON_CKPT_SHA256 or r['threshold'] != unseen.SEGNEURON_THRESHOLD or \
        [o['object_id'] for o in r['objects']] != r['object_ids']:
      raise ValueError(f"V100 SegNeuron crop {k}: not the test's run")
    x = b200[k]
    if x is None:
      failures[k] = 'no result file'
      continue
    check_file(x, k, v100[k], objects[k])
    if x['nonfinite_outputs']:
      failures[k] = f"{x['nonfinite_outputs']} non-finite SegNeuron outputs"
      continue
    try:
      sn[k] = times(x)
    except ValueError as e:
      failures[k] = str(e)
  out = {'failures': failures, 'objects_per_crop': objects,
         'segneuron_b200': sn, 'segneuron_v100': {k: times(v100[k]) for k in unseen.CROPS}, 'comparisons': {}}
  for way in WAYS:
    t = {k: sn[k][way] if k in sn else None for k in unseen.CROPS}
    out['comparisons'][way] = {
        r: compare(t, {k: j['speed_b200'][r]['crop_seconds'][str(k)] for k in unseen.CROPS}, objects, NAMES[r])
        for r in RUNS}
  out['agreement'] = {k: agreement(b200[k]['objects'], v100[k]['objects']) for k in sn}
  if sn:
    a = [b200[k]['objects'][i]['f1'] - v100[k]['objects'][i]['f1'] for k in sn for i in range(len(v100[k]['objects']))]
    out['agreement_pooled'] = {'objects': len(a), 'mean_f1_difference': float(np.mean(a)),
                               'max_abs_f1_difference': float(np.max(np.abs(a))),
                               f'objects_abs_difference_gt_{AGREEMENT_F1}': int((np.abs(a) > AGREEMENT_F1).sum())}
  return out
