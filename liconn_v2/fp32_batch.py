"""Float32 FFN traced many objects at once: crop-11 widths and the test crops (docs/fp32_batch_design.md).

Settings, run fingerprints and the reported comparisons (numpy only). The GPU runs are
examples/time_widths.py (crop 11) and examples/run_unseen_fp32x32.py (crops 41-50); the report is
examples/judge_fp32_batch.py.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from liconn_v2 import data
from liconn_v2 import unseen
from liconn_v2 import unseen_b200

DESIGN = Path(__file__).resolve().parent.parent / 'docs' / 'fp32_batch_design.md'

# run: (number format, objects at once, residual stream scale), in the order of the crop-11 job
RUNS = {
    'F32x1': ('fp32', 1, 1.0),
    'F32x8': ('fp32', 8, 1.0),
    'F32x32': ('fp32', 32, 1.0),
    'F32x64': ('fp32', 64, 1.0),
    'H16x1': ('fp16_head32', 1, 1 / 16),
    'H16x8': ('fp16_head32', 8, 1 / 16),
    'H16x32': ('fp16_head32', 32, 1 / 16),
    'H16x64': ('fp16_head32', 64, 1 / 16),
}
WIDTHS = (1, 8, 32, 64)
FORMATS = {'F32': 'fp32', 'H16': 'fp16_head32'}
RERUN = ('H16x1', 'H16x8', 'H16x32', 'H16x64')  # rerun each in its own process after the first job ran out of memory
TEST_RUN = 'F32x32'  # width fixed in the design: that of every other tracer on the test crops
# The timing study's runs with the same settings (docs/distil_timing_evidence.json, run_fingerprints)
EARLIER = {'F32x1': ('F1', '4e60f463b90c6bb8bd97596886b8136b565ae9a2806c0c24625ea5e0b718e3e5'),
           'H16x32': ('F5', 'cb4e707207002ac68cde63e5cd1b6074e3a587326f69a7db0a631330309e9d07')}


def design_sha256() -> str:
  return data.sha256_file(DESIGN)


def check_run(name: str) -> tuple:
  if name not in RUNS:
    raise ValueError(f'run {name} is not one of {list(RUNS)}')
  return RUNS[name]


def fingerprint(records: list) -> str:
  """As docs/distil_timing_evidence.json: sha256 of [[object_id, steps, mask_sha256, history_sha256], ...]."""
  rows = [[r['object_id'], r['steps'], r['mask_sha256'], r['history_sha256']] for r in records]
  return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def identity(a: list, b: list) -> dict:
  """Objects with the same steps, mask and history in two runs over the same objects."""
  if [r['object_id'] for r in a] != [r['object_id'] for r in b]:
    raise ValueError('the two runs traced different objects')
  same = [x['steps'] == y['steps'] and x['mask_sha256'] == y['mask_sha256'] and x['history_sha256'] == y['history_sha256']
          for x, y in zip(a, b)]
  return {'objects': len(same), 'identical': int(sum(same)),
          'same_mask': int(sum(x['mask_sha256'] == y['mask_sha256'] for x, y in zip(a, b))),
          'same_steps': int(sum(x['steps'] == y['steps'] for x, y in zip(a, b)))}


def widths_report(runs: dict) -> dict:
  """runs: name -> {'summary': {...}, 'records': [...]} for the crop-11 runs present (a job stopped by
  its time limit keeps the runs it finished; the others are listed as missing). Per-run summaries,
  identity of each batched run against its format's width-1 run, fingerprints against the timing
  study, and the speed-up split by crop time and by time per step; no pass mark."""
  sec = {n: r['summary']['crop_seconds'] for n, r in runs.items()}
  ms = {n: r['summary']['ms_per_step'] for n, r in runs.items()}
  keys = ('crop_seconds', 'ms_per_step', 'steps', 'mean_f1', 'mean_recall', 'mean_precision', 'merges_ge_1000',
          'merged_voxels', 'nonfinite_outputs')
  out = {'missing_runs': [n for n in RUNS if n not in runs],
         'runs': {n: {key: r['summary'].get(key) for key in keys} for n, r in runs.items()},
         'identity_vs_width_1': {}, 'fingerprints': {}, 'same_as_timing_study': {},
         'batching_speedup': {}, 'batching_speedup_per_step': {}, 'precision_speedup_at_width': {},
         'precision_speedup_per_step_at_width': {}}
  for n, r in runs.items():
    out['fingerprints'][n] = fingerprint(r['records'])
    if n in EARLIER:
      out['same_as_timing_study'][n] = {'run': EARLIER[n][0], 'same': out['fingerprints'][n] == EARLIER[n][1]}
  for p in FORMATS:
    one = f'{p}x1'
    for w in WIDTHS[1:]:
      n = f'{p}x{w}'
      if one in runs and n in runs:
        out['identity_vs_width_1'][n] = identity(runs[one]['records'], runs[n]['records'])
        out['batching_speedup'][n] = sec[one] / sec[n]
        out['batching_speedup_per_step'][n] = ms[one] / ms[n]
  for w in WIDTHS:
    a, b = f'F32x{w}', f'H16x{w}'
    if a in runs and b in runs:
      out['precision_speedup_at_width'][str(w)] = sec[a] / sec[b]
      out['precision_speedup_per_step_at_width'][str(w)] = ms[a] / ms[b]
  if 'F32x1' in runs and 'H16x32' in runs:
    out['fast_ffn_over_fp32_one_at_a_time'] = sec['F32x1'] / sec['H16x32']
    out['fast_ffn_over_fp32_one_at_a_time_per_step'] = ms['F32x1'] / ms['H16x32']
  fp32 = {n: t for n, t in sec.items() if RUNS[n][0] == 'fp32'}
  if fp32:
    out['fastest_fp32_width'] = RUNS[min(fp32, key=fp32.get)][1]
  return out


def check_widths_file(w: dict) -> None:
  """Part 1's widths.json: crop 11, all 512 objects, the test's teacher, a B200, TF32 off, this
  design, and each run with its settings in RUNS."""
  if w['crop'] != 11 or w['max_objects'] is not None or w['objects'] != 512:
    raise ValueError(f"widths file: crop {w['crop']}, {w['objects']} objects (max {w['max_objects']})")
  if w['teacher_sha256'] != unseen.TEACHER_SHA256 or w['tf32'] is not False or not w['device'].startswith('NVIDIA B200'):
    raise ValueError('widths file: another teacher, TF32 on, or not a B200')
  if w['design_sha256'] != design_sha256():
    raise ValueError(f'widths file: run under another version of {DESIGN.name}')
  ids = None
  for n, r in w['runs'].items():
    s = r['summary']
    if n not in RUNS or (s['format'], s['width'], s['stream_scale']) != RUNS[n] or s['objects'] != 512:
      raise ValueError(f'widths file: run {n} does not have the design\'s settings')
    run_ids = [x['object_id'] for x in r['records']]
    if len(run_ids) != 512 or (ids is not None and run_ids != ids):
      raise ValueError(f'widths file: run {n} has {len(run_ids)} records or other objects than the first run')
    ids = run_ids


def merge_widths(files: list) -> dict:
  """files: [(label, widths file), ...] in order, each checked with check_widths_file. A later file
  replaces a run of an earlier one (a setting rerun in its own process). With more than one file,
  no run in RERUN may come from the first file. Returns the runs in RUNS order, the file each came
  from, and for each replaced run its file, summary and identity against the run replacing it."""
  if len({label for label, _ in files}) != len(files):
    raise ValueError('two widths files have the same label')
  runs, source, replaced, ids = {}, {}, {}, None
  for label, w in files:
    check_widths_file(w)
    for n, r in w['runs'].items():
      run_ids = [x['object_id'] for x in r['records']]
      if ids is not None and run_ids != ids:
        raise ValueError(f'{label}: run {n} traced other objects than the earlier files')
      ids = run_ids
      if n in runs:
        replaced.setdefault(n, []).append({'file': source[n], 'summary': runs[n]['summary'],
                                           'identity_vs_replacing_run': identity(runs[n]['records'], r['records'])})
      runs[n], source[n] = r, label
  if len(files) > 1 and any(source.get(n) == files[0][0] for n in RERUN):
    raise ValueError(f'runs {[n for n in RERUN if source.get(n) == files[0][0]]} still come from the first file')
  ordered = {n: runs[n] for n in RUNS if n in runs}
  return {'runs': ordered, 'source': {n: source[n] for n in ordered}, 'replaced': replaced}


def cross_job(source: dict) -> dict:
  """For a merged report: the speed-ups that divide times of runs from different files. Each file is
  written by its own process; the files do not record their Slurm job or node, so runs from different
  files may come from different jobs and B200 nodes."""
  pairs = {}
  for p in FORMATS:
    for w in WIDTHS[1:]:
      pairs[f'batching_speedup/{p}x{w}'] = (f'{p}x1', f'{p}x{w}')
  for w in WIDTHS:
    pairs[f'precision_speedup_at_width/{w}'] = (f'F32x{w}', f'H16x{w}')
  pairs['fast_ffn_over_fp32_one_at_a_time'] = ('F32x1', 'H16x32')
  across = sorted(key for key, (a, b) in pairs.items() if a in source and b in source and source[a] != source[b])
  files = sorted(set(source.values()))
  return {'files': len(files), 'comparisons_across_files': across,
          'statement': (f'The runs come from {len(files)} files, each written by its own process; files may come '
                        'from different Slurm jobs and B200 nodes (they do not record which), so the comparisons '
                        'listed divide times measured in different processes and possibly on different '
                        'nodes.') if len(files) > 1 else 'All runs come from one process.'}


def check_test_file(x: dict, k: int, f1: dict) -> None:
  """A test-crop F32x32 result: the design's settings on crop k, on a B200, the same crop files and
  objects as the test's F1 result f1, the test's teacher."""
  s = x['run']['summary']
  if x['crop'] != k or x['smoke_objects'] is not None:
    raise ValueError(f'crop {k}: the file is for crop {x["crop"]} (smoke {x["smoke_objects"]})')
  if x['run_name'] != TEST_RUN or (s['format'], s['width'], s['stream_scale']) != RUNS[TEST_RUN] \
      or s['network'] != 'ffn' or s['box_only'] is not True:
    raise ValueError(f'crop {k}: not run {TEST_RUN} with the design\'s settings')
  if x['teacher_sha256'] != unseen.TEACHER_SHA256 or x['tf32'] is not False or not x['device'].startswith('NVIDIA B200'):
    raise ValueError(f'crop {k}: another teacher, TF32 on, or not a B200')
  if x['design_sha256'] != design_sha256():
    raise ValueError(f'crop {k}: run under another version of {DESIGN.name}')
  if x['crop_sha256'] != f1['crop_sha256']:
    raise ValueError(f'crop {k}: crop files differ from the test\'s')
  ids = [r['object_id'] for r in x['run']['records']]
  if ids != x['object_ids'] or ids != [r['object_id'] for r in f1['run']['records']] or s['objects'] != len(ids):
    raise ValueError(f'crop {k}: objects differ from the test\'s F1 run')


def test_report(f32x32: dict, f1: dict, f5: dict, segneuron: dict) -> dict:
  """f32x32: crop -> result file for the crops that finished; f1, f5: the test's files for every crop
  (examples/run_unseen.py layout); segneuron: the 'judged' section of docs/unseen_b200_evidence.json.
  Per crop: time, time per step, identity against F1 and the speed-up split; over all ten crops,
  only when every crop finished: totals, accuracy, changes against F1 and SegNeuron's break-even
  against F32x32. No pass mark."""
  done = [k for k in unseen.CROPS if k in f32x32]
  for k in done:
    check_test_file(f32x32[k], k, f1[k])
  files = {'F32x32': f32x32, 'F1': f1, 'F5': f5}
  sec = {r: {k: x[k]['run']['summary']['crop_seconds'] for k in done} for r, x in files.items()}
  ms = {r: {k: x[k]['run']['summary']['ms_per_step'] for k in done} for r, x in files.items()}
  ident = {k: identity(f1[k]['run']['records'], f32x32[k]['run']['records']) for k in done}
  out = {'missing_crops': [k for k in unseen.CROPS if k not in f32x32], 'crop_seconds': sec, 'ms_per_step': ms,
         'steps': {r: {k: x[k]['run']['summary']['steps'] for k in done} for r, x in files.items()},
         'per_crop': {k: {'batching_speedup_fp32': sec['F1'][k] / sec['F32x32'][k],
                          'precision_speedup_at_32': sec['F32x32'][k] / sec['F5'][k],
                          'batching_speedup_fp32_per_step': ms['F1'][k] / ms['F32x32'][k],
                          'precision_speedup_at_32_per_step': ms['F32x32'][k] / ms['F5'][k],
                          'mean_f1': float(np.mean([r['f1'] for r in f32x32[k]['run']['records']]))} for k in done},
         'identity_vs_F1': ident,
         'identical_objects': int(sum(v['identical'] for v in ident.values())),
         'same_mask_objects': int(sum(v['same_mask'] for v in ident.values()))}
  if out['missing_crops']:
    return out
  objects = {k: len(f1[k]['run']['records']) for k in unseen.CROPS}
  total = {r: float(sum(v.values())) for r, v in sec.items()}
  n = sum(objects.values())
  ref = [r for k in unseen.CROPS for r in f1[k]['run']['records']]
  cand = [r for k in unseen.CROPS for r in f32x32[k]['run']['records']]
  out.update({'objects': int(n), 'total_seconds': total, 'seconds_per_object': {r: t / n for r, t in total.items()},
              'batching_speedup_fp32': total['F1'] / total['F32x32'],
              'precision_speedup_at_32': total['F32x32'] / total['F5'],
              'fast_ffn_over_fp32_one_at_a_time': total['F1'] / total['F5'],
              'pooled': {'F32x32': unseen.pooled_means(cand), 'F1': unseen.pooled_means(ref)},
              'changes_vs_F1': unseen._flips(ref, cand),  # pylint: disable=protected-access
              'counts': {'F32x32': unseen.object_counts(cand), 'F1': unseen.object_counts(ref)},
              'runaways_vs_F1': unseen.runaways(cand, ref)})
  sn = {how: {k: segneuron['segneuron_b200'][str(k)][how] for k in unseen.CROPS} for how in ('end_to_end', 'without_startup')}
  out['segneuron_break_even'] = {how: unseen_b200.compare(sn[how], sec['F32x32'], objects, 'float32 FFN, 32 at once')
                                 for how in sn}
  return out
