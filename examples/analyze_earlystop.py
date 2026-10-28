"""Early-stop analyses A, B and C (docs/earlystop_design.md) from recorded runs.

  analyze_earlystop.py --recorded [--out DIR]
  analyze_earlystop.py --fp32 DIR --fp16 DIR [--shift DIR] [--out DIR]

--fp32 is the float32 run with open-loop evaluation, --fp16 the float16 run, --shift
the optional fp16_shift open-loop run (all from examples/run_earlystop.py). --recorded
uses the runs recorded in the data folder (docs/earlystop_evidence.json).

Output: DIR/earlystop_analysis.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import earlystop  # pylint: disable=g-import-not-at-top

RECORDED = {'fp32': 'earlystop-fp32open-20261022', 'fp16': 'earlystop-fp16-20261022',
            'shift': 'earlystop-shift-20261022'}


def load(run_dir: Path):
  run = json.loads((run_dir / 'earlystop_run.json').read_text())
  steps = {n: earlystop.load_steps(run_dir / 'steps' / f'{n}.json') for n in run['seeds']}
  return run, steps


def analyze(fp32_dir: Path, fp16_dir: Path, shift_dir: Path | None) -> dict:
  r32, s32 = load(fp32_dir)
  r16, s16 = load(fp16_dir)
  names = r32['seeds']
  early = [n for n in names if n in earlystop.EARLY_STOP_SEEDS]
  controls = [n for n in names if n not in earlystop.EARLY_STOP_SEEDS]
  problems = []
  if r16['seeds'] != names:
    raise SystemExit('the float32 and float16 runs traced different seeds')
  if sorted(early) != sorted(earlystop.EARLY_STOP_SEEDS):
    problems.append('not all 10 early-stop seeds were traced')
  for run in (r32, r16):
    if [rec['name'] for rec in run['records']] != names:
      problems.append(f"{run['variant']}: records do not cover exactly the traced seeds")
    for rec in run['records']:
      if not rec['reproduces_manyseed']:
        problems.append(f"{run['variant']} {rec['name']}: does not reproduce the many-seed run")
      if rec.get('emulation_mismatches'):
        problems.append(f"{rec['name']}: open-loop emulation differs from FFN at {rec['emulation_mismatches']} steps")

  if problems:  # the design rejects runs that fail these checks before any analysis
    raise SystemExit('runs rejected, no analysis: ' + '; '.join(problems))

  # A. where float16's trace stops (primary: the frontier; secondary: the first difference)
  front = {n: earlystop.frontier(s32[n], s16[n]) for n in names}
  first = {n: earlystop.first_difference(s32[n], s16[n]) for n in names}

  def kinds(result, group, labels):
    out = {k: 0 for k in labels + ('none',)}
    for n in group:
      out[result[n]['kind'] if result[n] else 'none'] += 1
    return out

  fk = ('threshold', 'argmax', 'queued_not_evaluated')
  dk = ('threshold', 'argmax', 'order')
  a = {'frontier': front, 'first_difference': first,
       'frontier_kinds': {'early_stop': kinds(front, early, fk), 'controls': kinds(front, controls, fk)},
       'first_difference_kinds': {'early_stop': kinds(first, early, dk), 'controls': kinds(first, controls, dk)}}
  a['supported'] = a['frontier_kinds']['early_stop']['threshold'] >= earlystop.THRESHOLD_SUPPORT

  # B. bias of float16 on the same inputs (primary: the controls)
  near = {n: earlystop.near_threshold_errors(earlystop.face_errors(s32[n], 'fp16')).tolist() for n in names}
  every = {n: [e for _, e in earlystop.face_errors(s32[n], 'fp16')] for n in names}
  b = {'near_threshold_controls': earlystop.bootstrap_mean({n: near[n] for n in controls}),
       'near_threshold_early_stop': earlystop.bootstrap_mean({n: near[n] for n in early}),
       'all_faces_controls': earlystop.bootstrap_mean({n: every[n] for n in controls}),
       'losses_gains_controls': earlystop.losses_gains([x for n in controls for x in s32[n]], 'fp16'),
       'losses_gains_early_stop': earlystop.losses_gains([x for n in early for x in s32[n]], 'fp16')}
  b['non_finite'] = {v: earlystop.non_finite_faces([x for n in names for x in s32[n]], v) for v in earlystop.CANDIDATES}
  ci = b['near_threshold_controls']['ci95']
  b['supported'] = ci is not None and ci[1] < 0
  b['shift_c'] = -b['near_threshold_controls']['mean'] if b['supported'] else None

  # C. candidate screen on the same inputs (selection on the early-stop seeds)
  t = r32['seconds_per_eval']
  speed = {v: t['fp32'] / t[v] for v in earlystop.CANDIDATES}
  on_early = {v: earlystop.screen({n: s32[n] for n in early}, v) for v in earlystop.CANDIDATES}
  on_all = {v: earlystop.screen(s32, v) for v in earlystop.CANDIDATES}
  if shift_dir is not None:
    rs, ss = load(shift_dir)
    if (rs['seeds'] != names or [r['name'] for r in rs['records']] != names
        or not all(r['reproduces_manyseed'] and r.get('emulation_mismatches') == 0 for r in rs['records'])):
      problems.append('the fp16_shift run does not match the float32 run')
    if b['shift_c'] is None or abs(rs['shift'] - b['shift_c']) > 1e-6:
      problems.append('the fp16_shift run used a different offset than analysis B gives')
    if problems:
      raise SystemExit('runs rejected, no analysis: ' + '; '.join(problems))
    on_early['fp16_shift'] = earlystop.screen({n: ss[n] for n in early}, 'fp16_shift')
    on_all['fp16_shift'] = earlystop.screen(ss, 'fp16_shift')
    ts = rs['seconds_per_eval']
    speed['fp16_shift'] = ts['fp32'] / ts['fp16_shift']
  elif b['supported']:
    problems.append('analysis B found a bias but no fp16_shift run was given')
  if problems:
    raise SystemExit('runs rejected, no analysis: ' + '; '.join(problems))
  c = {'speedup_per_eval': speed, 'screens_early_stop': on_early, 'screens_all': on_all,
       'selected': earlystop.select_candidate(on_early, speed)}
  exploratory = {'note': 'not pre-registered; added after the runs',
                 'plateau': earlystop.plateau_stats(s32, earlystop.CANDIDATES)}
  return {'design': 'docs/earlystop_design.md', 'seeds': {'early_stop': early, 'controls': controls},
          'problems': problems, 'A': a, 'B': b, 'C': c, 'exploratory': exploratory}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--recorded', action='store_true')
  p.add_argument('--fp32')
  p.add_argument('--fp16')
  p.add_argument('--shift')
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  if a.recorded:
    runs = data.cache_dir() / 'runs'
    shift = runs / RECORDED['shift']
    result = analyze(runs / RECORDED['fp32'], runs / RECORDED['fp16'], shift if shift.is_dir() else None)
  else:
    result = analyze(Path(a.fp32), Path(a.fp16), Path(a.shift) if a.shift else None)
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / 'earlystop_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps({'problems': result['problems'],
                    'A': {k: v for k, v in result['A'].items() if k not in ('frontier', 'first_difference')},
                    'B': result['B'], 'C': result['C']}, indent=2))


if __name__ == '__main__':
  main()
