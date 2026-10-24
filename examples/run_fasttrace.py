"""Fast tracer runs (docs/fasttrace_design.md): bitwise identity and speed.

  run_fasttrace.py --variant fp32|fp16_head32 --set benchmark|fresh [--recorded PATH]
                   [--compare-reference] [--out DIR]

Traces the set's seeds with liconn_v2.fasttrace.FastTracer and compares each trace with
the recorded run (canvas sha256, history sha256, steps). --recorded is a benchmark.json
(set benchmark) or a run directory with one crop-XX/benchmark.json per fresh crop (set
fresh); without it (no recorded run exists for that set and format) only the in-job
comparison is made. With --compare-reference, each seed is also traced with the reference path in
the same job, for timing and an in-job identity check. Set LICONN_BACKEND=gpu on the
cluster.

Output: DIR/fasttrace_run.json
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import earlystop  # pylint: disable=g-import-not-at-top
from liconn_v2 import fasttrace  # pylint: disable=g-import-not-at-top
from liconn_v2 import fresh  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position


def work(set_name: str, recorded: Path):
  """(crop label, raw volume loader, seeds, recorded records) per crop."""
  if set_name == 'benchmark':
    gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
    recs = {r['name']: r for r in json.loads(recorded.read_text())['records']} if recorded else {}
    yield 'A', lambda: reference.normalize(data.load_raw_xyz()), benchmark.select_seeds(gt), recs
    return
  for c in fresh.load_manifest()['crops']:
    k = c['index']
    recs = ({r['name']: r for r in json.loads((recorded / f'crop-{k:02d}' / 'benchmark.json').read_text())['records']}
            if recorded else {})
    seeds = [{**s, 'seed_zyx': tuple(s['seed_zyx'])} for s in fresh.recorded_seeds() if s['crop'] == k]
    yield k, (lambda k=k: reference.normalize(fresh.load_raw_xyz(k))), seeds, recs


def fingerprint(t) -> dict:
  return {'steps': t.iterations, 'canvas_sha256': reference.canvas_sha256(t.seed),
          'history_sha256': reference.history_sha256(t.history)}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--variant', required=True, choices=('fp32', 'fp16_head32'))
  p.add_argument('--set', required=True, choices=('benchmark', 'fresh'))
  p.add_argument('--recorded')
  p.add_argument('--compare-reference', action='store_true')
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  started = time.time()
  records = []
  with reference.Tracer(data.checkpoint_path()) as tracer:
    if a.variant != 'fp32':
      earlystop.install(tracer, a.variant)
    fast = fasttrace.FastTracer(tracer)
    for crop, load_raw, seeds, recs in work(a.set, Path(a.recorded) if a.recorded else None):
      raw = load_raw()
      for s in seeds:
        t = fast.trace(raw, s['seed_zyx'])
        f = fingerprint(t)
        want = recs.get(s['name'])
        r = {'name': s['name'], 'crop': crop, **f, 'fast_seconds': t.trace_seconds,
             'matches_recorded': None if want is None else
             (f['steps'], f['canvas_sha256'], f['history_sha256'])
             == (want['steps'], want['canvas_sha256'], want['history_sha256'])}
        if a.compare_reference:
          ref = tracer.trace(raw, s['seed_zyx'])
          r['reference_seconds'] = ref.trace_seconds
          r['matches_reference'] = fingerprint(ref) == f
        records.append(r)
        print(json.dumps({k: v for k, v in r.items() if 'sha256' not in k}), flush=True)
  steps = sum(r['steps'] for r in records)
  summary = {'seeds': len(records),
             'all_match_recorded': None if not a.recorded else all(r['matches_recorded'] for r in records),
             'steps': steps, 'fast_ms_per_step': 1000 * sum(r['fast_seconds'] for r in records) / steps}
  if a.compare_reference:
    summary['all_match_reference'] = all(r['matches_reference'] for r in records)
    summary['reference_ms_per_step'] = 1000 * sum(r['reference_seconds'] for r in records) / steps
    summary['speedup_over_reference'] = summary['reference_ms_per_step'] / summary['fast_ms_per_step']
  result = {'design': 'docs/fasttrace_design.md', 'variant': a.variant, 'set': a.set, 'recorded': a.recorded,
            'records': records, 'summary': summary, 'wall_seconds_total': time.time() - started,
            'platform': provenance()}
  (out / 'fasttrace_run.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
