"""Recorded runs re-judged under the owner's 2% rule (docs/tolerance2_results.md).

  judge_tolerance2.py --recorded [--out DIR]

Judges float16 and fp16_head32 on the first 300 seeds and fp16_head32 on the 300 fresh
seeds, against their float32 runs, under the original rule (liconn_v2/manyseed.py) and
the 2% rule (liconn_v2/tolerance2.py). No new runs.

Output: DIR/tolerance2_report.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import fresh  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import tolerance2  # pylint: disable=g-import-not-at-top
import judge_fresh  # pylint: disable=g-import-not-at-top,wrong-import-position
import judge_manyseed  # pylint: disable=g-import-not-at-top,wrong-import-position


def _load(d: Path) -> dict:
  return json.loads((d / 'benchmark.json').read_text())


def verdicts() -> dict:
  runs = data.cache_dir() / 'runs'
  ckpt = {k: v for k, v in data.PINNED_SHA256.items() if k.startswith('ckpt/')}
  out = {}
  crops = [c['index'] for c in manyseed.load_manifest()['crops']]
  pinned = [{**c['sha256'], **ckpt} for c in manyseed.load_manifest()['crops']]
  ref = [_load(runs / judge_manyseed.RECORDED['fp32'] / f'crop-{k:02d}') for k in crops]
  for v in ('fp16', 'fp16_head32'):
    cand = [_load(runs / judge_manyseed.RECORDED[v] / f'crop-{k:02d}') for k in crops]
    out[f'first300_{v}'] = manyseed.verdict(ref, cand, manyseed.recorded_seeds(), pinned)
  crops = [c['index'] for c in fresh.load_manifest()['crops']]
  pinned = [{**c['sha256'], **ckpt} for c in fresh.load_manifest()['crops']]
  ref = [_load(runs / judge_fresh.FRESH_RUNS['fp32'] / f'crop-{k:02d}') for k in crops]
  cand = [_load(runs / judge_fresh.FRESH_RUNS['fp16_head32'] / f'crop-{k:02d}') for k in crops]
  out['fresh_fp16_head32'] = judge_fresh.fresh_verdict(ref, cand, fresh.recorded_seeds(), pinned, crops)
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--recorded', action='store_true', required=True)
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  report = {'design': 'docs/tolerance2_results.md', 'decided': tolerance2.DECIDED, 'cases': {}}
  for name, v in verdicts().items():
    report['cases'][name] = {'counts': v['counts'], 'original_rule': {'rules': v['rules'], 'passes': v['passes']},
                             'rule_2pct': tolerance2.apply(v)}
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / 'tolerance2_report.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps({k: {'counts': c['counts'], 'original': c['original_rule']['passes'],
                        '2%': c['rule_2pct']['passes']} for k, c in report['cases'].items()}, indent=2))


if __name__ == '__main__':
  main()
