"""Many-seed verdict for a speed variant (docs/manyseed_design.md).

  judge_manyseed.py --dev [--out DIR]
  judge_manyseed.py --ref RUN_DIR ... --cand RUN_DIR ... [--out DIR]
  judge_manyseed.py --recorded VARIANT [--out DIR]

--dev judges float16 on the 24 development seeds (benchmark crop A and held-out crop
B) from the recorded runs in the data folder; for information only, since those seeds
were examined before the rule was fixed. Otherwise --ref and --cand list one run
directory (benchmark.json) per crop, in the same crop order, and every seed recorded
in docs/manyseed_seeds.json must be present. --recorded judges a recorded candidate
run in the data folder against the recorded float32 run.

Output: DIR/manyseed_dev_verdict.json or DIR/manyseed_verdict.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top

RECORDED = {'fp32': 'manyseed-fp32-20261021', 'fp16': 'manyseed-fp16-20261021',
            'fp16_head32': 'manyseed-fp16_head32-20261022'}
DEV_PAIRS = [('gpu-baseline-a-20261019/merged', 'precision-fp16-20261021/merged'),
             ('heldout-fp32-20261021/merged', 'heldout-fp16-20261021/merged')]


def load(run_dir) -> dict:
  return json.loads((Path(run_dir) / 'benchmark.json').read_text())


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--dev', action='store_true')
  p.add_argument('--ref', nargs='*', default=[])
  p.add_argument('--cand', nargs='*', default=[])
  p.add_argument('--recorded', choices=sorted(set(RECORDED) - {'fp32'}))
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  if a.dev:
    runs = data.cache_dir() / 'runs'
    ref = [load(runs / r) for r, _ in DEV_PAIRS]
    cand = [load(runs / c) for _, c in DEV_PAIRS]
    result = manyseed.verdict(ref, cand)
    result['runs'] = DEV_PAIRS
    result['note'] = 'development seeds, examined before the rule was fixed; information only'
    name = 'manyseed_dev_verdict.json'
  else:
    if a.recorded:
      runs = data.cache_dir() / 'runs'
      crops = [f"crop-{c['index']:02d}" for c in manyseed.load_manifest()['crops']]
      a.ref = [str(runs / RECORDED['fp32'] / c) for c in crops]
      a.cand = [str(runs / RECORDED[a.recorded] / c) for c in crops]
    if not a.ref or len(a.ref) != len(a.cand):
      raise SystemExit('give one --ref and one --cand run directory per crop, in crop order')
    expected = manyseed.recorded_seeds()
    ckpt = {k: v for k, v in data.PINNED_SHA256.items() if k.startswith('ckpt/')}
    pinned = [{**c['sha256'], **ckpt} for c in manyseed.load_manifest()['crops']]
    result = manyseed.verdict([load(d) for d in a.ref], [load(d) for d in a.cand], expected, pinned)
    result['runs'] = list(zip(a.ref, a.cand))
    name = 'manyseed_verdict.json'
  (out / name).write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps({k: result[k] for k in ('seeds', 'counts', 'upper_bound_95', 'mean_recall_change',
                                           'mean_precision_change', 'rules', 'passes')}, indent=2))


if __name__ == '__main__':
  main()
