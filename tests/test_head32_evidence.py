"""The recorded fp16_head32 many-seed run (docs/head32_results.md) matches its evidence."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from liconn_v2 import data
from liconn_v2 import manyseed

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'head32_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
RUN = RUNS / 'manyseed-fp16_head32-20261022'
CROPS = [f'crop-{k:02d}' for k in range(len(manyseed.load_manifest()['crops']))]
KINDS = ('new_merge', 'new_early_stop', 'avoided_merge', 'extension')


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class Head32EvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    staged = dict(reversed(line.split()) for line in (RUN / 'source_sha256.txt').read_text().splitlines())
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(staged.get(rel), expected, rel)
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_run_records_format_platform_job_and_seeds(self):
    runs = [json.loads((RUN / c / 'benchmark.json').read_text()) for c in CROPS]
    self.assertEqual([r['name'] for run in runs for r in run['records']], [s['name'] for s in manyseed.recorded_seeds()])
    for run in runs:
      self.assertEqual(run['platform']['slurm_job'], str(EVIDENCE['slurm_jobs']['fp16_head32']))
      self.assertEqual(run['platform']['xla_flags'], '--xla_gpu_deterministic_ops=true')
      for r in run['records']:
        self.assertEqual((r['number_format'], r['backend'], r['device_kind']), ('fp16_head32', 'gpu', 'Tesla V100-SXM2-16GB'))
    recs = [r for run in runs for r in run['records']]
    steps = sum(r['steps'] for r in recs)
    doc = EVIDENCE['documented']
    self.assertEqual(steps, doc['total_steps'])
    self.assertEqual(round(1000 * sum(r['trace_seconds'] for r in recs) / steps, 2), doc['ms_per_step_2dp'])

  def test_documented_verdict_and_split(self):
    doc = EVIDENCE['documented']
    with tempfile.TemporaryDirectory() as d:
      subprocess.run([sys.executable, str(ROOT / 'examples' / 'judge_manyseed.py'), '--recorded', 'fp16_head32',
                      '--out', d], check=True, capture_output=True)
      v = json.loads((Path(d) / 'manyseed_verdict.json').read_text())
    self.assertEqual((v['counts'], v['dendrite_counts'], v['rules'], v['passes']),
                     (doc['counts'], doc['dendrite_counts'], doc['rules'], doc['passes']))
    self.assertEqual(round(100 * v['mean_recall_change'], 2), doc['mean_recall_change_pts_2dp'])
    self.assertEqual(round(100 * v['mean_precision_change'], 2), doc['mean_precision_change_pts_2dp'])
    self.assertEqual({k: round(100 * x, 1) for k, x in v['upper_bound_95'].items() if k != 'note'},
                     doc['upper_bound_pct_1dp'])
    self.assertEqual({k: [x for x in KINDS if e[x]] for k, e in v['flipped_seeds'].items()}, doc['flipped'])
    analysis = json.loads((RUNS / 'earlystop-analysis-20261022' / 'earlystop_analysis.json').read_text())
    selection = set(analysis['seeds']['early_stop'] + analysis['seeds']['controls'])
    fp16 = json.loads((RUNS / 'manyseed-verdict-fp16-20261021' / 'manyseed_verdict.json').read_text())
    for name, verdict in (('fp16', fp16), ('fp16_head32', v)):
      for group, keep in (('selection', lambda n: n in selection), ('other', lambda n: n not in selection)):
        got = {k: sum(1 for n, e in verdict['flipped_seeds'].items() if keep(n) and e[k])
               for k in ('new_early_stop', 'new_merge')}
        self.assertEqual(got, doc['split'][name][group], (name, group))


if __name__ == '__main__':
  unittest.main()
