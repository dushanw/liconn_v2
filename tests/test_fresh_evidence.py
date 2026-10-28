"""The recorded fresh-seed runs and report (docs/fresh_results.md) match their evidence."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from liconn_v2 import data
from liconn_v2 import fresh

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'fresh_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
KINDS = ('new_merge', 'new_early_stop', 'avoided_merge', 'extension')


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _round(x, digits=2):
  return round(x, digits) if isinstance(x, float) else x


class FreshEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    for v in ('fp32', 'fp16_head32'):
      staged = dict(reversed(line.split()) for line in
                    (RUNS / f'fresh-{v}-20261023' / 'source_sha256.txt').read_text().splitlines())
      for rel, expected in EVIDENCE['run_source_sha256'].items():
        self.assertEqual(staged.get(rel), expected, f'{v}: {rel}')
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_report_code_is_not_claimed_as_gpu_code(self):
    for rel in ('liconn_v2/erl.py', 'examples/judge_fresh.py'):
      self.assertNotIn(rel, EVIDENCE['run_source_sha256'])

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_runs_record_format_platform_jobs_and_seeds(self):
    crops = [c['index'] for c in fresh.load_manifest()['crops']]
    names = [s['name'] for s in fresh.recorded_seeds()]
    for v in ('fp32', 'fp16_head32'):
      runs = [json.loads((RUNS / f'fresh-{v}-20261023' / f'crop-{k:02d}' / 'benchmark.json').read_text()) for k in crops]
      self.assertEqual([r['name'] for run in runs for r in run['records']], names)
      for run in runs:
        self.assertEqual(run['platform']['slurm_job'], str(EVIDENCE['slurm_jobs'][v]))
        self.assertEqual(run['platform']['xla_flags'], '--xla_gpu_deterministic_ops=true')
        for r in run['records']:
          self.assertEqual((r['number_format'], r['backend'], r['device_kind']), (v, 'gpu', 'Tesla V100-SXM2-16GB'))

  def test_documented_report(self):
    doc = EVIDENCE['documented']
    self.assertTrue(doc['gates']['manyseed']['passes'])
    self.assertFalse(doc['gates']['fresh']['passes'])  # owner kept the 3x gate: fresh ERL withheld
    with tempfile.TemporaryDirectory() as d:
      subprocess.run([sys.executable, str(ROOT / 'examples' / 'judge_fresh.py'), '--recorded', '--out', d],
                     check=True, capture_output=True)
      rep = json.loads((Path(d) / 'fresh_report.json').read_text())
    recorded = json.loads((RUNS / 'fresh-report-20261023' / 'fresh_report.json').read_text())
    self.assertEqual(rep, recorded)  # the recorded report is what the current code computes
    v = rep['verdict']
    self.assertEqual((v['counts'], v['dendrite_counts'], v['rules'], v['passes']),
                     (doc['counts'], doc['dendrite_counts'], doc['rules'], doc['passes']))
    self.assertEqual(round(100 * v['mean_recall_change'], 2), doc['mean_recall_change_pts_2dp'])
    self.assertEqual(round(100 * v['mean_precision_change'], 2), doc['mean_precision_change_pts_2dp'])
    self.assertEqual({k: [x for x in KINDS if e[x]] for k, e in v['flipped_seeds'].items()}, doc['flipped'])
    for s in ('fresh', 'manyseed'):
      got = {k: {a: (_round(100 * b) if a.startswith('mean_') else _round(b)) for a, b in m.items()}
             for k, m in rep[s]['metrics'].items()}
      self.assertEqual(got, doc['metrics_2dp'][s], s)
      erl = rep[s]['erl']
      got_erl = erl if isinstance(erl, str) else {k: {a: _round(b) for a, b in m.items()} for k, m in erl.items()}
      self.assertEqual(got_erl, doc['erl_2dp'][s], s)
      self.assertEqual(isinstance(erl, str), not rep[s]['gate']['passes'], s)  # withheld exactly when the gate fails
      self.assertEqual({a: _round(b, 3) for a, b in rep[s]['gate'].items()}, doc['gates'][s], s)


if __name__ == '__main__':
  unittest.main()
