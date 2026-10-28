"""The recorded many-seed runs (docs/manyseed_results.md) match their evidence."""

import ast
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
EVIDENCE = json.loads((ROOT / 'docs' / 'manyseed_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
CROPS = [f'crop-{k:02d}' for k in range(len(manyseed.load_manifest()['crops']))]


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(n) -> str:
  if isinstance(n, ast.Assign):
    return ','.join(t.id for t in n.targets if isinstance(t, ast.Name)) or ast.dump(n)
  if isinstance(n, (ast.Import, ast.ImportFrom)):
    return 'import:' + ','.join(a.asname or a.name for a in n.names)
  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
    return 'docstring'
  return getattr(n, 'name', None) or ast.dump(n)


def _definitions(path: Path) -> dict:
  return {_key(n): ast.dump(n) for n in ast.parse(path.read_text()).body}


def _run(variant):
  return [json.loads((RUNS / f'manyseed-{variant}-20261021' / c / 'benchmark.json').read_text()) for c in CROPS]


class ManyseedEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    edits = EVIDENCE['post_run_edits']
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      if rel not in edits:
        self.assertEqual(_sha(ROOT / rel), expected, rel)
    for variant in ('fp32', 'fp16'):
      staged = dict(reversed(line.split()) for line in
                    (RUNS / f'manyseed-{variant}-20261021' / 'source_sha256.txt').read_text().splitlines())
      for rel, expected in EVIDENCE['run_source_sha256'].items():
        self.assertIn(rel, staged, f'{variant}: {rel} missing from the staged source manifest')
        self.assertEqual(staged[rel], expected, f'{variant}: {rel}')

  def test_post_run_edits_change_only_the_declared_definitions(self):
    for rel, edit in EVIDENCE['post_run_edits'].items():
      ran = RUNS / 'manyseed-fp32-20261021' / 'source' / rel
      self.assertEqual(_sha(ran), EVIDENCE['run_source_sha256'][rel], rel)
      if edit.get('reformatted_json'):
        self.assertEqual(json.loads((ROOT / rel).read_text()), json.loads(ran.read_text()), rel)
        self.assertEqual((ROOT / rel).read_text(), manyseed.seeds_text(json.loads(ran.read_text())['seeds']), rel)
        continue
      old, new = _definitions(ran), _definitions(ROOT / rel)
      changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
      self.assertEqual(changed, sorted(edit['changed_definitions']), rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_runs_record_format_platform_jobs_and_seeds(self):
    expected = [s['name'] for s in manyseed.recorded_seeds()]
    for variant in ('fp32', 'fp16'):
      runs = _run(variant)
      self.assertEqual([r['name'] for run in runs for r in run['records']], expected)
      for run in runs:
        self.assertEqual(run['platform']['slurm_job'], str(EVIDENCE['slurm_jobs'][variant]))
        self.assertEqual(run['platform']['xla_flags'], '--xla_gpu_deterministic_ops=true')
        for r in run['records']:
          self.assertEqual((r['number_format'], r['backend'], r['device_kind']),
                           (variant, 'gpu', 'Tesla V100-SXM2-16GB'))

  def test_documented_verdict(self):
    doc = EVIDENCE['documented']
    refs = [str(RUNS / 'manyseed-fp32-20261021' / c) for c in CROPS]
    cands = [str(RUNS / 'manyseed-fp16-20261021' / c) for c in CROPS]
    with tempfile.TemporaryDirectory() as d:
      subprocess.run([sys.executable, str(ROOT / 'examples' / 'judge_manyseed.py'), '--ref', *refs,
                      '--cand', *cands, '--out', d], check=True, capture_output=True)
      v = json.loads((Path(d) / 'manyseed_verdict.json').read_text())
    self.assertEqual((v['seeds'], v['dendrite_seeds']), (doc['seeds'], doc['dendrite_seeds']))
    self.assertEqual((v['counts'], v['dendrite_counts']), (doc['counts'], doc['dendrite_counts']))
    self.assertEqual((v['rules'], v['passes']), (doc['rules'], doc['passes']))
    self.assertEqual(round(100 * v['mean_recall_change'], 2), doc['mean_recall_change_pts_2dp'])
    self.assertEqual(round(100 * v['mean_precision_change'], 2), doc['mean_precision_change_pts_2dp'])
    self.assertEqual({k: round(100 * x, 1) for k, x in v['upper_bound_95'].items() if k != 'note'},
                     doc['upper_bound_pct_1dp'])
    got = {k: {'recall_pts_1dp': round(100 * e['recall_change'], 1),
               'precision_pts_1dp': round(100 * e['precision_change'], 1),
               'events': [x for x in ('new_merge', 'new_early_stop', 'avoided_merge', 'extension') if e[x]]}
           for k, e in v['flipped_seeds'].items()}
    self.assertEqual(got, doc['flipped'])
    self.assertEqual(v['reference_already_merged'], doc['reference_already_merged'])
    for variant in ('fp32', 'fp16'):
      recs = [r for run in _run(variant) for r in run['records']]
      steps = sum(r['steps'] for r in recs)
      self.assertEqual(steps, doc['total_steps'][variant])
      self.assertEqual(round(1000 * sum(r['trace_seconds'] for r in recs) / steps, 2), doc['ms_per_step_2dp'][variant])


if __name__ == '__main__':
  unittest.main()
