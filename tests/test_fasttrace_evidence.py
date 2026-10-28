"""The recorded fast-tracer runs (docs/fasttrace_results.md) match their evidence."""

import hashlib
import json
from pathlib import Path
import unittest

from liconn_v2 import data
from liconn_v2 import fresh

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'fasttrace_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
DIRS = {'fresh_fp32': 'fasttrace-fresh-fp32-20261024', 'fresh_fp16_head32': 'fasttrace-fresh-fp16_head32-20261024',
        'bench_fp32': 'fasttrace-bench-fp32-20261024', 'bench_fp16_head32': 'fasttrace-bench-fp16_head32-20261024'}
FINGERPRINT = ('steps', 'canvas_sha256', 'history_sha256')


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(key):
  return json.loads((RUNS / DIRS[key] / 'fasttrace_run.json').read_text())


def _fp(record):
  return tuple(record[k] for k in FINGERPRINT)


def _fresh_recorded(variant):
  out = {}
  for c in fresh.load_manifest()['crops']:
    path = RUNS / f'fresh-{variant}-20261023' / f"crop-{c['index']:02d}" / 'benchmark.json'
    for r in json.loads(path.read_text())['records']:
      out[r['name']] = r
  return out


class FasttraceEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    for key in DIRS:
      staged = dict(reversed(l.split()) for l in (RUNS / DIRS[key] / 'source_sha256.txt').read_text().splitlines())
      for rel, expected in EVIDENCE['run_source_sha256'].items():
        self.assertEqual(staged.get(rel), expected, f'{key}: {rel}')
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_runs_record_jobs_and_formats(self):
    for key in DIRS:
      run = _run(key)
      self.assertEqual(run['platform']['slurm_job'], str(EVIDENCE['slurm_jobs'][key]))
      self.assertEqual(run['platform']['xla_flags'], '--xla_gpu_deterministic_ops=true')
      self.assertEqual(run['variant'], key.split('_', 1)[1])

  def test_fresh_traces_equal_the_recorded_runs_independently(self):
    names = [s['name'] for s in fresh.recorded_seeds()]
    for variant in ('fp32', 'fp16_head32'):
      recs = _run(f'fresh_{variant}')['records']
      self.assertEqual([r['name'] for r in recs], names)
      recorded = _fresh_recorded(variant)
      for r in recs:  # steps, canvas sha256 and history sha256, compared here, not taken from the runner
        self.assertEqual(_fp(r), _fp(recorded[r['name']]), (variant, r['name']))

  def test_benchmark_float32_equals_the_recorded_run_independently(self):
    recorded = {r['name']: r for r in json.loads(
        (RUNS / 'gpu-baseline-a-20261019' / 'merged' / 'benchmark.json').read_text())['records']}
    recs = _run('bench_fp32')['records']
    self.assertEqual(sorted(r['name'] for r in recs), sorted(recorded))
    for r in recs:
      self.assertEqual(_fp(r), _fp(recorded[r['name']]), r['name'])

  def test_benchmark_runs_match_the_reference_path_in_the_same_job(self):
    # No recorded fp16_head32 run exists on these 12 seeds; here the runner compared the
    # fast and reference traces' fingerprints in the same job.
    for key in ('bench_fp32', 'bench_fp16_head32'):
      recs = _run(key)['records']
      self.assertEqual(len(recs), 12)
      self.assertTrue(all(r['matches_reference'] for r in recs), key)

  def test_documented_summaries(self):
    for key, doc in EVIDENCE['documented'].items():
      s = _run(key)['summary']
      self.assertEqual({a: (round(b, 2) if isinstance(b, float) else b) for a, b in s.items()}, doc, key)


if __name__ == '__main__':
  unittest.main()
