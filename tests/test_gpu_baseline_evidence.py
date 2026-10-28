"""The recorded GPU baseline (docs/gpu_baseline.md) matches its evidence.

The GPU runs themselves happened on Wahab; this test checks, without a GPU, that
the code which ran is the code checked here, that the run outputs in the data
folder are the recorded ones, and that the documented comparisons follow from them.
"""

import hashlib
import json
from pathlib import Path
import unittest

from liconn_v2 import compare
from liconn_v2 import data

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'gpu_baseline_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
CPU = RUNS / 'baseline-20261018'
GPU_A = RUNS / 'gpu-baseline-a-20261019' / 'merged'
GPU_B = RUNS / 'gpu-baseline-b-20261019' / 'merged'


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class GpuBaselineEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_run_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)
    manifest = dict(reversed(line.split()) for line in
                    (RUNS / 'gpu-baseline-a-20261019' / 'source_sha256.txt').read_text().splitlines())
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(manifest[rel], expected, rel)

  def test_job_logs_name_the_base_commit(self):
    # The jobs ran the last commit plus uncommitted changes (identified by the
    # file hashes above).
    for run in ('gpu-baseline-a-20261019', 'gpu-baseline-b-20261019'):
      lines = [l for p in sorted((RUNS / run / 'logs').glob('*.out'))
               for l in p.read_text().splitlines() if ' -> seed ' in l]
      self.assertEqual(len(lines), 12, run)
      self.assertEqual({l.split()[-1] for l in lines}, {EVIDENCE['base_commit'] + '-dirty'}, run)

  def test_gpu_runs_record_their_platform(self):
    for run_dir, job in ((GPU_A, 7358888), (GPU_B, 7358889)):
      run = compare.load_run(run_dir)
      self.assertEqual(len(run['records']), 12)
      for r in run['records']:
        self.assertEqual(r['backend'], 'gpu')
        self.assertIn('V100', r['device_kind'])
      for p in run['platforms']:
        self.assertEqual(p['backend'], 'gpu')
        self.assertEqual(p['xla_flags'], '--xla_gpu_deterministic_ops=true')
        self.assertEqual(p['slurm_job'], str(job))
        self.assertEqual(p['versions']['nvidia-cudnn-cu12'], '9.6.0.74')
        self.assertEqual(p['versions']['jaxlib'], '0.4.38')

  def test_gpu_runs_reproduce_each_other(self):
    r = compare.compare_runs(GPU_A, GPU_B)
    self.assertTrue(r['complete'] and r['inputs_sha256_checked'])
    self.assertEqual(len(r['per_seed']), 12)
    for row in r['per_seed']:
      self.assertTrue(row['same_canvas'], row['name'])
      self.assertTrue(row['same_history'], row['name'])
      self.assertEqual(row['mask_iou'], 1.0, row['name'])

  def test_documented_cpu_vs_gpu_comparison(self):
    r = compare.compare_runs(CPU, GPU_A)
    got = {x['name']: {'mask_iou_6dp': round(x['mask_iou'], 6), 'cpu_steps': x['base_steps'],
                       'gpu_steps': x['cand_steps']} for x in r['per_seed']}
    self.assertEqual(got, EVIDENCE['documented_cpu_vs_gpu'])
    self.assertEqual(r['seeds_flagged_for_review'], EVIDENCE['documented_flagged_for_review'])
    self.assertTrue(r['complete'])
    self.assertTrue(r['inputs_sha256_checked'])
    self.assertTrue(r['passes_tolerance'])
    # Within the owner-confirmed tolerance (all rules at 1%), with and without the
    # sensitive seeds.
    for group in ('all', 'without_sensitive_seeds'):
      self.assertTrue(r['overall'][group]['tolerance']['within_tolerance'], group)
    self.assertEqual(r['seeds_outside_seed_rules'], [])

  def test_reference_run_is_gpu_run_a(self):
    self.assertEqual(RUNS / compare.REFERENCE_RUN, GPU_A)


if __name__ == '__main__':
  unittest.main()
