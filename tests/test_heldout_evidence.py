"""The recorded held-out runs (docs/heldout_results.md) match their evidence."""

import hashlib
import json
from pathlib import Path
import unittest

from liconn_v2 import compare
from liconn_v2 import data

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'heldout_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
FP32 = RUNS / 'heldout-fp32-20261021' / 'merged'
FP16 = RUNS / 'heldout-fp16-20261021' / 'merged'


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _ms(run) -> float:
  return 1000 * sum(r['trace_seconds'] for r in run['records']) / sum(r['steps'] for r in run['records'])


class HeldoutEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_runs_record_format_platform_and_jobs(self):
    for run_dir, fmt, job in ((FP32, 'fp32', 7370167), (FP16, 'fp16', 7370168)):
      run = compare.load_run(run_dir)
      self.assertEqual(len(run['records']), 12)
      for r in run['records']:
        self.assertEqual((r['number_format'], r['backend']), (fmt, 'gpu'))
        self.assertTrue(r['name'].startswith('heldout_'))
      self.assertEqual({r['device_kind'] for r in run['records']}, {'Tesla V100-SXM2-16GB'})
      for p in run['platforms']:
        self.assertEqual(p['slurm_job'], str(job))
        self.assertEqual(p['xla_flags'], '--xla_gpu_deterministic_ops=true')

  def test_documented_comparison(self):
    doc = EVIDENCE['documented']
    c = compare.compare_runs(FP32, FP16)
    self.assertTrue(c['complete'] and c['inputs_sha256_checked'])
    self.assertEqual(c['passes_tolerance'], doc['passes_tolerance'])
    self.assertEqual(c['seeds_outside_seed_rules'], doc['seeds_outside_seed_rules'])
    t = c['overall']['all']['tolerance']
    self.assertEqual(t['seeds_with_new_merges'], doc['new_merges'])
    self.assertEqual(round(100 * t['mean_recall_drop'], 2), doc['mean_recall_drop_pts_2dp'])
    self.assertEqual(round(100 * t['mean_precision_drop'], 2), doc['mean_precision_drop_pts_2dp'])
    got = {x['name']: {'iou_4dp': round(x['mask_iou'], 4), 'steps_fp32': x['base_steps'],
                       'steps_fp16': x['cand_steps'], 'delta_recall_pts_2dp': round(100 * x['delta_recall'], 2),
                       'delta_precision_pts_2dp': round(100 * x['delta_precision'], 2)} for x in c['per_seed']}
    self.assertEqual(got, doc['per_seed'])
    rows = {x['name']: x for x in c['per_seed']}
    for name, m in doc['merges'].items():
      self.assertEqual({'fp32': rows[name]['base_merges'], 'fp16': rows[name]['cand_merges']}, m, name)
    a, b = compare.load_run(FP32), compare.load_run(FP16)
    self.assertEqual((round(_ms(a), 2), round(_ms(b), 2)), (doc['ms_per_step_fp32_2dp'], doc['ms_per_step_fp16_2dp']))
    self.assertEqual({'fp32': sum(r['canvas_inf_voxels'] for r in a['records']),
                      'fp16': sum(r['canvas_inf_voxels'] for r in b['records'])}, doc['canvas_inf_voxels_total'])


class ProbeEvidenceTest(unittest.TestCase):

  def test_profile(self):
    p = json.loads((RUNS / 'profile-fp16-20261021' / 'profile.json').read_text())
    doc = EVIDENCE['documented_profile']
    self.assertEqual(p['steps'], doc['steps'])
    self.assertEqual(round(p['ms_per_step'], 1), doc['ms_per_step_1dp'])
    self.assertEqual(round(p['counters']['executor-inference-time-ms'] / p['steps'], 1),
                     doc['executor_inference_ms_per_step_1dp'])

  def test_direct_call_probe(self):
    r = json.loads((RUNS / 'probe-direct-call-20261021' / 'result.json').read_text())
    doc = EVIDENCE['documented_probe']
    self.assertEqual(r['all_identical'], doc['all_identical'])
    self.assertTrue(all(v['canvas_identical'] and v['history_identical'] for v in r['per_seed'].values()))
    self.assertEqual(len(r['per_seed']), doc['seeds'])
    self.assertEqual((round(r['ms_per_step'], 2), round(r['reference_ms_per_step'], 2)),
                     (doc['ms_per_step_2dp'], doc['reference_ms_per_step_2dp']))


if __name__ == '__main__':
  unittest.main()
