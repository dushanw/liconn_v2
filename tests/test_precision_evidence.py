"""The recorded precision screen and benchmark runs (docs/precision_results.md) match their evidence.

The runs happened on Wahab's GPUs. This test checks, without a GPU, that the code
which ran is the checked code, that the outputs are the recorded ones, that every
screened trace matched the reference, that the selection follows the rule fixed in
advance, and that the documented comparisons and timings follow from the outputs.
"""

import hashlib
import json
from pathlib import Path
import unittest

from liconn_v2 import compare
from liconn_v2 import data
from liconn_v2 import precision

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'precision_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
REF = RUNS / EVIDENCE['reference_run']


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _ms_per_step(run) -> float:
  return 1000 * sum(r['trace_seconds'] for r in run['records']) / sum(r['steps'] for r in run['records'])


class PrecisionEvidenceTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.screen = json.loads((RUNS / 'precision-screen-20261021' / 'precision_screen.json').read_text())

  def test_code_that_ran_is_the_checked_code(self):
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_screen(self):
    s, doc = self.screen, EVIDENCE['documented_screen']
    self.assertEqual(s['backend'], 'gpu')
    self.assertEqual(len(s['checks']), 12)
    for name, c in s['checks'].items():
      self.assertTrue(all(c.values()), name)
    self.assertEqual(s['selected_for_benchmark'], precision.select_for_benchmark(s['variants']))
    self.assertEqual(s['selected_for_benchmark'], doc['selected'])
    self.assertEqual((s['device_kind'], s['inputs_total'], s['batch']),
                     (doc['device_kind'], doc['inputs_total'], doc['batch']))
    self.assertEqual(round(1000 * s['fp32_seconds_per_eval'], 2), doc['fp32_ms_per_eval_2dp'])
    got = {v['variant']: {'flip_rate_6dp': round(v['flip_rate'], 6),
                          'non_finite_rate_6dp': round(v['non_finite_rate'], 6),
                          'inputs_with_non_finite': v['inputs_with_non_finite'],
                          'ms_per_eval_2dp': round(1000 * v['seconds_per_eval'], 2)} for v in s['variants']}
    self.assertEqual(got, doc['variants'])

  def test_documented_runs(self):
    ref_ms = _ms_per_step(compare.load_run(REF))
    for name, doc in EVIDENCE['documented_runs'].items():
      run = compare.load_run(RUNS / name / 'merged')
      for r in run['records']:
        self.assertEqual((r['number_format'], r['backend']), (doc['variant'], 'gpu'))
        self.assertIn('V100', r['device_kind'])
      for p in run['platforms']:
        self.assertEqual(p['slurm_job'], str(EVIDENCE['slurm_jobs'][name]))
        self.assertEqual(p['xla_flags'], '--xla_gpu_deterministic_ops=true')
      self.assertEqual(round(_ms_per_step(run), 2), doc['ms_per_step_2dp'])
      self.assertEqual(sum(r['canvas_inf_voxels'] for r in run['records']), doc['canvas_inf_voxels_total'])
      # The re-run reproduces the first run bitwise, seed by seed.
      first = {r['name']: r for r in compare.load_run(RUNS / f'superseded-{name}' / 'merged')['records']}
      for r in run['records']:
        self.assertEqual((r['canvas_sha256'], r['history_sha256']),
                         (first[r['name']]['canvas_sha256'], first[r['name']]['history_sha256']), r['name'])
      self.assertEqual(round(ref_ms, 2), doc['reference_ms_per_step_2dp'])
      c = compare.compare_runs(REF, RUNS / name / 'merged')
      self.assertEqual(c['passes_tolerance'], doc['passes_tolerance'], name)
      self.assertEqual(c['seeds_outside_seed_rules'], doc['seeds_outside_seed_rules'], name)
      self.assertEqual(c['seeds_flagged_for_review'], doc['flagged_for_review'], name)
      got = {x['name']: {'iou_4dp': round(x['mask_iou'], 4), 'steps': x['cand_steps'],
                         'delta_recall_pts_2dp': round(100 * x['delta_recall'], 2),
                         'delta_precision_pts_2dp': round(100 * x['delta_precision'], 2),
                         'merges_cand': len(x['cand_merges'])} for x in c['per_seed']}
      self.assertEqual(got, doc['per_seed'], name)
      for g in ('all', 'without_sensitive_seeds'):
        o, d = c['overall'][g], doc[g]
        t = o['tolerance']
        self.assertEqual(o['cand_summary']['total_steps'], d['steps'])
        self.assertEqual(round(100 * t['mean_recall_drop'], 2), d['mean_recall_drop_pts_2dp'])
        self.assertEqual(round(100 * t['mean_precision_drop'], 2), d['mean_precision_drop_pts_2dp'])
        self.assertEqual(round(100 * t['mean_unlabeled_rise'], 2), d['mean_unlabeled_rise_pts_2dp'])
        self.assertEqual(round(100 * t['merged_voxels_growth'], 1), d['merged_voxels_growth_pct_1dp'])
        self.assertEqual(t['within_tolerance'], d['within_tolerance'])


class TrainingSurveyEvidenceTest(unittest.TestCase):

  def test_survey_numbers(self):
    s = json.loads((RUNS / 'training-survey-20261021' / 'training_data_survey.json').read_text())
    doc = EVIDENCE['documented_training_survey']
    self.assertEqual(round(s['expid82_proofread_coverage']['labeled_fraction'], 3), doc['labeled_fraction_3dp'])
    self.assertEqual(s['expid82_proofread_coverage']['objects'], doc['objects'])
    self.assertEqual(s['bucket_top_level'], doc['bucket_top_level'])
    self.assertEqual({k: {'finest_resolution_nm': v['finest_resolution_nm'], 'finest_size_xyz': v['finest_size_xyz']}
                      for k, v in s['volumes'].items()}, doc['volumes'])


if __name__ == '__main__':
  unittest.main()
