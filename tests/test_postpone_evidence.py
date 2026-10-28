"""The recorded policy runs (docs/postpone_results.md) match their evidence.

The runs happened on Wahab's GPUs. This test checks, without a GPU, that the code
which ran is the checked code, that the outputs are the recorded ones, that the
default-policy control reproduces the reference exactly, that every policy's
bookkeeping adds up, and that the documented comparisons follow from the outputs.
"""

import hashlib
import json
from pathlib import Path
import unittest

from liconn_v2 import compare
from liconn_v2 import data
from liconn_v2 import policies

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'postpone_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
REF = RUNS / EVIDENCE['reference_run']


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(policy):
  return RUNS / f'policy-{policy}-20261019' / 'merged'


class PostponeEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_runs_record_policy_platform_and_jobs(self):
    for policy, job in EVIDENCE['slurm_jobs'].items():
      run = compare.load_run(_run(policy))
      self.assertEqual(len(run['records']), 12, policy)
      for r in run['records']:
        self.assertEqual(r['policy'], policy)
        self.assertEqual(r['backend'], 'gpu')
        if policy != 'default':
          self.assertEqual(policies.check_stats(r['policy_stats'], r['steps']), [], (policy, r['name']))
      for p in run['platforms']:
        self.assertEqual(p['slurm_job'], str(job))
        self.assertEqual(p['xla_flags'], '--xla_gpu_deterministic_ops=true')

  def test_default_policy_reproduces_the_reference(self):
    r = compare.compare_runs(REF, _run('default'))
    self.assertTrue(r['passes_tolerance'])
    for row in r['per_seed']:
      self.assertTrue(row['same_canvas'] and row['same_history'], row['name'])

  def test_documented_comparisons(self):
    for policy, doc in EVIDENCE['documented'].items():
      r = compare.compare_runs(REF, _run(policy))
      self.assertEqual(r['passes_tolerance'], doc['passes_tolerance'], policy)
      self.assertEqual(r['seeds_outside_seed_rules'], doc['seeds_outside_seed_rules'], policy)
      got = {x['name']: {'iou_4dp': round(x['mask_iou'], 4), 'steps': x['cand_steps'],
                         'delta_recall_4dp': round(x['delta_recall'], 4)} for x in r['per_seed']}
      self.assertEqual(got, doc['per_seed'], policy)
      for g in ('all', 'without_sensitive_seeds'):
        o = r['overall'][g]
        self.assertEqual(o['cand_summary']['total_steps'], doc[g]['steps'], (policy, g))
        self.assertEqual(round(o['cand_summary']['evals_per_1000_correct_voxels'], 4),
                         doc[g]['evals_per_1000_4dp'], (policy, g))
        self.assertEqual(o['tolerance']['within_tolerance'], doc[g]['within_tolerance'], (policy, g))
        t = o['tolerance']
        self.assertEqual(round(100 * t['mean_recall_drop'], 2), doc[g]['mean_recall_drop_pts_2dp'])
        self.assertEqual(round(100 * t['mean_precision_drop'], 2), doc[g]['mean_precision_drop_pts_2dp'])
        self.assertEqual(round(100 * t['mean_unlabeled_rise'], 2), doc[g]['mean_unlabeled_rise_pts_2dp'])
        self.assertEqual(round(100 * t['merged_voxels_growth'], 1), doc[g]['merged_voxels_growth_pct_1dp'])
      self.assertEqual(r['seeds_flagged_for_review'], doc['flagged_for_review'], policy)
      row = {x['name']: x for x in r['per_seed']}['dendrite_1525']
      self.assertEqual({'merges_ref_cand': [len(row['base_merges']), len(row['cand_merges'])],
                        'delta_precision_pts_1dp': round(100 * row['delta_precision'], 1),
                        'delta_recall_pts_2dp': round(100 * row['delta_recall'], 2)}, doc['dendrite_1525'])
      if policy != 'default':
        total = {}
        for rec in compare.load_run(_run(policy))['records']:
          for k, v in rec['policy_stats'].items():
            total[k] = total.get(k, 0) + v
        self.assertEqual(total, doc['policy_stats_total'], policy)


if __name__ == '__main__':
  unittest.main()
