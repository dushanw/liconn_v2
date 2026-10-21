"""The recorded block-drop screen and variant runs (docs/block_drop_results.md) match their evidence.

The runs happened on Wahab's GPUs. This test checks, without a GPU, that the code
which ran is the checked code, that the outputs are the recorded ones, that every
screened trace matched the reference, that the selection follows the pre-registered
rule, and that the documented comparisons follow from the outputs.
"""

import hashlib
import json
from pathlib import Path
import unittest

from liconn_v2 import compare
from liconn_v2 import data
from liconn_v2 import pruned

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'block_drop_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
REF = RUNS / EVIDENCE['reference_run']


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _executable_sha(text: str) -> str:
  """sha256 of a shell script without its plain comment lines (keeps #! and #SBATCH)."""
  lines = [l for l in text.splitlines()
           if not (l.startswith('#') and not l.startswith(('#!', '#SBATCH')))]
  return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


class BlockDropEvidenceTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.screen = json.loads((RUNS / 'block-screen-20261020' / 'block_screen.json').read_text())

  def test_code_that_ran_is_the_checked_code(self):
    edits = EVIDENCE.get('post_run_comment_edits', {})
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      if rel in edits:
        # Only comment lines changed after the run: compare everything else.
        self.assertEqual(_executable_sha((ROOT / rel).read_text()),
                         edits[rel]['executable_lines_sha256'], rel)
      else:
        self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_screen_traces_matched_the_reference(self):
    self.assertEqual(self.screen['backend'], 'gpu')
    self.assertEqual(len(self.screen['checks']), 12)
    for name, c in self.screen['checks'].items():
      self.assertTrue(all(c.values()), name)

  def test_selection_follows_the_rule(self):
    chosen = pruned.select_variants(self.screen['single_drops'], self.screen['greedy_path'])
    self.assertEqual([{'label': l, 'skip': list(s)} for l, s in chosen], self.screen['selected_for_benchmark'])
    doc = EVIDENCE['documented_screen']
    self.assertEqual(self.screen['selected_for_benchmark'], doc['selected'])
    self.assertEqual(self.screen['inputs_total'], doc['inputs_total'])
    self.assertEqual({str(s['skip'][0]): round(s['flip_rate'], 4) for s in self.screen['single_drops']},
                     doc['single_flip_rate_4dp'])
    self.assertEqual([{'skip': g['skip'], 'flip_rate_4dp': round(g['flip_rate'], 4),
                       'gflop_per_eval_1dp': round(g['gflop_per_eval'], 1)} for g in self.screen['greedy_path']],
                     doc['greedy_path'])
    # Each greedy step adds exactly one block to the previous set.
    for a, b in zip(self.screen['greedy_path'], self.screen['greedy_path'][1:]):
      self.assertEqual(len(set(b['skip']) - set(a['skip'])), 1)
      self.assertTrue(set(a['skip']) < set(b['skip']))

  def test_documented_comparisons(self):
    for name, doc in EVIDENCE['documented_runs'].items():
      run = compare.load_run(RUNS / name / 'merged')
      for r in run['records']:
        self.assertEqual(r['skip_blocks'], doc['skip'])
        self.assertEqual(r['backend'], 'gpu')
        self.assertAlmostEqual(r['gflop_per_eval'], 81.600564672 - len(doc["skip"]) * 3.974344704, places=6)
      for p in run['platforms']:
        self.assertEqual(p['slurm_job'], str(EVIDENCE['slurm_jobs'][name]))
      c = compare.compare_runs(REF, RUNS / name / 'merged')
      self.assertEqual(c['passes_tolerance'], doc['passes_tolerance'], name)
      self.assertEqual(c['seeds_outside_seed_rules'], doc['seeds_outside_seed_rules'], name)
      got = {x['name']: {'iou_4dp': round(x['mask_iou'], 4), 'steps': x['cand_steps'],
                         'delta_recall_pts_2dp': round(100 * x['delta_recall'], 2),
                         'merges_cand': len(x['cand_merges'])} for x in c['per_seed']}
      self.assertEqual(got, doc['per_seed'], name)
      for g in ('all', 'without_sensitive_seeds'):
        o, d = c['overall'][g], doc[g]
        t = o['tolerance']
        self.assertEqual(o['cand_summary']['total_steps'], d['steps'])
        self.assertEqual(round(o['cand_summary']['gflop_per_correct_voxel'], 4), d['gflop_per_correct_voxel_4dp'])
        self.assertEqual(round(o['base_summary']['gflop_per_correct_voxel'], 4), d['base_gflop_per_correct_voxel_4dp'])
        self.assertEqual(round(100 * t['mean_recall_drop'], 2), d['mean_recall_drop_pts_2dp'])
        self.assertEqual(round(100 * t['mean_precision_drop'], 2), d['mean_precision_drop_pts_2dp'])
        self.assertEqual(round(100 * t['merged_voxels_growth'], 1), d['merged_voxels_growth_pct_1dp'])
        self.assertEqual(t['within_tolerance'], d['within_tolerance'])


if __name__ == '__main__':
  unittest.main()
