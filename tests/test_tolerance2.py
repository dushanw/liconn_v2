"""The owner's 2% rule (liconn_v2/tolerance2.py, docs/tolerance2_results.md)."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from liconn_v2 import data
from liconn_v2 import manyseed
from liconn_v2 import tolerance2

ROOT = Path(__file__).resolve().parent.parent
REPORT = data.cache_dir() / 'runs' / 'tolerance2-report-20261023' / 'tolerance2_report.json'
REPORT_SHA256 = '9298dfef543104da9601fdffbf0efffe35b452b395328f70669ed5c843288553'


def verdict(merges, stops, **rules):
  base = {'complete_same_inputs': True, 'all_300_seeds': True, 'new_merges': merges <= 3,
          'new_early_stops': stops <= 3, 'mean_recall_drop': True, 'mean_precision_drop': True}
  base.update(rules)
  return {'counts': {'new_merge': merges, 'new_early_stop': stops, 'avoided_merge': 0, 'extension': 0},
          'rules': base, 'passes': all(base.values()), 'mean_recall_change': 0.0, 'mean_precision_change': 0.0}


class RuleTest(unittest.TestCase):

  def test_six_of_300_pass_seven_fail(self):
    self.assertEqual((tolerance2.MAX_NEW_MERGES, tolerance2.MAX_NEW_EARLY_STOPS), (6, 6))
    self.assertTrue(tolerance2.apply(verdict(6, 6))['passes'])
    self.assertFalse(tolerance2.apply(verdict(7, 0))['passes'])
    self.assertFalse(tolerance2.apply(verdict(0, 7))['passes'])
    self.assertFalse(verdict(5, 3)['passes'])  # the original rule
    self.assertTrue(tolerance2.apply(verdict(5, 3))['passes'])

  def test_other_rules_are_unchanged(self):
    for rule in ('complete_same_inputs', 'all_300_seeds', 'mean_recall_drop', 'mean_precision_drop'):
      self.assertFalse(tolerance2.apply(verdict(0, 0, **{rule: False}))['passes'], rule)

  def test_original_rule_code_is_unchanged(self):
    self.assertEqual((manyseed.MAX_NEW_MERGES, manyseed.MAX_NEW_EARLY_STOPS), (3, 3))


class RecordedTest(unittest.TestCase):

  def test_recorded_report_is_what_the_code_computes(self):
    self.assertEqual(hashlib.sha256(REPORT.read_bytes()).hexdigest(), REPORT_SHA256)
    with tempfile.TemporaryDirectory() as d:
      subprocess.run([sys.executable, str(ROOT / 'examples' / 'judge_tolerance2.py'), '--recorded', '--out', d],
                     check=True, capture_output=True)
      rep = json.loads((Path(d) / 'tolerance2_report.json').read_text())
    self.assertEqual(rep, json.loads(REPORT.read_text()))
    got = {k: (c['counts']['new_merge'], c['counts']['new_early_stop'], c['original_rule']['passes'],
               c['rule_2pct']['passes']) for k, c in rep['cases'].items()}
    self.assertEqual(got, {'first300_fp16': (3, 10, False, False), 'first300_fp16_head32': (1, 1, True, True),
                           'fresh_fp16_head32': (5, 3, False, True)})
    means = {k: (round(100 * c['rule_2pct']['mean_recall_change'], 2), round(100 * c['rule_2pct']['mean_precision_change'], 2))
             for k, c in rep['cases'].items()}
    self.assertEqual(means, {'first300_fp16': (-0.11, -0.31), 'first300_fp16_head32': (0.27, 0.13),
                             'fresh_fp16_head32': (-0.16, -0.48)})


if __name__ == '__main__':
  unittest.main()
