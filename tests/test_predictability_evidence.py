"""The recorded predictability run (docs/predictability_results.md) matches its evidence.

The traces ran on Wahab's GPU. This test checks, without a GPU, that the code that
produced them is unchanged, that the outputs are the recorded ones, that every trace
matched the adopted reference, and that the documented numbers follow from the
recorded per-step data with the current analysis code.
"""

import hashlib
import json
from pathlib import Path
import unittest
import warnings

from liconn_v2 import data
from liconn_v2 import predict

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'predictability_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
RUN = RUNS / 'predictability-20261019'


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _r3(x):
  return None if x is None else round(x, 3)


class PredictabilityEvidenceTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    import numpy as np  # pylint: disable=g-import-not-at-top
    with np.load(RUN / 'predictability_steps.npz') as f:
      names = sorted(k[len('new_fg_'):] for k in f.files if k.startswith('new_fg_'))
      per_seed = {n: {k[:-len(n) - 1]: f[k] for k in f.files if k.endswith('_' + n)} for n in names}
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)  # spurious Accelerate matmul warnings
      cls.analysis = predict.analyse(predict.step_table(per_seed))
    cls.recorded = json.loads((RUN / 'predictability.json').read_text())

  def test_trace_code_is_unchanged(self):
    for rel, expected in EVIDENCE['trace_source_sha256'].items():
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_every_trace_matched_the_reference(self):
    self.assertEqual(self.recorded['backend'], 'gpu')
    self.assertTrue(self.recorded['all_checks_pass'])
    self.assertEqual(len(self.recorded['checks']), 11)
    for name, c in self.recorded['checks'].items():
      self.assertTrue(c.get('reference_record_present', True), name)
      self.assertTrue(c['canvas_matches_reference'], name)
      self.assertTrue(c['history_matches_reference'], name)
    self.assertTrue(self.analysis['queue_score_matched_popped_entry'])

  def test_current_analysis_reproduces_the_run(self):
    for t in predict.TARGETS:
      self.assertEqual(self.analysis[t]['model_oof_auc'], self.recorded['analysis'][t]['model_oof_auc'], t)
      self.assertEqual(self.analysis[t]['feature_auc'], self.recorded['analysis'][t]['feature_auc'], t)

  def test_documented_numbers(self):
    for scope, a in (('all_steps', self.analysis), ('near_enclosed', self.analysis['near_enclosed_subset'])):
      doc = EVIDENCE['documented'][scope]
      self.assertEqual(a['steps'], doc['steps'], scope)
      for t in predict.TARGETS:
        x, d = a[t], doc[t]
        self.assertEqual(_r3(x['prevalence']), d['prevalence'], (scope, t))
        self.assertEqual(_r3(x['model_oof_auc']), d['model_oof_auc'], (scope, t))
        self.assertEqual([_r3(v) for v in x['model_per_seed_auc_quartiles']], d['per_seed_auc_quartiles'])
        self.assertEqual({k: _r3(v) for k, v in x['feature_auc_oriented'].items()}, d['feature_auc_oriented'])
        self.assertEqual({k: {kk: _r3(vv) for kk, vv in v.items()} for k, v in x['feature_set_oof_auc'].items()},
                         d['feature_set_oof_auc'], (scope, t))


if __name__ == '__main__':
  unittest.main()
