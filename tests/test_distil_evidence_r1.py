"""The distillation round-0 diagnostic and on-policy round 1 (docs/distil_results.md) match their evidence.

docs/distil_evidence_r1.json holds verbatim extracts of the MIT run records, each with the sha256 of its
source file. This test checks every source file against its sha256 in the data cache
(runs/distil-20261031), every extract against its source (dictionaries field by field, lists item by item,
per-object rows to the 5 decimals the extract states), the on-policy collection against the per-crop
lines of its job logs, and the numbers docs/distil_results.md states from these records.
"""

import json
from pathlib import Path
import unittest

from liconn_v2 import data

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'distil_evidence_r1.json').read_text())
RESULTS = (ROOT / 'docs' / 'distil_results.md').read_text()
RUN = data.cache_dir() / 'runs' / 'distil-20261031'
TRACES = ('trace-dilated', 'trace-dilated-r1', 'trace-dilated-r1-last', 'trace-undilated')


def extract_of(sub, src, where=''):
  """sub is an extract of src: every key of a dictionary and every item of a list equals its source."""
  if isinstance(sub, dict):
    assert isinstance(src, dict), where
    for k, v in sub.items():
      assert k in src, f'{where}.{k} not in the source'
      extract_of(v, src[k], f'{where}.{k}')
  elif isinstance(sub, list):
    assert isinstance(src, list) and len(sub) == len(src), where
    for i, (a, b) in enumerate(zip(sub, src)):
      extract_of(a, b, f'{where}[{i}]')
  else:
    assert sub == src, f'{where}: {sub} != {src}'


def source(entry: dict):
  return json.loads((RUN / entry['file']).read_text())


class SourcesTest(unittest.TestCase):

  def test_every_source_file_has_its_sha256(self):
    entries = [v for v in EVIDENCE.values() if isinstance(v, dict) and 'file' in v]
    entries += EVIDENCE['job_logs'] + EVIDENCE['onpolicy-collection']['files']
    self.assertGreaterEqual(len(entries), 15)
    for e in entries:
      self.assertEqual(data.sha256_file(RUN / e['file']), e['sha256'], e['file'])


class ExtractsTest(unittest.TestCase):

  def test_trace_summaries(self):
    for k in TRACES:
      self.assertEqual(EVIDENCE[k]['summary'], source(EVIDENCE[k])['summary'], k)

  def test_per_object_rows(self):
    for k in TRACES:
      if 'per_object' not in EVIDENCE[k]:
        continue
      cols = EVIDENCE[k]['per_object']['columns']
      self.assertEqual(cols, ['object_id', 'precision', 'recall', 'teacher_f1', 'steps', 'teacher_steps'])
      records = source(EVIDENCE[k])['records']
      rows = EVIDENCE[k]['per_object']['rows']
      self.assertEqual(len(rows), len(records), k)
      for row, r in zip(rows, records):
        expected = [r[c] if isinstance(r[c], int) else round(r[c], 5) for c in cols]
        self.assertEqual(row, expected, (k, r['object_id']))

  def test_training_logs_and_move_diagnostic(self):
    for k in ('train-dilated', 'train-dilated-r1', 'moves-r0-teacher11'):
      sub = {f: v for f, v in EVIDENCE[k].items() if f not in ('file', 'sha256')}
      extract_of(sub, source(EVIDENCE[k]), k)

  def test_best_checkpoints_are_the_lowest_validation_loss(self):
    """Round 0 keeps the lowest validation loss; in round 1 its starting checkpoint (step 0) cannot be the best."""
    for k, skip_step_0 in (('train-dilated', False), ('train-dilated-r1', True)):
      log = source(EVIDENCE[k])
      valid = [v for v in log['valid'] if not (skip_step_0 and v['step'] == 0)]
      lowest = min(valid, key=lambda v: v['loss'])
      self.assertEqual((EVIDENCE[k]['best']['step'], EVIDENCE[k]['best']['loss']), (lowest['step'], lowest['loss']), k)

  def test_onpolicy_collection_is_the_job_logs(self):
    logged = []
    for f in EVIDENCE['onpolicy-collection']['files']:
      for line in (RUN / f['file']).read_text().splitlines():
        if line.startswith('{"crop"') and '"traces"' in line:
          logged.append(json.loads(line))
    per_crop = EVIDENCE['onpolicy-collection']['per_crop']
    self.assertEqual([c['crop'] for c in per_crop], list(range(11)))
    for c in per_crop:
      self.assertIn(c, logged, c['crop'])
    self.assertEqual(sum(c['samples'] for c in per_crop), EVIDENCE['train-dilated-r1']['extra_samples'])


class ResultsTest(unittest.TestCase):
  """Numbers docs/distil_results.md states from these records, formatted from the evidence."""

  def assertStated(self, text):
    self.assertIn(text, RESULTS)

  def test_round_0_and_round_1(self):
    f1 = {k: EVIDENCE[k]['summary']['student_mean_f1'] for k in TRACES}
    teacher = EVIDENCE['trace-dilated']['summary']['teacher_mean_f1']
    self.assertStated(f"the dilated decoder reached {f1['trace-dilated']:.3f}, the undilated one "
                      f"{f1['trace-undilated']:.3f}")
    self.assertStated(f"crop-11 mean F1 {f1['trace-dilated-r1']:.3f} against {teacher:.3f} for the teacher")
    t = EVIDENCE['train-dilated-r1']
    self.assertStated(f"{t['extra_samples']:,} new samples")
    self.assertStated(f"{t['extra_samples']:,} on-policy), {t['total_seconds'] / 3600:.1f} h")
    self.assertStated(f"{t['valid_samples']:,} crop-11 teacher samples")
    self.assertStated(f"The validation loss hardly moves: {t['valid'][0]['loss']:.4f} at the start, best "
                      f"{t['best']['loss']:.4f} at step {t['best']['step']:,}")
    best0 = EVIDENCE['train-dilated']['best']
    self.assertStated(f"step {best0['step']:,}, loss {best0['loss']:.4f}")


if __name__ == '__main__':
  unittest.main()
