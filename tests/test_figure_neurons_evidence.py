"""The figure-neurons run's evidence (docs/figure_neurons_evidence/) against the design note's Run section.

The files are the run's crop-K/figure.json, byte for byte, and the final lines of its job logs; the B200
job (examples/trace_figure_neurons.py) wrote a pane only after the re-run matched the test's record.
"""

import hashlib
import json
from pathlib import Path
import re
import unittest

from liconn_v2 import figure_neurons as fn

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / 'docs' / 'figure_neurons_evidence'
DESIGN = (ROOT / 'docs' / 'figure_neurons_design.md').read_text()
DRAWN = [(50, 68079), (45, 353342), (43, 343390)]  # the pinned draw (tests/test_figure_neurons.py)


def table():
  """crop -> (object, sha256, failures, F1 values) from the design note's Run table."""
  rows = re.findall(r'^\| (\d+) \| (\d+) \| `([0-9a-f]{64})` \| (\w+) \| ([0-9., ]+) \|$', DESIGN, re.M)
  return {int(k): (int(o), h, f, [float(x) for x in v.split(',')]) for k, o, h, f, v in rows}


def report(k):
  return json.loads((EV / f'crop-{k}.json').read_text())


class EvidenceTest(unittest.TestCase):

  def test_files_are_those_in_the_design(self):
    t = table()
    self.assertEqual(sorted(t), sorted(k for k, _ in DRAWN))
    for k, (o, sha, failures, _) in t.items():
      self.assertEqual(hashlib.sha256((EV / f'crop-{k}.json').read_bytes()).hexdigest(), sha, k)
      self.assertEqual((k, o), next(d for d in DRAWN if d[0] == k))
      self.assertEqual(failures, 'none')

  def test_every_method_verified(self):
    for k, o in DRAWN:
      r = report(k)
      self.assertEqual((r['crop'], r['drawn'], r['failures']), (k, [o], []), k)
      self.assertEqual(set(r['panes']), {str(o)})
      self.assertEqual(set(r['panes'][str(o)]), {'label', *fn.METHODS}, k)
      for m in fn.METHODS:
        st = r['panes'][str(o)][m]
        self.assertEqual(st['file'], f'{o}_{m}.npz')
        for key in ('f1', 'recall', 'precision'):
          self.assertTrue(0 <= st[key] <= 1, (k, m, key))
        self.assertEqual(st['merges'], 0)  # the legend: no merge in any pane
        keys = {'f1', 'recall', 'precision', 'merges', 'file'} | ({'steps'} if m in fn.TRACERS else set())
        self.assertEqual(set(st), keys, (k, m))
        if m in fn.TRACERS:
          self.assertGreater(st['steps'], 0)

  def test_design_f1_values(self):
    for k, (o, _, _, f1) in table().items():
      panes = report(k)['panes'][str(o)]
      self.assertEqual(f1, [round(panes[m]['f1'], 3) for m in fn.METHODS], k)

  def test_logs_agree(self):
    lines = dict(l.split(': ', 1) for l in (EV / 'logs.txt').read_text().splitlines() if l.startswith('liconn-'))
    self.assertEqual(sorted(lines), ['liconn-figneurons-28663157_43', 'liconn-figneurons-28663157_45',
                                     'liconn-figneurons-28663157_50', 'liconn-figneurons-28668556_43'])
    self.assertIn('OutOfMemoryError', lines['liconn-figneurons-28663157_43'])  # the first crop-43 job
    for job, k in (('28663157_45', 45), ('28663157_50', 50), ('28668556_43', 43)):
      last = json.loads(lines[f'liconn-figneurons-{job}'])
      r = report(k)
      self.assertEqual((last['crop'], last['drawn'], last['failures']), (k, r['drawn'], []))
      if 'panes' in last:
        self.assertEqual({o: sorted(p) for o, p in last['panes'].items()}, {o: sorted(p) for o, p in r['panes'].items()})
    for job in ('28663157', '28668556'):
      self.assertIn(job, DESIGN)

  def test_source_of_each_job(self):
    src = dict(re.findall(r'^source-(\d+_\d+)\.txt sha256 ([0-9a-f]{64})$', (EV / 'logs.txt').read_text(), re.M))
    self.assertEqual(sorted(src), ['28663157_43', '28663157_45', '28663157_50', '28668556_43'])
    self.assertEqual(len({src[f'28663157_{k}'] for k in (43, 45, 50)}), 1)  # one staged source for the array
    self.assertNotEqual(src['28668556_43'], src['28663157_43'])  # re-staged with --methods before the re-run


if __name__ == '__main__':
  unittest.main()
