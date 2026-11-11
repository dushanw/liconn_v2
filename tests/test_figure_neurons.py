"""Whole test-crop neurons for the figure (liconn_v2/figure_neurons.py, docs/figure_neurons_design.md)."""

from pathlib import Path
import sys
import unittest

import numpy as np

from liconn_v2 import figure_neurons as fn
from liconn_v2 import unseen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'examples'))


def ids(n=5):
  return {k: [k * 1000 + i for i in range(n)] for k in unseen.CROPS}


class DrawTest(unittest.TestCase):

  def test_draw_is_fixed_and_uses_the_order_only(self):
    a = fn.draw(ids(), population=50)
    self.assertEqual(a, fn.draw(ids(), population=50))
    self.assertEqual(len(a), fn.DRAWN)
    self.assertEqual(len(set(a)), fn.DRAWN)
    flat = [(k, o) for k, v in ids().items() for o in v]
    picks = np.random.default_rng(20261111).choice(len(flat), 3, replace=False)
    self.assertEqual(a, [flat[int(i)] for i in picks])
    for k, o in a:
      self.assertIn(o, ids()[k])

  def test_draw_refuses_another_population(self):
    with self.assertRaises(ValueError):
      fn.draw(ids())  # 50 objects, not the test's 4,579
    short = ids()
    short[45] = short[45][:-1]
    with self.assertRaises(ValueError):
      fn.draw(short, population=50)
    dup = ids()
    dup[45] = dup[45][:-1] + dup[45][:1]
    with self.assertRaises(ValueError):
      fn.draw(dup, population=50)
    self.assertEqual(fn.POPULATION, 4579)

  def test_draw_on_the_test_counts(self):
    # Synthetic objects with the test's per-crop counts (crops 41-50; docs/figure_neurons_design.md): the draw
    # takes flat positions 4512, 2253 and 1002, i.e. the 432nd object of crop 50, the 437th of 45, the 70th of 43.
    counts = dict(zip(unseen.CROPS, (401, 532, 451, 433, 464, 446, 415, 496, 443, 498)))
    got = {k: [k * 10000 + i for i in range(n)] for k, n in counts.items()}
    self.assertEqual(fn.draw(got), [(50, 500431), (45, 450436), (43, 430069)])

  def test_settings(self):
    import time_crop11 as tc  # pylint: disable=g-import-not-at-top
    self.assertEqual(set(fn.TRACERS), set(unseen.B200_RUNS))
    for n in fn.TRACERS:
      self.assertEqual((*tc.RUNS[n], tc.SCALE.get(n, 1.0)), unseen.RUN_SETTINGS[n], n)

  def test_design_fixes_the_seed(self):
    self.assertIn(f'default_rng({fn.SEED})', (ROOT / 'docs' / 'figure_neurons_design.md').read_text())


class PartsTest(unittest.TestCase):

  def test_parts(self):
    gt = np.zeros((6, 6, 6), np.int32)
    gt[1:3, 1:3, 1:3] = 7   # the object
    gt[3:5, 1:3, 1:3] = 9   # a neighbour
    pred = np.zeros(gt.shape, bool)
    pred[2:5, 1:3, 1:3] = True  # half the object, all of the neighbour
    pred[2, 4, 4] = True        # one unlabelled voxel
    lo, vol = fn.parts(pred, gt, 7)
    self.assertEqual(lo, [1, 1, 1])
    full = np.zeros(gt.shape, np.uint8)
    full[tuple(slice(a, a + s) for a, s in zip(lo, vol.shape))] = vol
    self.assertTrue((full[2, 1:3, 1:3] == fn.PARTS['found']).all())
    self.assertTrue((full[1, 1:3, 1:3] == fn.PARTS['missed']).all())
    self.assertTrue((full[3:5, 1:3, 1:3] == fn.PARTS['other']).all())
    self.assertEqual(full[2, 4, 4], fn.PARTS['unlabelled'])
    self.assertEqual(int((full > 0).sum()), 8 + 8 + 1)

  def test_reconstruction_round_trip(self):
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 4, (8, 9, 10)).astype(np.int32)
    pred = rng.random(gt.shape) < 0.3
    lo, vol = fn.parts(pred, gt, 2)
    self.assertTrue(np.array_equal(fn.reconstruction(lo, vol, gt.shape), pred))
    self.assertEqual(fn.mask_sha256(fn.reconstruction(lo, vol, gt.shape)), fn.mask_sha256(pred))

  def test_label_pane_and_empty(self):
    gt = np.zeros((4, 4, 4), np.int32)
    gt[1, 2, 3] = 5
    lo, vol = fn.label_pane(gt, 5)
    self.assertEqual((lo, vol.shape, int(vol.sum())), ([1, 2, 3], (1, 1, 1), 1))
    with self.assertRaises(ValueError):
      fn.label_pane(gt, 6)


class CheckTest(unittest.TestCase):

  def test_trace_check(self):
    rec = {'object_id': 1, 'steps': 10, 'mask_sha256': 'm', 'history_sha256': 'h', 'f1': 0.9}
    fn.check_trace(dict(rec), rec)
    for key, value in (('steps', 11), ('mask_sha256', 'x'), ('history_sha256', 'x'), ('object_id', 2)):
      with self.assertRaises(ValueError, msg=key):
        fn.check_trace({**rec, key: value}, rec)

  def test_segneuron_check(self):
    row = {'object_id': 1, 'recall': 0.5, 'precision': 1.0, 'f1': 2 / 3}
    fn.check_segneuron(dict(row), row)
    for key in ('recall', 'precision', 'f1'):
      with self.assertRaises(ValueError, msg=key):
        fn.check_segneuron({**row, key: row[key] + 1e-12}, row)


  def test_score_check(self):
    rec = {'object_id': 1, 'recall': 0.5, 'precision': 0.25, 'merges': [[3, 1200]]}
    fn.check_score({'recall': 0.5, 'precision': 0.25 + 1e-13, 'merges': [(3, 1200)]}, rec)
    for change in ({'recall': 0.6}, {'precision': 0.3}, {'merges': []}):
      with self.assertRaises(ValueError, msg=change):
        fn.check_score({'recall': 0.5, 'precision': 0.25, 'merges': [[3, 1200]], **change}, rec)


class JobTest(unittest.TestCase):

  def test_job_script(self):
    s = (ROOT / 'cluster' / 'mit' / 'figure_neurons.sbatch').read_text()
    self.assertIn('examples/trace_figure_neurons.py', s)
    self.assertIn('test "$K" -ge 41 && test "$K" -le 50', s)
    self.assertIn('--test-dir "$ROOT/runs/unseen-20261104"', s)
    self.assertIn('sha256sum -c --quiet source_sha256.txt', s)
    self.assertIn('for m in F1 F5 S10 P2 Q2 segneuron; do run --methods "$m"; done', s)
    self.assertNotIn('rm ', s)

  def test_one_process_per_method_is_the_default(self):
    s = (ROOT / 'cluster' / 'mit' / 'figure_neurons.sbatch').read_text()
    self.assertIn('if [ "${ONE:-0}" = 1 ]; then\n  run\nelse\n'
                  '  for m in F1 F5 S10 P2 Q2 segneuron; do run --methods "$m"; done\nfi', s)
    self.assertIn('test ! -e "$OUT/crop-$K/figure.json"', s)
    self.assertNotIn('EACH', s)

  def test_report_written_after_each_tracer(self):
    src = (ROOT / 'examples' / 'trace_figure_neurons.py').read_text()
    loop = src[src.index('  for name in [m for m in fn.TRACERS'):src.index('    del net')]
    self.assertIn('write(folder, report)', loop)

  def test_rerun_after_a_partial_failure(self):
    import tempfile  # pylint: disable=g-import-not-at-top
    import trace_figure_neurons as tfn  # pylint: disable=g-import-not-at-top
    with tempfile.TemporaryDirectory() as d:
      d = Path(d)
      r = tfn.start_report(d, 43, [7], ['F1', 'F5', 'S10'])
      r['panes']['7'].update({'label': {}, 'F1': {'f1': 0.9}, 'F5': {'f1': 0.8}})
      r['failures'].append({'method': 'S10', 'object_id': 7, 'error': 'mask differs'})
      tfn.write(d, r)
      r = tfn.start_report(d, 43, [7], ['S10'])  # S10 re-run alone
      self.assertEqual(r['panes']['7'], {'label': {}, 'F1': {'f1': 0.9}, 'F5': {'f1': 0.8}})
      self.assertEqual(r['failures'], [])
      r['panes']['7']['S10'] = {'f1': 0.7}
      tfn.write(d, r)
      r = tfn.start_report(d, 43, [7], ['F5'])  # a re-run drops only that method's earlier pane
      self.assertEqual(sorted(r['panes']['7']), ['F1', 'S10', 'label'])
      with self.assertRaises(ValueError):
        tfn.start_report(d, 43, [8], ['F1'])  # another draw
      with self.assertRaises(ValueError):
        tfn.start_report(d, 44, [7], ['F1'])

  def test_write(self):
    import tempfile  # pylint: disable=g-import-not-at-top
    import json  # pylint: disable=g-import-not-at-top
    import trace_figure_neurons as tfn  # pylint: disable=g-import-not-at-top
    with tempfile.TemporaryDirectory() as d:
      d = Path(d)
      tfn.write(d, {'crop': 43, 'panes': {'1': {'F1': {'f1': 0.5}}}})
      tfn.write(d, {'crop': 43, 'panes': {'1': {'F1': {'f1': 0.5}, 'F5': {'f1': 0.4}}}})
      self.assertEqual(json.loads((d / 'figure.json').read_text())['panes']['1']['F5'], {'f1': 0.4})
      self.assertEqual(sorted(x.name for x in d.iterdir()), ['figure.json'])


if __name__ == '__main__':
  unittest.main()
