"""fp16_head32 on the 300-seed rule (docs/head32_design.md) and the milestone-11 fixes."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from liconn_v2 import earlystop

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'examples'))
import analyze_earlystop  # pylint: disable=g-import-not-at-top,wrong-import-position
import judge_manyseed  # pylint: disable=g-import-not-at-top,wrong-import-position
import run_manyseed  # pylint: disable=g-import-not-at-top,wrong-import-position


class VoxelFlipTest(unittest.TestCase):

  def test_non_finite_outputs_count_as_changed(self):
    ts = float(earlystop.segment_threshold())
    ref = np.array([ts - 1, ts - 1, ts + 1, ts + 1], np.float32)
    y = np.array([np.nan, ts - 1, np.inf, ts - 2], np.float32)
    # NaN against background, +inf (no decision) against foreground, and a real flip
    self.assertEqual(earlystop.voxel_flips(y, ref), 3)
    self.assertEqual(earlystop.voxel_flips(ref, ref), 0)


class AnalysisRejectsInvalidRunsTest(unittest.TestCase):

  def _run(self, root: Path, variant, reproduces=True, mismatches=0):
    names = list(earlystop.EARLY_STOP_SEEDS) + [f'm0_axon_{i}' for i in range(10)]
    (root / 'steps').mkdir(parents=True)
    records = []
    for n in names:
      earlystop.save_steps(root / 'steps' / f'{n}.json', [])
      r = {'name': n, 'variant': variant, 'reproduces_manyseed': reproduces}
      if variant == 'fp32':
        r['emulation_mismatches'] = mismatches
      records.append(r)
    (root / 'earlystop_run.json').write_text(json.dumps(
        {'variant': variant, 'seeds': names, 'records': records,
         'seconds_per_eval': {'fp32': 1.0, 'fp16': 0.3, 'fp16_head32': 0.3, 'fp16_res32': 0.3}}))

  def test_failed_reproduction_or_emulation_stops_the_analysis(self):
    for kwargs32, kwargs16 in (({}, {'reproduces': False}), ({'mismatches': 2}, {})):
      with tempfile.TemporaryDirectory() as d:
        self._run(Path(d) / 'a', 'fp32', **kwargs32)
        self._run(Path(d) / 'b', 'fp16', **kwargs16)
        with self.assertRaisesRegex(SystemExit, 'rejected'):
          analyze_earlystop.analyze(Path(d) / 'a', Path(d) / 'b', None)


class RecordParityTest(unittest.TestCase):

  def test_run_layout_writes_the_same_records_as_run_variant(self):
    from unittest import mock  # pylint: disable=g-import-not-at-top
    from liconn_v2 import precision  # pylint: disable=g-import-not-at-top
    from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
    canvas = np.full((40, 40, 40), np.nan, np.float32)
    canvas[18:23, 18:23, 18:23] = 2.0
    gt = np.zeros((40, 40, 40), np.uint32)
    gt[17:24, 17:24, 17:24] = 7

    class FakeTracer:
      params = None
      def __enter__(self):
        return self
      def __exit__(self, *a):
        return False
      def trace(self, raw, seed):
        return reference.Trace(canvas, [(20, 20, 20)], 1, 0.0, 0.5, {}, 'gpu', 'Tesla V100-SXM2-16GB')

    seeds = [{'name': 's', 'class': 'axon', 'seed_zyx': (20, 20, 20)}]
    with mock.patch.object(reference, 'Tracer', lambda path: FakeTracer()), \
         mock.patch.object(reference, 'gflop_per_eval', lambda p: 81.6), \
         mock.patch.object(precision, 'install', lambda t, v: None), \
         mock.patch.object(earlystop, 'install', lambda t, v: None):
      a, ma = precision.run_variant(seeds, None, gt, 'ckpt', 'fp16')
      b, mb = earlystop.run_layout(seeds, None, gt, 'ckpt', 'fp16_head32')
    self.assertEqual(list(a[0]), list(b[0]))
    self.assertEqual({**a[0], 'number_format': None}, {**b[0], 'number_format': None})
    self.assertEqual((a[0]['number_format'], b[0]['number_format']), ('fp16', 'fp16_head32'))
    np.testing.assert_array_equal(ma['s'], mb['s'])


class CandidatePathTest(unittest.TestCase):

  def test_runner_and_judge_know_the_candidate(self):
    self.assertIn('fp16_head32', run_manyseed.VARIANTS)
    self.assertIn('fp16_head32', earlystop.LAYOUTS)
    self.assertEqual(judge_manyseed.RECORDED['fp32'], 'manyseed-fp32-20261021')
    self.assertIn('fp16_head32', judge_manyseed.RECORDED)
    with self.assertRaises(ValueError):
      earlystop.install(None, 'fp16')  # plain precision variants go through precision.install

  def test_runner_rejects_unknown_variants(self):
    r = subprocess.run([sys.executable, str(ROOT / 'examples' / 'run_manyseed.py'), '--variant', 'fp8', '--crop', '0'],
                       capture_output=True, text=True)
    self.assertNotEqual(r.returncode, 0)
    self.assertIn('invalid choice', r.stderr)


if __name__ == '__main__':
  unittest.main()
