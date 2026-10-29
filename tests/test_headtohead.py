"""Tests for liconn_v2.headtohead (docs/headtohead_design.md)."""

import json
from pathlib import Path
import unittest

from liconn_v2 import headtohead as hh
from liconn_v2 import vitseg

ROOT = Path(__file__).resolve().parent.parent


def row(i, p, r, merges=(), steps=10, trace=1.0, wall=1.5, lost=False):
  return {'object_id': i, 'precision': p, 'recall': r, 'f1': hh.f1(p, r), 'merges': [list(m) for m in merges],
          'merged_voxels': sum(m[1] for m in merges), 'steps': steps, 'trace_seconds': trace, 'wall_seconds': wall,
          'seed_in_background': lost}


class HeadToHeadTest(unittest.TestCase):

  def test_settings_are_the_recorded_ones(self):
    ev = json.loads((ROOT / 'docs' / 'segneuron_finetune_evidence.json').read_text())
    self.assertEqual(hh.THRESHOLD, ev['ours']['threshold'])
    self.assertEqual(hh.CROP, vitseg.TUNE_CROP)
    self.assertEqual(hh.FFN_VARIANT, 'fp16_head32')

  def test_f1(self):
    self.assertEqual(hh.f1(0.0, 0.0), 0.0)
    self.assertAlmostEqual(hh.f1(1.0, 0.5), 2 / 3)

  def test_paired(self):
    ffn = [row(1, 1.0, 0.5, merges=[(9, 2000)]), row(2, 0.5, 0.5), row(3, 1.0, 1.0)]
    sn = [row(1, 0.0, 0.0, lost=True), row(2, 1.0, 1.0, merges=[(8, 1500), (7, 1000)]), row(3, 1.0, 1.0)]
    p = hh.paired(ffn, sn)
    self.assertEqual((p['ffn_higher'], p['segneuron_higher'], p['equal']), (1, 1, 1))
    self.assertAlmostEqual(p['ffn']['f1'], (2 / 3 + 0.5 + 1) / 3)
    self.assertAlmostEqual(p['mean_f1_difference'], (2 / 3 - 0.5) / 3)
    self.assertEqual(p['f1_difference_quantiles']['value'][3], 0.0)  # the median
    self.assertAlmostEqual(p['f1_difference_quantiles']['value'][0], -0.5)
    self.assertAlmostEqual(p['f1_difference_quantiles']['value'][-1], 2 / 3)
    self.assertEqual((p['ffn_merges'], p['ffn_objects_with_merge'], p['ffn_merged_voxels']), (1, 1, 2000))
    self.assertEqual((p['segneuron_merges'], p['segneuron_objects_with_merge'], p['segneuron_merged_voxels']),
                     (2, 1, 2500))
    self.assertEqual(p['segneuron_seeds_in_background'], 1)
    self.assertAlmostEqual(p['segneuron_mean_f1_where_seed_found'], 1.0)
    self.assertAlmostEqual(p['ffn_mean_f1_where_segneuron_seed_found'], 0.75)
    with self.assertRaises(ValueError):
      hh.paired(ffn, sn[::-1])

  def test_speed(self):
    ffn = [row(1, 1, 1, steps=100, trace=0.5, wall=0.6), row(2, 1, 1, steps=300, trace=1.5, wall=1.8)]
    s = hh.speed(ffn, ffn_setup_seconds=30.0, inference_wall_seconds=200.0, inference_seconds=100.0,
                 post_seconds=40.0)
    self.assertAlmostEqual(s['ffn']['wall_seconds'], 2.4)
    self.assertAlmostEqual(s['ffn']['ms_per_step'], 5.0)
    self.assertAlmostEqual(s['ffn']['wall_seconds_per_object'], 1.2)
    self.assertEqual(s['ffn']['setup_seconds'], 30.0)
    self.assertAlmostEqual(s['segneuron']['total_seconds'], 240.0)
    self.assertAlmostEqual(s['segneuron']['seconds_per_object'], 120.0)
    self.assertAlmostEqual(s['ffn_over_segneuron'], 2.4 / 240.0)
    self.assertAlmostEqual(s['break_even_objects']['end_to_end'], 240.0 / 1.2)  # objects at 1.2 s each
    self.assertAlmostEqual(s['break_even_objects']['gpu_only'], 100.0 / 1.0)  # trace clock 1.0 s each


if __name__ == '__main__':
  unittest.main()
