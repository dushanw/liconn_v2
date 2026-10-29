"""Tests for liconn_v2.segneuron_ft (docs/segneuron_finetune_design.md)."""

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from liconn_v2 import segneuron_ft as ft
from liconn_v2 import vitseg


def authors_lr(iters, base, end, warmup, decay, power):
  """supervised_train.calculate_lr, transcribed from the pinned checkout."""
  if iters < warmup:
    return (base - end) * pow(float(iters) / warmup, power) + end
  if iters < decay:
    return (base - end) * pow(1 - float(iters - warmup) / decay, power) + end
  return end


class SegneuronFtTest(unittest.TestCase):

  def test_crop_split_is_the_vits(self):
    self.assertEqual(ft.TRAIN_CROPS, vitseg.TRAIN_CROPS)
    self.assertEqual(ft.VALID_CROP, vitseg.TUNE_CROP)
    self.assertNotIn(ft.VALID_CROP, ft.TRAIN_CROPS)

  def test_learning_rate_is_the_authors_schedule(self):
    total = 5000
    for step in (1, 10, 999, 2500, 4999, 5000, 6000):
      self.assertAlmostEqual(ft.learning_rate(step, total),
                             authors_lr(step, ft.BASE_LR, ft.END_LR, 0, total, ft.POWER), places=15)
    self.assertAlmostEqual(ft.learning_rate(0, total), ft.BASE_LR)
    self.assertEqual(ft.learning_rate(total, total), ft.END_LR)

  def test_relabel_keeps_the_partition(self):
    lab = np.array([[[0, 7, 7], [2**40, 0, 5]]], np.uint64)
    r = ft.relabel(lab)
    self.assertEqual(r.dtype, np.uint32)
    self.assertTrue(np.array_equal(r == 0, lab == 0))
    for a in np.unique(lab):
      self.assertEqual(len(np.unique(r[lab == a])), 1)
    self.assertEqual(len(np.unique(r)), len(np.unique(lab)))
    self.assertEqual(ft.relabel(np.array([[[3, 3, 9]]], np.uint64)).min(), 1)  # no 0: none invented

  def test_draws_stay_inside_and_never_use_crop_12(self):
    rng = np.random.default_rng(0)
    shapes = {k: (500, 500, 500) for k in ft.TRAIN_CROPS}
    seen = set()
    for _ in range(2000):
      k, c = ft.train_draw(rng, shapes)
      seen.add(k)
      self.assertTrue(all(0 <= v <= s - p for v, s, p in zip(c, shapes[k], ft.PATCH)))
    self.assertEqual(seen, set(ft.TRAIN_CROPS))
    with self.assertRaises(ValueError):
      ft.train_draw(rng, {**shapes, ft.VALID_CROP: (500, 500, 500)})
    with self.assertRaises(ValueError):
      ft.corner(rng, (10, 500, 500))

  def test_valid_corners_are_fixed(self):
    a, b = ft.valid_corners((500, 500, 500)), ft.valid_corners((500, 500, 500))
    self.assertEqual(a, b)
    self.assertEqual(len(a), ft.VALID_PATCHES)
    self.assertTrue(all(0 <= v <= 500 - p for c in a for v, p in zip(c, ft.PATCH)))

  def test_write_crops(self):
    raw = lambda k: np.full((4, 5, 6), k, np.uint8)
    lab = lambda k: np.arange(120, dtype=np.uint64).reshape(4, 5, 6) * 1000
    with tempfile.TemporaryDirectory() as d:
      m = ft.write_crops(Path(d), raw, lab, crops=(0, 12))
      self.assertEqual(json.loads((Path(d) / 'manifest.json').read_text()), m)
      self.assertEqual(m['crops']['12'], {'shape_zyx': [4, 5, 6], 'objects': 119, 'labelled_fraction': 119 / 120})
      self.assertEqual(np.load(Path(d) / ft.RAW.format(12))[0, 0, 0], 12)
      self.assertEqual(np.load(Path(d) / ft.LABELS.format(0)).dtype, np.uint32)
      with self.assertRaises(ValueError):
        ft.write_crops(Path(d), lambda k: np.zeros((4, 5, 6), np.float32), lab, crops=(0,))

  def test_compact_patch(self):
    lab = np.array([[[0, 70000, 70000, 5]]], np.uint64)
    r = ft.compact_patch(lab)
    self.assertEqual(r.tolist(), [[[0, 2, 2, 1]]])
    big = np.arange(1, 2**16 + 1, dtype=np.uint64).reshape(1, 1, -1)
    with self.assertRaises(ValueError):
      ft.compact_patch(big)

  def test_boundary_errors(self):
    lab = np.zeros((1, 1, 4), np.uint32)
    lab[0, 0] = [1, 1, 2, 0]
    pred = np.ones((3, 1, 1, 4), np.float32)
    pred[2, 0, 0, 1] = 0.2  # x pair (1, 0): same neuron, predicted apart
    e = ft.boundary_errors(pred, lab)
    self.assertEqual(e, {'same': 1, 'same_below_0.5': 1, 'diff': 1, 'diff_at_or_above_0.5': 1})
    pred[2, 0, 0, 2] = 0.1  # x pair (2, 1): different neurons, predicted apart
    self.assertEqual(ft.boundary_errors(pred, lab)['diff_at_or_above_0.5'], 0)

  def test_crop_12_is_apart_from_the_training_crops(self):
    from pathlib import Path as P  # pylint: disable=g-import-not-at-top
    crops = json.loads((P(__file__).resolve().parent.parent / 'docs' / 'manyseed_crops.json').read_text())['crops']
    xyz = {c['index']: c['xyz'] for c in crops}
    self.assertEqual(min(ft.crop_gap(xyz[ft.VALID_CROP], xyz[k]) for k in ft.TRAIN_CROPS), 500)
    self.assertEqual(ft.crop_gap((0, 0, 0), (500, 0, 0)), 0)
    self.assertLess(ft.crop_gap((0, 0, 0), (499, 0, 0)), 0)

  def test_best_step(self):
    log = [{'step': 0, 'valid_loss': 0.5}, {'step': 1000, 'valid_loss': 0.3},
           {'step': 2000, 'valid_loss': float('nan')}, {'step': 3000, 'valid_loss': 0.3}]
    self.assertEqual(ft.best_step(log)['step'], 1000)
    with self.assertRaises(ValueError):
      ft.best_step([{'step': 0, 'valid_loss': float('inf')}])


if __name__ == '__main__':
  unittest.main()
