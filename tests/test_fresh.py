"""Fresh seeds for confirming fp16_head32 (liconn_v2/fresh.py, docs/fresh_design.md)."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from liconn_v2 import benchmark
from liconn_v2 import fresh
from liconn_v2 import manyseed


class FreshSelectionTest(unittest.TestCase):

  def test_order_continues_after_the_many_seed_crops(self):
    self.assertEqual(fresh.first_crop(), len(manyseed.load_manifest()['crops']))
    order = manyseed.select_crops(np.load(manyseed.cache_dir() / manyseed.COARSE_FILE), None)
    m = fresh.load_manifest()
    self.assertEqual(m['first_crop'], fresh.first_crop())
    n = len(m['crops'])
    self.assertEqual([(c['index'], c['xyz'], c['coverage']) for c in m['crops']],
                     [(c['index'], c['xyz'], c['coverage']) for c in order[fresh.first_crop():fresh.first_crop() + n]])

  def test_fresh_seeds_are_new_unique_and_exactly_300(self):
    seeds = fresh.recorded_seeds()
    ids = [s['object_id'] for s in seeds]
    self.assertEqual(len(seeds), fresh.TARGET_SEEDS)
    self.assertEqual(len(ids), len(set(ids)))
    self.assertFalse(set(ids) & fresh.excluded_objects())
    self.assertTrue(set(ids).isdisjoint(s['object_id'] for s in manyseed.recorded_seeds()))
    crops = [s['crop'] for s in seeds]
    self.assertEqual(crops, sorted(crops))
    self.assertGreaterEqual(min(crops), fresh.first_crop())
    self.assertLess(sum(c != max(crops) for c in crops), fresh.TARGET_SEEDS)  # extra crops only while short
    for k in sorted(set(crops)):
      mine = [s for s in seeds if s['crop'] == k]
      self.assertLessEqual(len(mine), manyseed.SEEDS_PER_CROP)
      self.assertLessEqual(sum(s['class'] == 'dendrite' for s in mine), manyseed.MAX_DENDRITES_PER_CROP)

  def test_pins_and_rederivation(self):
    m = fresh.load_manifest()
    for c in m['crops']:
      fresh.verify_crop(c['index'], m)
    seeds = fresh.recorded_seeds()
    for k in (min(s['crop'] for s in seeds), max(s['crop'] for s in seeds)):
      gt = fresh.load_proofread_zyx(k)
      mine = fresh.seeds_for_crop(k, gt)
      for s in mine:
        self.assertEqual(int(gt[tuple(s['seed_zyx'])]), s['object_id'])
        self.assertTrue(all(benchmark.MARGIN <= v < n - benchmark.MARGIN for v, n in zip(s['seed_zyx'], gt.shape)))


class FreshVerdictTest(unittest.TestCase):

  def test_runs_for_crops_numbered_from_13_pair_by_position(self):
    import sys  # pylint: disable=g-import-not-at-top
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'examples'))
    import judge_fresh  # pylint: disable=g-import-not-at-top
    seeds, ref, cand = [], [], []
    for crop in (13, 14):
      run = {'seeds': [], 'records': [], 'inputs_sha256': {'raw': str(crop)}}
      for i in range(150):
        s = {'name': f'm{crop}_axon_{i}', 'crop': crop, 'class': 'axon', 'object_id': crop * 1000 + i,
             'seed_zyx': [i, crop, 0]}
        seeds.append(s)
        run['seeds'].append(dict(s))
        run['records'].append({'name': s['name'], 'recall': 0.9, 'precision': 0.95,
                               'object_id': s['object_id'], 'seed_zyx': s['seed_zyx']})
      ref.append(run)
      cand.append(json.loads(json.dumps(run)))
    pinned = [{'raw': '13'}, {'raw': '14'}]
    v = judge_fresh.fresh_verdict(ref, cand, seeds, pinned, [13, 14])
    self.assertTrue(v['passes'], v['problems'][:3])
    # without the mapping, every seed would look traced outside its crop's run pair
    self.assertFalse(manyseed.verdict(ref, cand, seeds, pinned)['rules']['complete_same_inputs'])
    swapped = judge_fresh.fresh_verdict(ref, cand, seeds, pinned, [14, 13])
    self.assertFalse(swapped['passes'])


class StorageCapTest(unittest.TestCase):
  """The download stops before passing the owner-approved storage and pins nothing."""

  def test_cap_stops_and_pins_nothing(self):
    tmp = Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, tmp)

    def fake_fetch(c, image, seg):
      d = tmp / f"crop-{c['index']:02d}"
      d.mkdir(exist_ok=True)
      np.save(d / manyseed.RAW_FILE, np.zeros((8, 8, 8), np.uint8))
      np.save(d / manyseed.PROOFREAD_FILE, np.zeros((8, 8, 8), np.uint32))

    one = lambda gt, k, excluded, classes=None: [{'name': f'm{k}_axon_{k}', 'crop': k, 'class': 'axon',
                                                  'object_id': 10 ** 7 + k, 'seed_zyx': [1, 1, 1], 'candidates': 1}]
    with mock.patch.object(fresh, 'MANIFEST', tmp / 'crops.json'), \
         mock.patch.object(fresh, 'SEEDS_FILE', tmp / 'seeds.json'), \
         mock.patch.object(fresh, 'MAX_FRESH_BYTES', 3000), \
         mock.patch.object(manyseed, 'crop_dir', lambda k: tmp / f'crop-{k:02d}'), \
         mock.patch.object(manyseed, '_open', return_value=None), \
         mock.patch.object(manyseed, 'fetch_crop', side_effect=fake_fetch), \
         mock.patch.object(manyseed, 'select_seeds_for_crop', side_effect=one):
      with self.assertRaisesRegex(RuntimeError, 'approved'):
        fresh.fetch_and_select()
    self.assertFalse((tmp / 'crops.json').exists())
    self.assertFalse((tmp / 'seeds.json').exists())

  def test_running_out_of_crops_pins_nothing(self):
    tmp = Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, tmp)
    few = [{'index': 13, 'xyz': [0, 0, 111], 'coverage': 0.6}, {'index': 14, 'xyz': [500, 0, 111], 'coverage': 0.6}]
    one = lambda gt, k, excluded, classes=None: [{'name': f'm{k}_axon_{k}', 'crop': k, 'class': 'axon',
                                                  'object_id': 10 ** 7 + k, 'seed_zyx': [1, 1, 1], 'candidates': 1}]

    def fake_fetch(c, image, seg):
      d = tmp / f"crop-{c['index']:02d}"
      d.mkdir(exist_ok=True)
      np.save(d / manyseed.PROOFREAD_FILE, np.zeros((8, 8, 8), np.uint32))

    with mock.patch.object(fresh, 'MANIFEST', tmp / 'crops.json'), \
         mock.patch.object(fresh, 'SEEDS_FILE', tmp / 'seeds.json'), \
         mock.patch.object(manyseed, 'select_crops', return_value=few), \
         mock.patch.object(manyseed, 'crop_dir', lambda k: tmp / f'crop-{k:02d}'), \
         mock.patch.object(manyseed, '_open', return_value=None), \
         mock.patch.object(manyseed, 'fetch_crop', side_effect=fake_fetch), \
         mock.patch.object(manyseed, 'select_seeds_for_crop', side_effect=one):
      with self.assertRaisesRegex(RuntimeError, 'ran out'):
        fresh.fetch_and_select()
    self.assertFalse((tmp / 'crops.json').exists())


if __name__ == '__main__':
  unittest.main()
