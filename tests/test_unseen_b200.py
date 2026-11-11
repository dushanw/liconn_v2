"""The SegNeuron-on-B200 addendum's measures (liconn_v2/unseen_b200.py), on synthetic result files."""

import copy
import unittest

from liconn_v2 import unseen
from liconn_v2 import unseen_b200 as ub

CROPS = unseen.CROPS


def sn_file(k: int, ids: list, f1: list, wall=100.0, loop=60.0, post=40.0, nonfinite=0) -> dict:
  return {'crop': k, 'smoke_objects': None, 'design_sha256': unseen.design_sha256(),
          'checkpoint_sha256': unseen.SEGNEURON_CKPT_SHA256, 'threshold': unseen.SEGNEURON_THRESHOLD,
          'crop_sha256': f'crop{k}', 'object_ids': ids, 'object_count': len(ids), 'nonfinite_outputs': nonfinite,
          'inference_wall_seconds': wall, 'inference': {'inference_seconds': loop}, 'post_seconds': post,
          'objects': [{'object_id': i, 'f1': f} for i, f in zip(ids, f1)]}


def evidence(crop_seconds: dict, objects: dict) -> dict:
  """crop_seconds {run: {k: s}} for the five B200 runs."""
  return {'judged': {'objects_per_crop': {str(k): objects[k] for k in CROPS},
                     'speed_b200': {r: {'crop_seconds': {str(k): crop_seconds[r][k] for k in CROPS}}
                                    for r in ub.RUNS}}}


class StatementTest(unittest.TestCase):

  def test_wording(self):
    self.assertEqual(ub.statement(10, 0, 10, 'batched fast FFN'),
                     'SegNeuron was cheaper than batched fast FFN on every test crop')
    self.assertEqual(ub.statement(0, 10, 10, 'batched fast FFN'),
                     'batched fast FFN was cheaper than SegNeuron on every test crop')
    self.assertEqual(ub.statement(3, 7, 10, 'P8'), 'SegNeuron was cheaper on 3 of 10 crops')
    self.assertEqual(ub.statement(0, 9, 10, 'P8'), 'SegNeuron was cheaper on 0 of 10 crops, equal on 1')
    self.assertEqual(ub.statement(2, 7, 9, 'P8'), 'SegNeuron was cheaper on 2 of 9 judged crops (1 not judged)')
    self.assertEqual(ub.statement(2, 6, 9, 'P8'),
                     'SegNeuron was cheaper on 2 of 9 judged crops, equal on 1 (1 not judged)')
    self.assertEqual(ub.statement(0, 0, 0, 'P8'), 'not judged')


class CompareTest(unittest.TestCase):

  def test_break_even(self):
    objects = {k: 400 for k in CROPS}
    run = {k: 200.0 for k in CROPS}  # 0.5 s per object
    sn = {k: 100.0 for k in CROPS}
    c = ub.compare(sn, run, objects, 'batched fast FFN')
    self.assertEqual(c['crops_segneuron_cheaper'], 10)
    self.assertEqual(c['break_even_per_crop'][41], 200.0)
    self.assertEqual(c['totals']['break_even_per_crop'], 200.0)
    self.assertEqual(c['statement'], 'SegNeuron was cheaper than batched fast FFN on every test crop')

  def test_totals_weight_crops_by_time_and_objects(self):
    objects = {k: (300 if k < 46 else 500) for k in CROPS}
    run = {k: float(objects[k]) for k in CROPS}  # 1 s per object
    sn = {k: (250.0 if k < 46 else 600.0) for k in CROPS}
    c = ub.compare(sn, run, objects, 'P4')
    self.assertEqual(c['cheaper_on'], [41, 42, 43, 44, 45])
    self.assertEqual(c['statement'], 'SegNeuron was cheaper on 5 of 10 crops')
    self.assertEqual(c['break_even'], {'median': 425.0, 'min': 250.0, 'max': 600.0})
    self.assertAlmostEqual(c['totals']['break_even_per_crop'], (4250 / 10) / (4000 / 4000))

  def test_equal_time_is_not_cheaper(self):
    c = ub.compare({k: 1.0 for k in CROPS}, {k: 1.0 for k in CROPS}, {k: 10 for k in CROPS}, 'P8')
    self.assertEqual(c['crops_segneuron_cheaper'], 0)
    self.assertEqual(c['crops_segneuron_dearer'], 0)
    self.assertEqual(c['statement'], 'SegNeuron was cheaper on 0 of 10 crops, equal on 10')

  def test_refuses_invalid_run_times(self):
    for bad in (0.0, -1.0, float('nan'), float('inf'), None):
      run = {k: 1.0 for k in CROPS}
      run[45] = bad
      with self.assertRaises(ValueError, msg=bad):
        ub.compare({k: 1.0 for k in CROPS}, run, {k: 10 for k in CROPS}, 'P8')

  def test_crop_not_timed(self):
    sn = {k: 1.0 for k in CROPS}
    sn[47] = None
    c = ub.compare(sn, {k: 2.0 for k in CROPS}, {k: 10 for k in CROPS}, 'P8')
    self.assertEqual(c['crops_judged'], 9)
    self.assertNotIn(47, c['break_even_per_crop'])
    self.assertNotIn('totals', c)
    self.assertEqual(c['statement'], 'SegNeuron was cheaper on 9 of 9 judged crops (1 not judged)')


class StagingTest(unittest.TestCase):

  ENV = 'numpy==2.5.4\ntifffile==2026.10.15\ntorch==2.7.1+cu128\ntriton==3.3.1\n'
  PINNED = {'examples/run_unseen_wahab.py': 'a' * 64, 'liconn_v2/unseen.py': 'b' * 64}
  STAGED = (f"{'a' * 64}  examples/run_unseen_wahab.py\n{'b' * 64}  liconn_v2/unseen.py\n"
            f"{'c' * 64}  liconn_v2/unseen_b200.py\n")

  def test_same_source_and_pinned_versions(self):
    ub.check_staging({41: self.STAGED, 42: self.STAGED}, {41: self.ENV, 42: self.ENV}, self.PINNED)

  def test_refuses_mixed_sources(self):
    other = self.STAGED.replace('c' * 64, 'd' * 64)
    with self.assertRaises(ValueError):
      ub.check_staging({41: self.STAGED, 42: other}, {41: self.ENV, 42: self.ENV}, self.PINNED)

  def test_refuses_changed_or_missing_test_code(self):
    for staged in (self.STAGED.replace('b' * 64, 'e' * 64),
                   self.STAGED.replace(f"{'b' * 64}  liconn_v2/unseen.py\n", '')):
      with self.assertRaises(ValueError):
        ub.check_staging({41: staged}, {41: self.ENV}, self.PINNED)

  def test_refuses_other_versions(self):
    for env in (self.ENV.replace('+cu128', '+cu126'), self.ENV.replace('numpy==2.5.4', 'numpy==2.5.3'),
                self.ENV.replace('tifffile==2026.10.15\n', '')):
      with self.assertRaises(ValueError):
        ub.check_staging({41: self.STAGED}, {41: env}, self.PINNED)


class JudgeTest(unittest.TestCase):

  def setUp(self):
    self.ids = {k: [k * 10 + i for i in range(4)] for k in CROPS}
    self.v100 = {k: sn_file(k, self.ids[k], [0.9, 0.8, 0.7, 0.6], wall=300, loop=200, post=300) for k in CROPS}
    self.b200 = {k: sn_file(k, self.ids[k], [0.9, 0.8, 0.7, 0.45]) for k in CROPS}
    seconds = {r: {k: s for k in CROPS} for r, s in
               (('F1', 1000.0), ('F5', 150.0), ('S10', 90.0), ('P2', 80.0), ('Q2', 50.0))}
    self.evidence = evidence(seconds, {k: 4 for k in CROPS})

  def test_measures(self):
    j = ub.judge(self.b200, self.v100, self.evidence)
    self.assertEqual(j['failures'], {})
    self.assertEqual(j['segneuron_b200'][41]['end_to_end'], 140.0)
    self.assertEqual(j['segneuron_b200'][41]['without_startup'], 100.0)
    self.assertEqual(j['segneuron_v100'][41]['end_to_end'], 600)
    e2e, ws = j['comparisons']['end_to_end'], j['comparisons']['without_startup']
    self.assertEqual(e2e['F1']['statement'], 'SegNeuron was cheaper than float32 FFN one object at a time on every test crop')
    self.assertEqual(e2e['F5']['statement'], 'SegNeuron was cheaper than batched fast FFN on every test crop')
    self.assertEqual(e2e['S10']['statement'], 'the student was cheaper than SegNeuron on every test crop')
    self.assertEqual(ws['S10']['crops_segneuron_cheaper'], 0)  # 100 s against 90 s
    self.assertEqual(ws['F5']['totals']['break_even_per_crop'], 100.0 / (150.0 / 4))
    a = j['agreement'][41]
    self.assertAlmostEqual(a['mean_f1_difference'], -0.15 / 4)
    self.assertEqual(a['objects_abs_difference_gt_0.1'], 1)
    self.assertEqual(a['identical_f1'], 3)
    self.assertEqual(j['agreement_pooled']['objects'], 40)
    self.assertEqual(j['agreement_pooled']['objects_abs_difference_gt_0.1'], 10)

  def test_failures_are_not_judged(self):
    self.b200[43] = None
    self.b200[44] = sn_file(44, self.ids[44], [0.9] * 4, nonfinite=5)
    j = ub.judge(self.b200, self.v100, self.evidence)
    self.assertEqual(j['failures'], {43: 'no result file', 44: '5 non-finite SegNeuron outputs'})
    self.assertEqual(j['comparisons']['end_to_end']['F5']['crops_judged'], 8)
    self.assertNotIn(43, j['agreement'])
    self.assertNotIn(44, j['agreement'])
    self.assertEqual(j['agreement_pooled']['objects'], 32)

  def test_invalid_times_are_not_judged(self):
    for field, bad in (('inference_wall_seconds', float('nan')), ('post_seconds', -1.0),
                       ('inference_seconds', float('inf')), ('inference_wall_seconds', None)):
      b = copy.deepcopy(self.b200)
      if field == 'inference_seconds':
        b[42]['inference'][field] = bad
      else:
        b[42][field] = bad
      j = ub.judge(b, self.v100, self.evidence)
      self.assertIn(42, j['failures'], field)
      self.assertEqual(j['comparisons']['end_to_end']['F5']['crops_judged'], 9)
      self.assertNotIn('totals', j['comparisons']['end_to_end']['F5'])

  def test_device_must_be_a_b200(self):
    ub.check_device('NVIDIA B200, 580.82.07', 41)
    for gpu in ('Tesla V100-SXM2-16GB, 580.178.04', 'NVIDIA H200, 580.82.07', ''):
      with self.assertRaises(ValueError):
        ub.check_device(gpu, 41)

  def test_object_count_must_be_the_test_s(self):
    b, v = copy.deepcopy(self.b200), copy.deepcopy(self.v100)
    for f in (b[41], v[41]):  # both runs scored three of the crop's four objects
      f['object_ids'] = f['object_ids'][:3]
      f['objects'] = f['objects'][:3]
    with self.assertRaises(ValueError):
      ub.judge(b, v, self.evidence)
    b = copy.deepcopy(self.b200)
    b[41]['object_count'] = 3
    with self.assertRaises(ValueError):
      ub.judge(b, self.v100, self.evidence)

  def test_refuses_other_v100_files(self):
    for change, value in (('crop', 42), ('smoke_objects', 20), ('design_sha256', 'x'), ('checkpoint_sha256', 'x'),
                          ('threshold', 0.5)):
      v = copy.deepcopy(self.v100)
      v[41][change] = value
      with self.assertRaises(ValueError, msg=change):
        ub.judge(self.b200, v, self.evidence)

  def test_refuses_other_files(self):
    for change, value in (('crop', 42), ('smoke_objects', 20), ('design_sha256', 'x'), ('checkpoint_sha256', 'x'),
                          ('threshold', 0.5), ('crop_sha256', 'x'), ('object_ids', [1, 2, 3, 4])):
      b = copy.deepcopy(self.b200)
      b[41][change] = value
      with self.assertRaises(ValueError, msg=change):
        ub.judge(b, self.v100, self.evidence)
    b = copy.deepcopy(self.b200)
    b[41]['objects'] = b[41]['objects'][::-1]
    with self.assertRaises(ValueError):
      ub.judge(b, self.v100, self.evidence)


if __name__ == '__main__':
  unittest.main()
