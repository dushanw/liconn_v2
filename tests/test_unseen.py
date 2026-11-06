"""The unseen crops 41-50 (liconn_v2/unseen.py, docs/unseen_design.md): crops, refusals and the
pre-stated analysis on synthetic records. No crop of 41-50 is read here."""

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

from liconn_v2 import headtohead as hh
from liconn_v2 import manyseed
from liconn_v2 import unseen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'examples'))
import judge_unseen  # pylint: disable=g-import-not-at-top,wrong-import-position


def rec(oid, f1=0.9, precision=0.9, recall=0.9, steps=100, merges=0, merged_voxels=0):
  return {'object_id': oid, 'f1': f1, 'precision': precision, 'recall': recall, 'steps': steps,
          'merges': merges, 'merged_voxels': merged_voxels}


def crops_of(records_for_crop):
  return {k: records_for_crop(k) for k in unseen.CROPS}


class CropsTest(unittest.TestCase):

  def test_design_table_is_the_crop_list(self):
    table = re.findall(r'^ *\| (\d+) \| (\d+), (\d+), (\d+) \| ([\d.]+) \|$', unseen.DESIGN.read_text(), flags=re.M)
    self.assertEqual({int(k): (int(x), int(y), int(z)) for k, x, y, z, _ in table}, unseen.CROP_XYZ)
    self.assertEqual(tuple(sorted(unseen.CROP_XYZ)), unseen.CROPS)
    self.assertEqual(unseen.CROPS, tuple(range(41, 51)))

  def test_crops_are_the_rest_of_the_fixed_order(self):
    """Reads only the pinned coarse coverage map that chose crops 0-40."""
    coarse = manyseed.cache_dir() / manyseed.COARSE_FILE
    self.assertEqual(manyseed.data.sha256_file(coarse), manyseed.load_manifest()['coarse_sha256'])
    rest = unseen.ordered_crops(np.load(coarse))
    self.assertEqual([c['index'] for c in rest], list(unseen.CROPS))
    self.assertEqual(len(manyseed.select_crops(np.load(coarse), None)), unseen.USED_BEFORE + len(unseen.CROPS))

  def test_no_overlap_with_earlier_or_training_crops(self):
    used = [c['xyz'] for c in manyseed.load_manifest()['crops']] + \
           [c['xyz'] for c in json.loads((ROOT / 'docs' / 'fresh_crops.json').read_text())['crops']]
    for k, xyz in unseen.CROP_XYZ.items():
      for u in used + [list(manyseed.CROP_A_XYZ), list(manyseed.CROP_B_XYZ)]:
        self.assertFalse(manyseed.overlaps(xyz, u), (k, u))

  def test_other_crops_are_refused_before_any_file_is_read(self):
    for k in (0, 11, 12, 13, 40, 51):
      with self.assertRaises(ValueError):
        unseen.check_crop(k)
      with self.assertRaises(ValueError):
        unseen.load_raw_xyz(k)
      with self.assertRaises(ValueError):
        unseen.load_proofread_zyx(k)

  def test_run_scripts_refuse_other_crops(self):
    import run_unseen  # pylint: disable=g-import-not-at-top
    import run_unseen_wahab  # pylint: disable=g-import-not-at-top
    for mod in (run_unseen, run_unseen_wahab):
      for k, smoke in ((12, None), (13, None), (40, None), (51, None), (41, 20), (12, 20)):
        with self.assertRaises(ValueError, msg=f'{mod.__name__} {k} {smoke}'):
          mod.load_crop(k, smoke)

  def test_frozen_settings_match_the_design_and_earlier_runs(self):
    design = unseen.DESIGN.read_text()
    self.assertEqual(unseen.SEGNEURON_THRESHOLD, hh.THRESHOLD)
    self.assertEqual(unseen.V100_FFN_VARIANT, hh.FFN_VARIANT)
    self.assertEqual(unseen.FLIP, 0.10)
    self.assertIn('margin fixed now of 0.02 F1', design)
    self.assertIn('count <= 0.02 N + EPS', design)
    self.assertIn('mean d(recall) >= -0.01 - EPS; mean d(precision) >= -0.01 - EPS', design)
    self.assertIn('10,000 resamples, `numpy.random.default_rng(20261104)`', design)
    self.assertEqual((unseen.MAX_FLIP_FRACTION, unseen.MAX_MEAN_DROP, unseen.EPS), (0.02, 0.01, 1e-9))
    self.assertEqual((unseen.BOOTSTRAP_DRAWS, unseen.BOOTSTRAP_SEED), (10000, 20261104))
    for name, (rel, sha) in unseen.CHECKPOINTS.items():
      self.assertIn(f'{rel}, step', design, name)
      self.assertIn(f'({sha[:8]}...{sha[-5:]})', design, name)
    self.assertIn(f'({unseen.TEACHER_SHA256[:8]}...{unseen.TEACHER_SHA256[-5:]})', design)
    self.assertIn(f'({unseen.SEGNEURON_CKPT_SHA256[:8]}...{unseen.SEGNEURON_CKPT_SHA256[-5:]})', design)
    for run in unseen.B200_RUNS:
      self.assertIn(f'| {run} |', design)
    evidence = json.loads((ROOT / 'docs' / 'prune_evidence.json').read_text())['timing_job_E']
    self.assertEqual(unseen.CHECKPOINTS['student'][0], evidence['student_checkpoint'])
    self.assertEqual(unseen.CHECKPOINTS['p8'][0], evidence['pruned']['p8']['checkpoint'])
    self.assertEqual(unseen.CHECKPOINTS['p4'][0], evidence['pruned']['p4']['checkpoint'])


class FastModeRuleTest(unittest.TestCase):
  """Ten crops of ten objects: 100 objects, so at most 2 new merges and 2 new early stops."""

  def rule(self, cand):
    ref = crops_of(lambda k: [rec(i) for i in range(10)])
    return unseen.fast_mode_rule(ref, crops_of(cand))

  def test_identical_runs_pass(self):
    v = self.rule(lambda k: [rec(i) for i in range(10)])
    self.assertTrue(v['passes'])
    self.assertEqual(v['limit_objects'], 2.0)

  def test_two_new_merges_pass_three_fail(self):
    def cand(n):
      return lambda k: [rec(i, precision=0.7 if k == 41 and i < n else 0.9) for i in range(10)]
    self.assertEqual(self.rule(cand(2))['counts']['new_merges'], 2)
    # 2 objects fall 20 points: the mean falls 0.4 points, within the 1-point limit
    self.assertTrue(self.rule(cand(2))['rules']['new_merges'])
    self.assertFalse(self.rule(cand(3))['rules']['new_merges'])
    self.assertFalse(self.rule(cand(3))['passes'])

  def test_a_fall_of_exactly_ten_points_is_not_a_flip(self):
    v = self.rule(lambda k: [rec(i, recall=0.8 if i == 0 else 0.9) for i in range(10)])
    self.assertEqual(v['counts']['new_early_stops'], 0)

  def test_mean_limits(self):
    ok = self.rule(lambda k: [rec(i, recall=0.89) for i in range(10)])  # exactly 1 point
    self.assertTrue(ok['rules']['mean_recall_drop'])
    bad = self.rule(lambda k: [rec(i, precision=0.888) for i in range(10)])
    self.assertFalse(bad['rules']['mean_precision_drop'])
    self.assertFalse(bad['passes'])

  def test_needs_all_crops_and_the_same_objects(self):
    ref = crops_of(lambda k: [rec(i) for i in range(10)])
    with self.assertRaises(ValueError):
      unseen.fast_mode_rule({k: v for k, v in ref.items() if k != 50}, ref)
    with self.assertRaises(ValueError):
      unseen.fast_mode_rule(ref, crops_of(lambda k: [rec(i) for i in reversed(range(10))]))


class CropComparisonTest(unittest.TestCase):

  def test_paired_crops_by_hand(self):
    d = np.array([0.01, 0.03, -0.02, 0.05, 0.00, 0.02, 0.04, 0.01, -0.01, 0.03])
    a = {k: 0.8 + x for k, x in zip(unseen.CROPS, d)}
    b = {k: 0.8 for k in unseen.CROPS}
    r = unseen.paired_crops(a, b)
    half = 2.2621571627409915 * d.std(ddof=1) / np.sqrt(10)
    self.assertAlmostEqual(r['mean'], d.mean(), places=12)
    self.assertAlmostEqual(r['t_interval'][0], d.mean() - half, places=9)
    self.assertAlmostEqual(r['t_interval'][1], d.mean() + half, places=9)
    self.assertEqual(r, unseen.paired_crops(a, b))  # the bootstrap is seeded
    self.assertLess(r['bootstrap_interval'][0], d.mean())
    self.assertGreater(r['bootstrap_interval'][1], d.mean())
    self.assertEqual((r['crops_a_higher'], r['crops_b_higher']), (7, 2))
    with self.assertRaises(ValueError):
      unseen.paired_crops({k: v for k, v in a.items() if k != 41}, b)

  def test_margin_verdicts(self):
    label = lambda lo, hi: unseen.margin_verdict({'t_interval': [lo, hi]})['verdict']
    self.assertEqual(label(-0.015, 0.004), 'not worse than FFN by more than 0.02')
    self.assertEqual(label(-0.01, 0.0), 'not worse than FFN by more than 0.02')
    self.assertEqual(label(-0.015, -0.005), 'worse than FFN, by less than 0.02')
    self.assertEqual(label(-0.05, -0.03), 'worse than FFN')
    self.assertEqual(label(-0.02, 0.01), 'inconclusive')  # L must lie above -0.02
    self.assertTrue(unseen.margin_verdict({'t_interval': [0.001, 0.02]})['better_than_ffn'])
    self.assertFalse(unseen.difference_claimed({'t_interval': [-0.01, 0.0]}))
    self.assertTrue(unseen.difference_claimed({'t_interval': [0.001, 0.02]}))

  def test_runaways_and_merges(self):
    ref = [rec(0, steps=100), rec(1, steps=100), rec(2, steps=1)]
    small = [rec(0, steps=300, merges=2, merged_voxels=5000), rec(1, steps=299), rec(2, steps=50, merges=1)]
    self.assertEqual(unseen.object_counts(small), {'objects': 3, 'merges': 3, 'objects_with_merge': 2,
                                                   'merged_voxels': 5000})
    self.assertEqual(unseen.runaways(small, ref), {'runaway_traces': 2, 'object_ids': [0, 2],
                                                   'reference_stopped_at_first_step': [2]})
    with self.assertRaises(ValueError):
      unseen.runaways(small[::-1], ref)
    sn = [dict(rec(0), merges=[[5, 1200]], merged_voxels=1200)]
    self.assertEqual(unseen.object_counts(sn)['merges'], 1)

  def test_class_split_does_not_enter_the_rule(self):
    ref = crops_of(lambda k: [rec(10 * k + i) for i in range(10)])
    cand = crops_of(lambda k: [rec(10 * k + i, precision=0.7 if k == 41 and i < 2 else 0.9) for i in range(10)])
    classes = unseen.class_of({'dendrites': [410, 411], 'axons': [412]})
    v = unseen.fast_mode_rule(ref, cand, classes)
    self.assertEqual(v['by_class']['dendrites']['new_merges'], 2)
    self.assertEqual(v['by_class']['axons']['new_merges'], 0)
    self.assertEqual(v['by_class']['unclassified']['objects'], 97)
    self.assertEqual(v['passes'], unseen.fast_mode_rule(ref, cand)['passes'])
    self.assertTrue(v['passes'])

  def synthetic(self):
    def run(f1_by_crop):
      return lambda k: [rec(i, f1=f1_by_crop(k) + 0.001 * i) for i in range(5)]
    results = {}
    for name, f in (('F1', lambda k: 0.90), ('F5', lambda k: 0.90), ('S10', lambda k: 0.88 - 0.001 * (k % 3)),
                    ('P2', lambda k: 0.89), ('Q2', lambda k: 0.84 + 0.002 * (k % 2)), ('V', lambda k: 0.90)):
      results[name] = crops_of(run(f))
    results['segneuron'] = crops_of(lambda k: [dict(rec(i, f1=0.82 + 0.01 * (k % 4)), merges=[],
                                                    seed_in_background=i == 0) for i in range(5)])
    return results

  def test_analysis_end_to_end(self):
    results = self.synthetic()
    r = unseen.analysis(results)
    self.assertEqual(r['failures'], {})
    self.assertTrue(r['q1_fast_mode']['passes'])
    self.assertTrue(r['q1_fast_mode']['rules']['complete'])
    self.assertTrue(r['q2_ffn_vs_segneuron']['claims_agree'])
    self.assertEqual(r['q2_ffn_vs_segneuron']['V_vs_F5_agreement']['max_abs_f1_difference'], 0.0)
    self.assertEqual(r['q2_ffn_vs_segneuron']['F5_minus_segneuron']['objects']['segneuron_seeds_in_background'], 10)
    self.assertAlmostEqual(r['q2_ffn_vs_segneuron']['F5_minus_segneuron']['mean'],
                           np.mean([0.902 - (0.82 + 0.01 * (k % 4)) for k in unseen.CROPS]), places=12)
    self.assertEqual(r['q3_smaller_ffns']['P4_minus_ffn']['verdict'], 'worse than FFN')
    self.assertEqual(r['q3_smaller_ffns']['P8_minus_ffn']['verdict'], 'worse than FFN, by less than 0.02')
    lo, hi = r['q3_smaller_ffns']['P8_minus_ffn']['t_interval']
    self.assertAlmostEqual(lo, -0.01, places=9)
    self.assertAlmostEqual(hi, -0.01, places=9)
    self.assertEqual(sorted(r['runaways']), ['F5_vs_F1', 'P2_vs_F1', 'P2_vs_F5', 'Q2_vs_F1', 'Q2_vs_F5',
                                             'S10_vs_F1', 'S10_vs_F5'])
    self.assertEqual(r['runaways']['S10_vs_F1']['runaway_traces'], 0)
    self.assertEqual(r['objects_per_crop'], {k: 5 for k in unseen.CROPS})
    self.assertEqual(r['object_ids_in_more_than_one_crop'], 5)  # the synthetic crops reuse IDs 0-4
    results['V'] = crops_of(lambda k: [rec(i) for i in reversed(range(5))])
    with self.assertRaises(ValueError):
      unseen.analysis(results)

  def test_failures_are_reported_not_filled_in(self):
    results = self.synthetic()
    del results['F5'][43]
    with self.assertRaises(ValueError):  # a missing crop must be declared a failure
      unseen.analysis(results)
    r = unseen.analysis(results, {'F5': {43: '3 non-finite network outputs'}})
    self.assertFalse(r['q1_fast_mode']['passes'])
    self.assertEqual(r['q1_fast_mode']['not_judged'], {'F5': {43: '3 non-finite network outputs'}})
    self.assertIn('not_judged', r['q3_smaller_ffns']['P8_minus_ffn'])
    self.assertIn('not_judged', r['q2_ffn_vs_segneuron']['F5_minus_segneuron'])
    self.assertIn('difference_claimed', r['q2_ffn_vs_segneuron']['V_minus_segneuron'])
    self.assertIn('difference_claimed', r['q3_p8_minus_student'])
    self.assertNotIn('F5', r['pooled'])
    self.assertEqual(sorted(r['runaways']), ['P2_vs_F1', 'Q2_vs_F1', 'S10_vs_F1'])
    r = unseen.analysis(self.synthetic(), {'F1': {41: 'not run'}})
    self.assertIsNone(r['q1_fast_mode']['passes'])
    r = unseen.analysis(self.synthetic(), {'S10': {50: 'non-finite encoder features'}})
    self.assertIn('not_judged', r['q3_p8_minus_student'])
    self.assertTrue(r['q1_fast_mode']['passes'])


class JudgeTest(unittest.TestCase):
  """Synthetic result files, two objects per crop; the analysis itself is tested above."""

  def manifest(self):
    return {'design_sha256': 'abc', 'crops': [{'index': k, 'sha256': {'raw': f'r{k}'}} for k in unseen.CROPS]}

  def write(self, d: Path, design='abc', smoke=None, threshold=0.6, ids=(1, 2), records=(1, 2),
            crop_sha=None, drop=(), nonfinite=0, sn_ckpt=unseen.SEGNEURON_CKPT_SHA256, checkpoints=None,
            sn_nonfinite=0, settings=None, tf32=False, failed=()):
    def recs(ids_):
      return [dict(rec(i), merges=[], seed_in_background=False, trace_seconds=1.0, wall_seconds=1.0)
              for i in ids_]
    for k in unseen.CROPS:
      head = {'crop': k, 'smoke_objects': smoke, 'design_sha256': design, 'object_count': len(ids),
              'object_ids': list(ids), 'crop_sha256': crop_sha or {'raw': f'r{k}'}}
      folder = d / 'b200' / f'crop-{k:02d}'
      folder.mkdir(parents=True, exist_ok=True)
      for n in unseen.B200_RUNS:
        if (k, n) in drop:
          continue
        network, fmt, width, box_only, scale = (settings or {}).get(n, unseen.RUN_SETTINGS[n])
        run = {'summary': {'objects': len(ids), 'nonfinite_outputs': nonfinite if n == 'F5' else 0,
                           'crop_seconds': 10.0, 'network': network, 'format': fmt, 'width': width,
                           'box_only': box_only, 'stream_scale': scale},
               'records': [dict(x, merges=0) for x in recs(records)]}
        if (k, n) in failed:
          run = {'failed': '3 non-finite encoder features in float16'}
        frozen = judge_unseen.frozen_checkpoints()
        used = {c: frozen[c] for c in (unseen.RUN_CHECKPOINT.get(n),) if c}
        (folder / f'{n}.json').write_text(json.dumps(
            {**head, 'teacher_sha256': unseen.TEACHER_SHA256, 'run_name': n, 'tf32': tf32,
             'checkpoints': used if checkpoints is None else checkpoints,
             'device': 'NVIDIA B200', 'host': f'node{k}', 'run': run}))
      v = d / 'v100' / f'crop-{k:02d}'
      v.mkdir(parents=True, exist_ok=True)
      if (k, 'V') not in drop:
        (v / 'ffn.json').write_text(json.dumps(
            {**head, 'variant': 'fp16_head32', 'records': recs(records), 'setup_seconds': 1.0, 'overlap': {},
             'checkpoint_sha256': {f: judge_unseen.data.PINNED_SHA256[f] for f in judge_unseen.data.CHECKPOINT_FILES}}))
      (v / 'segneuron.json').write_text(json.dumps(
          {**head, 'threshold': threshold, 'checkpoint_sha256': sn_ckpt, 'nonfinite_outputs': sn_nonfinite,
           'objects': recs(records),
           'inference_wall_seconds': 50.0, 'inference': {'inference_seconds': 40.0}, 'post_seconds': 30.0}))

  def judge(self, manifest=None, **kw):
    with tempfile.TemporaryDirectory() as t:
      d = Path(t)
      self.write(d, **kw)
      return judge_unseen.judge(d / 'b200', d / 'v100', manifest or self.manifest(), 'abc')

  def test_complete_files(self):
    r = self.judge()
    self.assertEqual(r['failures'], {})
    self.assertTrue(r['q1_fast_mode']['passes'])
    self.assertEqual(r['speed_v100']['totals']['objects'], 20)
    self.assertAlmostEqual(r['speed_b200']['F5']['speedup_vs_F1'], 1.0)

  def test_refusals(self):
    for kw in ({'design': 'other'}, {'smoke': 20}, {'threshold': 0.5}, {'records': (2, 1)}, {'records': (1,)},
               {'crop_sha': {'raw': 'other'}}, {'sn_ckpt': 'other'}, {'checkpoints': {}},
               {'nonfinite': None}, {'tf32': True},
               {'settings': {'F5': ('ffn', 'fp16', 32, True, 1 / 16)}},
               {'settings': {'F1': ('ffn', 'fp32', 32, True, 1.0)}},
               {'settings': {'P2': ('p8', 'fp16_head32', 32, False, 1 / 16)}},
               {'settings': {'Q2': ('p4', 'fp16_head32', 32, True, 1.0)}},
               {'settings': {'S10': ('ffn', 'fp16_head32', 32, True, 1 / 16)}},
               {'settings': {'F1': ('ffn', 'fp32', True, True, 1.0)}},
               {'settings': {'F1': ('ffn', 'fp32', 1, 1, 1.0)}}):
      with self.assertRaises(ValueError, msg=str(kw)):
        self.judge(**kw)
    with self.assertRaises(ValueError):  # the design changed after the fetch
      self.judge(manifest={**self.manifest(), 'design_sha256': 'older'})

  def test_missing_and_non_finite_runs_are_failures(self):
    r = self.judge(drop={(44, 'P2'), (45, 'V')})
    self.assertEqual(r['failures'], {'P2': {44: 'no B200 result file: the task stopped before the run ended'},
                                     'V': {45: 'no V100 result file ffn.json'}})
    self.assertIn('not_judged', r['q3_smaller_ffns']['P8_minus_ffn'])
    self.assertIn('not_judged', r['speed_v100'])
    self.assertTrue(r['q1_fast_mode']['passes'])
    r = self.judge(nonfinite=2)
    self.assertFalse(r['q1_fast_mode']['passes'])
    self.assertEqual(r['failures']['F5'][41], '2 non-finite network outputs')
    r = self.judge(sn_nonfinite=5)
    self.assertEqual(r['failures']['segneuron'][41], '5 non-finite SegNeuron outputs')
    self.assertIn('not_judged', r['q2_ffn_vs_segneuron']['F5_minus_segneuron'])

  def test_one_method_failing_leaves_the_others_judged(self):
    """Each run is its own task: the student failing or missing on a crop takes nothing else with it."""
    r = self.judge(failed={(41, 'S10')}, drop={(42, 'S10'), (43, 'V')})
    self.assertEqual(set(r['failures']), {'S10', 'V'})
    self.assertEqual(set(r['failures']['S10']), {41, 42})
    self.assertTrue(r['q1_fast_mode']['passes'])
    self.assertNotIn('not_judged', r['q3_smaller_ffns']['P8_minus_ffn'])
    self.assertNotIn('not_judged', r['q2_ffn_vs_segneuron']['F5_minus_segneuron'])
    self.assertIn('not_judged', r['speed_v100'])
    self.assertIn('total_seconds', r['speed_b200']['F1'])

  def test_settings_are_the_timing_job_s(self):
    import time_crop11  # pylint: disable=g-import-not-at-top
    import run_unseen  # pylint: disable=g-import-not-at-top
    for n in unseen.B200_RUNS:
      self.assertEqual((*time_crop11.RUNS[n], time_crop11.SCALE.get(n, 1.0)), unseen.RUN_SETTINGS[n], n)
      run_unseen.check_run(n)
    with self.assertRaises(ValueError):
      run_unseen.check_run('F3')
    for n, (network, fmt, width, _, scale) in unseen.RUN_SETTINGS.items():
      table = {'ffn': 'FFN', 'student': 'student', 'p8': 'P8', 'p4': 'P4'}[network]
      fmt_text = 'float32, TF32 off' if fmt == 'fp32' else f'`{fmt}`'
      scale_text = '1' if scale == 1.0 else '1/16'
      self.assertIn(f'| {n} | {table} | {fmt_text} | {width} | {scale_text} |', unseen.DESIGN.read_text())

  def test_job_scripts_give_each_method_its_own_task(self):
    mit = (ROOT / 'cluster' / 'mit' / 'unseen.sbatch').read_text()
    self.assertIn(f'RUNS=({" ".join(unseen.B200_RUNS)})', mit)
    self.assertIn('--array=0-49', mit)
    self.assertIn('--run "$NAME"', mit)
    wahab = (ROOT / 'cluster' / 'wahab' / 'unseen.sbatch').read_text()
    ffn_branch = wahab[wahab.index('if [ "$METHOD" = ffn ]'):wahab.index('fi\n')]
    self.assertIn('run_unseen_wahab.py ffn', ffn_branch)
    self.assertIn('exit 0', ffn_branch)
    self.assertNotIn('inference.py', ffn_branch)
    self.assertNotIn('run_unseen_wahab.py ffn', wahab[wahab.index('fi\n'):])

  def test_fetch_never_repins_a_changed_design(self):
    with tempfile.TemporaryDirectory() as t:
      manifest = Path(t) / 'unseen_crops.json'
      manifest.write_text(json.dumps({'design_sha256': 'older', 'crops': []}))
      old, unseen.MANIFEST = unseen.MANIFEST, manifest
      try:
        with self.assertRaises(ValueError):
          unseen.fetch()  # refuses before opening the bucket
      finally:
        unseen.MANIFEST = old
      self.assertEqual(json.loads(manifest.read_text())['design_sha256'], 'older')


if __name__ == '__main__':
  unittest.main()
