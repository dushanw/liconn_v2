"""Float32 batching study (liconn_v2/fp32_batch.py, docs/fp32_batch_design.md), on synthetic records."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from liconn_v2 import fp32_batch as fb
from liconn_v2 import unseen
from liconn_v2 import unseen_b200

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'examples'))

CROPS = unseen.CROPS


def rec(i: int, steps=10, mask='m', hist='h', f1=0.9) -> dict:
  return {'object_id': i, 'steps': steps, 'mask_sha256': f'{mask}{i}', 'history_sha256': f'{hist}{i}', 'f1': f1,
          'precision': 0.95, 'recall': 0.9, 'merges': 0, 'merged_voxels': 0}


def run(seconds: float, records: list, name='F32x1', setup=0.0) -> dict:
  """A run whose traces took `seconds` minus `setup` (crop time includes the image copy)."""
  fmt, width, scale = fb.RUNS[name]
  steps = sum(r['steps'] for r in records)
  return {'summary': {'format': fmt, 'width': width, 'stream_scale': scale, 'objects': len(records),
                      'crop_seconds': seconds, 'steps': steps, 'ms_per_step': 1000 * (seconds - setup) / steps,
                      'mean_f1': 0.9, 'merges_ge_1000': 0, 'nonfinite_outputs': None},
          'records': records}


def result_file(k: int, ids: list, seconds: float, name='F32x32', steps=10, **changes) -> dict:
  fmt, width, scale = fb.RUNS.get(name, ('fp32', 32, 1.0))
  records = [rec(i, steps=steps) for i in ids]
  x = {'crop': k, 'smoke_objects': None, 'run_name': name, 'teacher_sha256': unseen.TEACHER_SHA256, 'tf32': False,
       'design_sha256': fb.design_sha256(), 'host': 'node', 'device': 'NVIDIA B200', 'crop_sha256': {'raw': f'c{k}'},
       'object_ids': list(ids),
       'run': {'summary': {'network': 'ffn', 'box_only': True, 'format': fmt, 'width': width, 'stream_scale': scale,
                           'objects': len(ids), 'crop_seconds': seconds, 'steps': steps * len(ids),
                           'ms_per_step': 1000 * seconds / (steps * len(ids))},
               'records': records}}
  x.update(changes)
  return x


def widths_file(runs: dict, **changes) -> dict:
  w = {'crop': 11, 'max_objects': None, 'objects': 512, 'teacher_sha256': unseen.TEACHER_SHA256, 'tf32': False,
       'device': 'NVIDIA B200', 'design_sha256': fb.design_sha256(), 'runs': runs}
  w.update(changes)
  return w


class SettingsTest(unittest.TestCase):

  def test_reference_runs_have_the_timing_study_s_settings(self):
    import time_crop11  # pylint: disable=g-import-not-at-top
    for name, (earlier, _) in fb.EARLIER.items():
      _, fmt, width, box_only = time_crop11.RUNS[earlier]
      self.assertTrue(box_only)
      self.assertEqual((fmt, width, time_crop11.SCALE.get(earlier, 1.0)), fb.RUNS[name], name)

  def test_test_run_is_float32_at_the_other_tracers_width(self):
    self.assertEqual(fb.RUNS[fb.TEST_RUN], ('fp32', 32, 1.0))
    for n in ('F5', 'S10', 'P2', 'Q2'):
      self.assertEqual(unseen.RUN_SETTINGS[n][2], fb.RUNS[fb.TEST_RUN][1], n)

  def test_design_lists_every_run(self):
    text = fb.DESIGN.read_text()
    for n, (fmt, width, scale) in fb.RUNS.items():
      f = 'float32' if fmt == 'fp32' else f'`{fmt}`'
      s = '1' if scale == 1.0 else '1/16'
      self.assertIn(f'| {n} | {f} | {width} | {s} |', text)
    self.assertIn('**Width fixed now: 32**', text)

  def test_unknown_run_is_refused(self):
    with self.assertRaises(ValueError):
      fb.check_run('F1')

  def test_job_script_runs_the_two_parts(self):
    s = (ROOT / 'cluster' / 'mit' / 'fp32_batch.sbatch').read_text()
    self.assertIn('examples/time_widths.py', s)
    self.assertIn('examples/run_unseen_fp32x32.py', s)
    self.assertIn('K=$((41 + SLURM_ARRAY_TASK_ID))', s)
    self.assertIn('--array=0-9', s)
    self.assertIn('widths-each)', s)
    self.assertIn('for r in ${RUNS:-H16x1 H16x8 H16x32 H16x64}; do', s)
    self.assertIn('--runs "$r"', s)

  def test_runner_refuses_other_crops(self):
    import run_unseen  # pylint: disable=g-import-not-at-top
    for k, smoke in ((12, None), (40, None), (11, None), (41, 20), (12, 20)):
      with self.assertRaises(ValueError, msg=(k, smoke)):
        run_unseen.load_crop(k, smoke)


class FingerprintTest(unittest.TestCase):

  def test_as_the_timing_evidence(self):
    records = [rec(1), rec(2, steps=3)]
    rows = [[1, 10, 'm1', 'h1'], [2, 3, 'm2', 'h2']]
    self.assertEqual(fb.fingerprint(records), hashlib.sha256(json.dumps(rows).encode()).hexdigest())

  def test_identity(self):
    a = [rec(1), rec(2), rec(3)]
    b = [rec(1), rec(2, hist='x'), rec(3, steps=11)]
    self.assertEqual(fb.identity(a, b), {'objects': 3, 'identical': 1, 'same_mask': 3, 'same_steps': 2})
    with self.assertRaises(ValueError):
      fb.identity(a, [rec(1), rec(3), rec(2)])


class WidthsReportTest(unittest.TestCase):

  def test_split_and_identity(self):
    base = [rec(i) for i in range(4)]
    secs = {'F32x1': 1200.0, 'F32x8': 600.0, 'F32x32': 500.0, 'F32x64': 520.0,
            'H16x1': 600.0, 'H16x8': 250.0, 'H16x32': 230.0, 'H16x64': 240.0}
    runs = {n: run(s, copy.deepcopy(base), n) for n, s in secs.items()}
    runs['F32x64']['records'][0]['steps'] = 99
    r = fb.widths_report(runs)
    self.assertEqual(r['identity_vs_width_1']['F32x8']['identical'], 4)
    self.assertEqual(r['identity_vs_width_1']['F32x64']['identical'], 3)
    self.assertAlmostEqual(r['batching_speedup']['F32x32'], 2.4)
    self.assertAlmostEqual(r['precision_speedup_at_width']['32'], 500 / 230)
    self.assertAlmostEqual(r['precision_speedup_at_width']['1'], 2.0)
    self.assertAlmostEqual(r['fast_ffn_over_fp32_one_at_a_time'], 1200 / 230)
    self.assertAlmostEqual(r['precision_speedup_per_step_at_width']['32'], 500 / 230)
    self.assertAlmostEqual(r['batching_speedup_per_step']['F32x64'], 1200 / 520)
    self.assertEqual(r['fastest_fp32_width'], 32)
    self.assertFalse(r['same_as_timing_study']['F32x1']['same'])

  def test_per_step_split_differs_when_step_counts_do(self):
    runs = {'F32x32': run(500.0, [rec(i, steps=10) for i in range(4)], 'F32x32', setup=20.0),
            'H16x32': run(230.0, [rec(i, steps=12) for i in range(4)], 'H16x32', setup=20.0)}
    r = fb.widths_report(runs)
    self.assertAlmostEqual(r['precision_speedup_at_width']['32'], 500 / 230)
    self.assertAlmostEqual(r['precision_speedup_per_step_at_width']['32'], (480 / 40) / (210 / 48))
    self.assertEqual(r['runs']['H16x32']['steps'], 48)

  def test_partial_job(self):
    r = fb.widths_report({'F32x1': run(10.0, [rec(1)])})
    self.assertEqual(r['identity_vs_width_1'], {})
    self.assertNotIn('fast_ffn_over_fp32_one_at_a_time', r)
    self.assertEqual(len(r['missing_runs']), 7)

  def test_widths_file_checks(self):
    runs = {'F32x1': run(1.0, [rec(i) for i in range(512)], 'F32x1'),
            'F32x8': run(1.0, [rec(i) for i in range(512)], 'F32x8')}
    fb.check_widths_file(widths_file(runs))
    wrong = copy.deepcopy(runs)
    wrong['F32x8']['summary']['width'] = 32
    short = copy.deepcopy(runs)
    short['F32x8']['records'] = short['F32x8']['records'][:1]
    reordered = copy.deepcopy(runs)
    reordered['F32x8']['records'] = reordered['F32x8']['records'][::-1]
    for w in (widths_file(runs, crop=12), widths_file(runs, max_objects=20), widths_file(runs, tf32=True),
              widths_file(runs, device='Tesla V100'), widths_file(runs, teacher_sha256='x'),
              widths_file(runs, design_sha256='x'), widths_file(wrong), widths_file(short), widths_file(reordered)):
      with self.assertRaises(ValueError):
        fb.check_widths_file(w)

  def test_merge_replaces_reruns_and_keeps_order(self):
    ids = range(512)
    first = widths_file({'F32x1': run(100.0, [rec(i) for i in ids], 'F32x1'),
                         'H16x1': run(90.0, [rec(i) for i in ids], 'H16x1'),
                         'H16x8': run(30.0, [rec(i) for i in ids], 'H16x8')})
    each = {n: widths_file({n: run(t, [rec(i, mask='x' if n == 'H16x8' else 'm') for i in ids], n)})
            for n, t in (('H16x1', 50.0), ('H16x8', 20.0), ('H16x32', 15.0), ('H16x64', 14.0))}
    again = widths_file({'H16x1': run(40.0, [rec(i) for i in ids], 'H16x1')})
    files = [('widths/widths.json', first)] + [(f'{n}/widths.json', w) for n, w in each.items()]
    m = fb.merge_widths(files + [('H16x1-again/widths.json', again)])
    self.assertEqual(list(m['runs']), ['F32x1', 'H16x1', 'H16x8', 'H16x32', 'H16x64'])
    self.assertEqual(m['source'], {'F32x1': 'widths/widths.json', 'H16x1': 'H16x1-again/widths.json',
                                   'H16x8': 'H16x8/widths.json', 'H16x32': 'H16x32/widths.json',
                                   'H16x64': 'H16x64/widths.json'})
    self.assertEqual(m['runs']['H16x1']['summary']['crop_seconds'], 40.0)
    self.assertEqual([(x['file'], x['summary']['crop_seconds']) for x in m['replaced']['H16x1']],
                     [('widths/widths.json', 90.0), ('H16x1/widths.json', 50.0)])
    self.assertEqual(m['replaced']['H16x1'][0]['identity_vs_replacing_run']['identical'], 512)
    self.assertEqual(m['replaced']['H16x8'][0]['identity_vs_replacing_run']['same_mask'], 0)
    self.assertEqual(set(m['replaced']), {'H16x1', 'H16x8'})
    r = fb.widths_report(m['runs'])
    self.assertEqual(r['precision_speedup_at_width']['1'], 100.0 / 40.0)
    self.assertEqual(r['batching_speedup']['H16x8'], 40.0 / 20.0)
    c = fb.cross_job(m['source'])
    self.assertEqual(c['files'], 5)
    self.assertIn('precision_speedup_at_width/1', c['comparisons_across_files'])
    self.assertIn('batching_speedup/H16x8', c['comparisons_across_files'])
    self.assertNotIn('batching_speedup/F32x8', c['comparisons_across_files'])  # F32x8 is not present
    self.assertIn('own process', c['statement'])
    self.assertIn('possibly on different nodes', c['statement'])
    one = fb.cross_job({'F32x1': 'a', 'F32x8': 'a'})
    self.assertEqual((one['files'], one['comparisons_across_files']), (1, []))
    with self.assertRaises(ValueError):  # two inputs with the same label
      fb.merge_widths([('widths/widths.json', first), ('widths/widths.json', each['H16x1'])])
    with self.assertRaises(ValueError):  # a rerun file left out: H16x8 would still come from the first job
      fb.merge_widths([f for f in files if f[0] != 'H16x8/widths.json'])

  def test_single_run_report(self):
    for n in ('H16x1', 'H16x8'):
      r = fb.widths_report({n: run(10.0, [rec(1)], n)})
      self.assertEqual(r['identity_vs_width_1'], {})
      self.assertEqual(len(r['missing_runs']), 7)
      self.assertNotIn('fastest_fp32_width', r)

  def test_merge_refuses_other_objects_or_a_bad_file(self):
    first = widths_file({'F32x1': run(1.0, [rec(i) for i in range(512)], 'F32x1')})
    other = widths_file({'H16x1': run(1.0, [rec(i) for i in range(1, 513)], 'H16x1')})
    with self.assertRaises(ValueError):
      fb.merge_widths([('a', first), ('b', other)])
    with self.assertRaises(ValueError):
      fb.merge_widths([('a', first), ('b', widths_file(other['runs'], tf32=True))])


class TestReportTest(unittest.TestCase):

  def setUp(self):
    self.ids = {k: list(range(k * 10, k * 10 + 4)) for k in CROPS}
    self.f32 = {k: result_file(k, self.ids[k], 400.0) for k in CROPS}
    self.f1 = {k: result_file(k, self.ids[k], 1000.0, name='F1') for k in CROPS}
    self.f5 = {k: result_file(k, self.ids[k], 200.0, name='F5') for k in CROPS}
    self.sn = {'segneuron_b200': {str(k): {'end_to_end': 100.0, 'without_startup': 80.0} for k in CROPS}}

  def test_split(self):
    r = fb.test_report(self.f32, self.f1, self.f5, self.sn)
    self.assertAlmostEqual(r['batching_speedup_fp32'], 2.5)
    self.assertAlmostEqual(r['precision_speedup_at_32'], 2.0)
    self.assertAlmostEqual(r['fast_ffn_over_fp32_one_at_a_time'], 5.0)
    self.assertEqual(r['identical_objects'], 40)
    self.assertEqual(r['objects'], 40)
    be = r['segneuron_break_even']['end_to_end']
    self.assertEqual(be, unseen_b200.compare({k: 100.0 for k in CROPS}, {k: 400.0 for k in CROPS},
                                             {k: 4 for k in CROPS}, 'float32 FFN, 32 at once'))
    self.assertAlmostEqual(be['totals']['break_even_per_crop'], 1.0)

  def test_changed_trace_is_counted(self):
    self.f32[41]['run']['records'][0]['mask_sha256'] = 'other'
    self.f32[42]['run']['records'][0]['precision'] = 0.5
    r = fb.test_report(self.f32, self.f1, self.f5, self.sn)
    self.assertEqual(r['identity_vs_F1'][41]['identical'], 3)
    self.assertEqual(r['identical_objects'], 39)
    self.assertEqual(r['same_mask_objects'], 39)
    self.assertEqual(r['changes_vs_F1']['new_merges'], 1)
    self.assertEqual(r['runaways_vs_F1']['runaway_traces'], 0)

  def test_per_step_split_uses_each_run_s_steps(self):
    self.f5 = {k: result_file(k, self.ids[k], 200.0, name='F5', steps=20) for k in CROPS}
    r = fb.test_report(self.f32, self.f1, self.f5, self.sn)
    self.assertAlmostEqual(r['per_crop'][41]['precision_speedup_at_32'], 2.0)
    self.assertAlmostEqual(r['per_crop'][41]['precision_speedup_at_32_per_step'], 4.0)

  def test_missing_crop_is_reported_without_totals(self):
    del self.f32[50]
    r = fb.test_report(self.f32, self.f1, self.f5, self.sn)
    self.assertEqual(r['missing_crops'], [50])
    self.assertEqual(sorted(r['per_crop']), list(CROPS[:-1]))
    self.assertNotIn('total_seconds', r)
    self.assertNotIn('segneuron_break_even', r)

  def test_refuses_wrong_files(self):
    good = result_file(41, self.ids[41], 1.0)
    fb.check_test_file(good, 41, self.f1[41])
    bad = [dict(good, smoke_objects=20), dict(good, crop=42), dict(good, tf32=True), dict(good, design_sha256='x'),
           dict(good, run_name='F1'), dict(good, device='Tesla V100-SXM2-16GB'), dict(good, crop_sha256={'raw': 'x'}),
           dict(good, object_ids=self.ids[41][::-1])]
    for key, value in (('width', 8), ('network', 'student'), ('box_only', False), ('objects', 3)):
      s = copy.deepcopy(good)
      s['run']['summary'][key] = value
      bad.append(s)
    other = copy.deepcopy(good)
    other['run']['records'] = other['run']['records'][::-1]
    bad.append(other)
    for x in bad:
      with self.assertRaises(ValueError):
        fb.check_test_file(x, 41, self.f1[41])
    self.f32[41] = bad[0]
    with self.assertRaises(ValueError):
      fb.test_report(self.f32, self.f1, self.f5, self.sn)


class JudgeTest(unittest.TestCase):

  def test_refuses_test_files_that_are_not_the_pinned_ones(self):
    import judge_fp32_batch  # pylint: disable=g-import-not-at-top
    with tempfile.TemporaryDirectory() as d:
      b200 = Path(d)
      pins = {}
      for k in CROPS:
        folder = b200 / f'crop-{k:02d}'
        folder.mkdir()
        (folder / 'F1.json').write_text(json.dumps({'crop': k}))
        pins[f'b200/crop-{k:02d}/F1.json'] = hashlib.sha256((folder / 'F1.json').read_bytes()).hexdigest()
      self.assertEqual(judge_fp32_batch.load_pinned(b200, 'F1', pins)[41], {'crop': 41})
      (b200 / 'crop-45' / 'F1.json').write_text(json.dumps({'crop': 45, 'changed': True}))
      with self.assertRaises(ValueError):
        judge_fp32_batch.load_pinned(b200, 'F1', pins)

  def test_labels_tell_files_apart(self):
    import judge_fp32_batch  # pylint: disable=g-import-not-at-top
    with tempfile.TemporaryDirectory() as d:
      paths = [Path(d) / 'job1' / 'widths' / 'widths.json', Path(d) / 'job2' / 'widths' / 'widths.json',
               Path(d) / 'job2' / 'widths-each' / 'H16x1' / 'widths.json']
      self.assertEqual(judge_fp32_batch.labels(paths),
                       ['job1/widths/widths.json', 'job2/widths/widths.json', 'job2/widths-each/H16x1/widths.json'])
      self.assertEqual(judge_fp32_batch.labels(paths[:1]), ['widths/widths.json'])

  def test_pins_cover_the_test_s_f1_and_f5_files(self):
    pins = json.loads((ROOT / 'docs' / 'unseen_evidence.json').read_text())['result_files_sha256']
    for k in CROPS:
      for r in ('F1', 'F5'):
        self.assertIn(f'b200/crop-{k:02d}/{r}.json', pins)


if __name__ == '__main__':
  unittest.main()
