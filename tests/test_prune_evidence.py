"""The recorded pruned-FFN runs (docs/prune_results.md) match their evidence.

Every number in docs/prune_evidence.json is recomputed here from the copied MIT records in the data
cache (per-object records where they exist), not compared with the extract itself.
"""

import hashlib
import json
from pathlib import Path
import re
import unittest

from liconn_v2 import data

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'prune_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
RUN = RUNS / EVIDENCE['run']
REL = 1e-5  # evidence values are rounded to 6 significant digits


def _sha(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 24), b''):
      h.update(chunk)
  return h.hexdigest()


def _json(name):
  return json.loads((RUN / name).read_text())


def _f1(p, r):
  return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _staged(job):
  return dict(reversed(l.split()) for l in (RUN / f'source-{job}.txt').read_text().splitlines() if l.strip())


def _launch(job):
  """Fields of the first log line, as printed by cluster/mit/distil.sbatch before it runs."""
  line = (RUN / f'liconn-distil-{job}.out').read_text().splitlines()[0]
  head, tail = line.split(f' job {job} ')
  fields = dict(re.findall(r'(\w+)=(.*?)(?= \w+=|$)', head))
  return fields, tail


def _expected_output(f):
  """Output directory that cluster/mit/distil.sbatch writes for these launch fields."""
  mode = f['MODE']
  if mode == 'prune-probe':
    return f"probe-{f['NAME']}"
  if mode == 'prune-train':
    return f"train-{f['NAME']}"
  if mode == 'trace':
    return f"trace-{f['NAME']}" + (f"-{f['CKPT']}" if f['CKPT'] else '')
  if mode == 'onpolicy':
    return f['OUTNAME'] or 'onpolicy'
  if mode == 'timing':
    return f"timing-{f['NAME']}" + (f"-{f['TAG']}" if f['TAG'] else '')
  raise ValueError(mode)


def _last_json(job):
  """The last top-level JSON object the job printed (one line or several)."""
  lines = (RUN / f'liconn-distil-{job}.out').read_text().splitlines()
  starts = [i for i, l in enumerate(lines) if l.startswith('{')]
  for i in reversed(starts):
    try:
      return json.loads('\n'.join(lines[i:]))
    except json.JSONDecodeError:
      continue
  raise AssertionError(f'no JSON at the end of job {job} log')


def _same(x, y):
  return (x['object_id'], x['steps'], x['mask_sha256'], x['history_sha256']) == \
         (y['object_id'], y['steps'], y['mask_sha256'], y['history_sha256'])


class PruneEvidenceTest(unittest.TestCase):

  def close(self, a, b, msg=None):
    self.assertAlmostEqual(a, b, delta=REL * max(1.0, abs(b)), msg=msg)

  def run_summary(self, rec, ev, extra_seconds, msg):
    """Per-run numbers recomputed from per-object records."""
    n = len(rec)
    self.assertEqual(n, ev['objects'], msg)
    steps = sum(x['steps'] for x in rec)
    wall = sum(x['wall_seconds'] for x in rec) if all(x['wall_seconds'] is not None for x in rec) else None
    self.assertEqual(steps, ev['steps'], msg)
    self.close(sum(_f1(x['precision'], x['recall']) for x in rec) / n, ev['mean_f1'], msg)
    self.close(sum(x['recall'] for x in rec) / n, ev['mean_recall'], msg)
    self.close(sum(x['precision'] for x in rec) / n, ev['mean_precision'], msg)
    self.assertEqual(sum(x['merges'] for x in rec), ev['merges_ge_1000'], msg)
    self.assertEqual(sum(x['merged_voxels'] for x in rec), ev['merged_voxels'], msg)
    if wall is not None:
      self.close(wall, ev['wall_seconds'], msg)
    self.close(1000 * ev['wall_seconds'] / steps, ev['ms_per_step'], msg)
    self.close(ev['wall_seconds'] + extra_seconds, ev['crop_seconds'], msg)

  def test_code_that_ran_is_the_checked_code(self):
    jobs = sorted(set(EVIDENCE['jobs'].values()))
    first = _staged(jobs[0])
    same = all(_staged(j) == first for j in jobs[1:])
    self.assertEqual(same, EVIDENCE['all_jobs_same_source'])
    self.assertTrue(same)
    for rel, expected in EVIDENCE['source_sha256'].items():
      self.assertEqual(first[rel], expected, rel)
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_jobs_made_the_stated_outputs(self):
    jobs = {str(j) for j in EVIDENCE['jobs'].values()}
    self.assertEqual(jobs | {'27684986'}, set(EVIDENCE['job_outputs']))
    for job, out in EVIDENCE['job_outputs'].items():
      fields, _ = _launch(job)
      self.assertEqual(_expected_output(fields), out, job)
    for job in ('27770944', '27684986'):
      self.assertEqual(_last_json(job)['crop_seconds'],
                       _json(f"{EVIDENCE['job_outputs'][job]}/timing.json")['crop_seconds'], job)
    self.assertEqual(EVIDENCE['job_outputs'][str(EVIDENCE['timing_job_E']['slurm_job'])], 'timing-dilated-r1-e')
    self.assertEqual(EVIDENCE['job_outputs'][str(EVIDENCE['timing_job_E']['job_D_student_float32_S2']['slurm_job'])],
                     'timing-dilated-r1-d')
    for job, out in EVIDENCE['job_outputs'].items():
      fields, _ = _launch(job)
      if fields['MODE'] == 'trace':
        self.assertEqual(_last_json(job), _json(f'{out}/student_crop11.json')['summary'], job)
      elif fields['MODE'] == 'prune-probe':
        self.assertEqual(_last_json(job)['probe_seconds_per_step'],
                         _json(f'{out}/train_log.json')['probe_seconds_per_step'], job)
      elif fields['MODE'] == 'prune-train' and job != '27722462':  # 27722462 paused for the resume job
        self.assertEqual(_last_json(job)['best'], _json(f'{out}/train_log.json')['best'], job)
    resumed = _launch('27722463')[0]
    self.assertEqual(resumed['RESUME'], '1')
    self.assertEqual(_launch('27722462')[0]['RESUME'], '')

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_crop11_traces(self):
    traces = dict(EVIDENCE['crop11_traces'], **{'trace-dilated-r1': EVIDENCE['factorised_student_crop11_trace']})
    for name, ev in traces.items():
      d = _json(f'{name}/student_crop11.json')
      rec = d['records']
      n = len(rec)
      self.assertEqual(n, ev['objects'])
      self.assertEqual(d['summary']['checkpoint_step'], ev['checkpoint_step'])
      self.assertTrue(d['summary']['checkpoint'].endswith(ev['checkpoint']))
      self.close(sum(_f1(x['precision'], x['recall']) for x in rec) / n, ev['mean_f1'], name)
      self.close(sum(x['recall'] for x in rec) / n, ev['mean_recall'])
      self.close(sum(x['precision'] for x in rec) / n, ev['mean_precision'])
      self.close(sum(x['teacher_f1'] for x in rec) / n, ev['teacher_mean_f1'])
      self.close(sum(x['vs_teacher']['f1'] for x in rec) / n, ev['mean_f1_vs_teacher_mask'], name)
      self.assertEqual(sum(1 for x in rec if _f1(x['precision'], x['recall']) > x['teacher_f1']),
                       ev['objects_higher_than_teacher'], name)
      steps, teacher = sum(x['steps'] for x in rec), sum(x['teacher_steps'] for x in rec)
      self.assertEqual(steps, ev['steps'])
      self.assertEqual(teacher, ev['teacher_steps'])
      self.close(steps / teacher, ev['steps_ratio'])
      self.assertEqual(sum(len(x['merges']) for x in rec), ev['merges_ge_1000'])
      self.assertEqual(sum(1 for x in rec if x['steps'] >= 3 * max(1, x['teacher_steps'])), ev['runaway_traces_3x'])

  def test_selection_rule(self):
    tr = EVIDENCE['crop11_traces']
    pruned = _json('timing-dilated-r1-e/timing.json')['pruned']
    for net in ('p8', 'p4'):
      candidates = [k for k in tr if k.startswith(f'trace-{net}')]
      self.assertEqual(len(candidates), 3)
      f1 = {k: sum(_f1(x['precision'], x['recall']) for x in rec) / len(rec)
            for k in candidates for rec in [_json(f'{k}/student_crop11.json')['records']]}
      chosen = max(f1, key=f1.get)
      self.assertEqual(chosen, EVIDENCE['selected'][net])
      summary = _json(f'{chosen}/student_crop11.json')['summary']
      self.assertEqual(pruned[net]['checkpoint'], summary['checkpoint'])  # job E timed the chosen checkpoint
      self.assertEqual(pruned[net]['step'], summary['checkpoint_step'])
      ev = EVIDENCE['timing_job_E']['pruned'][net]
      self.assertTrue(pruned[net]['checkpoint'].endswith(ev['checkpoint']))
      for k in ('step', 'keep', 'dilations'):
        self.assertEqual(pruned[net][k], ev[k], (net, k))

  def test_probes(self):
    for name, ev in EVIDENCE['probes'].items():
      l = _json(f'{name}/train_log.json')
      self.assertEqual(int(l['args']['steps']), ev['steps'])
      self.assertEqual(l['args']['probe'], 'True')
      self.close(l['probe_seconds_per_step'], ev['seconds_per_step'])
      self.close(l['peak_memory_gb'], ev['peak_memory_gb'])

  def test_training(self):
    for name, ev in EVIDENCE['training'].items():
      l = _json(f'{name}/train_log.json')
      for k in ('train_samples', 'extra_samples', 'valid_samples', 'parameters'):
        self.assertEqual(l[k], ev[k], (name, k))
      self.close(l['gflop_per_box'], ev['gflop_per_box'])
      for k in ('keep', 'dilations', 'init', 'extra'):
        self.assertEqual(l['args'][k], ev[k], (name, k))
      self.assertEqual(int(l['args']['steps']), ev['steps'])
      valid = l['valid']
      candidates = valid[1:] if name.endswith('-r1') else valid  # round 1's step 0 cannot be best (design)
      best = min(candidates, key=lambda v: v['loss'])
      self.assertEqual(best['step'], ev['best_step'], name)
      self.assertEqual(l['best']['step'], ev['best_step'], name)
      self.close(best['loss'], ev['best_valid_loss'])
      self.close(best['interior_agreement'], ev['best_interior_agreement'])
      self.close(best['shell_agreement'], ev['best_shell_agreement'])
      self.assertEqual(valid[0]['step'], 0)
      self.close(valid[0]['loss'], ev['step0_valid_loss'])
      self.assertEqual(valid[-1]['step'], ev['last_step'])
      self.assertEqual(ev['last_step'], ev['steps'])
      self.close(valid[-1]['loss'], ev['last_valid_loss'])
      self.assertEqual(len(ev['job_seconds']), len(l.get('job_seconds', [l['total_seconds']])))
      for a, b in zip(l.get('job_seconds', [l['total_seconds']]), ev['job_seconds']):
        self.close(a, b)
      self.assertEqual(l.get('resumed_at', []), ev['resumed_at'])
      self.assertEqual(set(ev), {'train_samples', 'extra_samples', 'valid_samples', 'parameters', 'gflop_per_box',
                                 'steps', 'keep', 'dilations', 'init', 'extra', 'step0_valid_loss', 'best_step',
                                 'best_valid_loss', 'last_step', 'last_valid_loss', 'best_interior_agreement',
                                 'best_shell_agreement', 'job_seconds', 'resumed_at'}, name)  # all checked here
      if name.endswith('-r1'):
        self.assertGreater(best['step'], 0)
        self.assertTrue(l['args']['init'].endswith(f"{name[:-3]}/best.pt"))
        r0 = _json(f'{name[:-3]}/train_log.json')
        self.close(valid[0]['loss'], r0['best']['loss'])  # round 1 starts from round 0's best

  def test_every_extracted_field_is_checked(self):
    run_fields = {'box_only', 'crop_seconds', 'format', 'mean_f1', 'mean_precision', 'mean_recall', 'merged_voxels',
                  'merges_ge_1000', 'ms_per_step', 'network', 'nonfinite_outputs', 'objects', 'steps',
                  'stopped_at_first_step', 'stream_scale', 'wall_seconds', 'width'}
    t = EVIDENCE['timing_job_E']
    self.assertEqual(set(EVIDENCE), {'what', 'run', 'jobs', 'all_jobs_same_source', 'job_outputs', 'source_sha256',
                                     'outputs_sha256', 'probes', 'training', 'crop11_traces',
                                     'factorised_student_crop11_trace', 'selected', 'timing_job_E'})
    self.assertEqual(set(t), {'slurm_job', 'device', 'torch', 'tf32', 'objects', 'student_checkpoint',
                              'student_checkpoint_step', 'pruned', 'encoder', 'ffn_image_copy_seconds', 'runs',
                              'comparisons', 'repeats_of_job_D', 'job_D_student_float32_S2', 'timed_fp32_vs_trace_job'})
    for k, r in t['runs'].items():
      self.assertEqual(set(r), run_fields | {'speedup_vs_F1'}, k)
    self.assertEqual(set(t['job_D_student_float32_S2']), run_fields | {'slurm_job'})

  def test_timing_runs(self):
    t, ev = _json('timing-dilated-r1-e/timing.json'), EVIDENCE['timing_job_E']
    for k in ('device', 'torch', 'tf32', 'objects'):
      self.assertEqual(t[k], ev[k], k)
    self.assertTrue(t['checkpoint'].endswith(ev['student_checkpoint']))
    self.assertEqual(t['checkpoint_step'], ev['student_checkpoint_step'])
    for k, v in t['encoder'].items():
      self.assertEqual(v, ev['encoder'][k]) if isinstance(v, bool) else self.close(v, ev['encoder'][k])
    self.close(t['ffn_image_copy_seconds'], ev['ffn_image_copy_seconds'])
    ids = [x['object_id'] for x in t['runs']['F1']['records']]
    for k, r in t['runs'].items():
      rec, e = r['records'], ev['runs'][k]
      self.assertEqual([x['object_id'] for x in rec], ids, k)
      extra = t['encoder']['direct_seconds'] if e['network'] == 'student' else t['ffn_image_copy_seconds']
      self.run_summary(rec, e, extra, k)
      s = r['summary']
      for f in ('network', 'format', 'width', 'box_only', 'stopped_at_first_step', 'nonfinite_outputs', 'stream_scale'):
        self.assertEqual(s[f], e[f], (k, f))
      self.close(s['wall_seconds'], e['wall_seconds'], k)  # batched runs time the crop, not each object
      self.close(t['crop_seconds']['F1'] / t['crop_seconds'][k], e['speedup_vs_F1'], k)
    s2, d = ev['job_D_student_float32_S2'], _json('timing-dilated-r1-d/timing.json')
    self.run_summary(d['runs']['S2']['records'], s2, d['encoder']['direct_seconds'], 'S2 (job D)')
    self.close(d['crop_seconds']['S2'], s2['crop_seconds'])
    sd = d['runs']['S2']['summary']
    for f in ('network', 'format', 'width', 'box_only', 'stopped_at_first_step', 'nonfinite_outputs', 'stream_scale'):
      self.assertEqual(sd[f], s2[f], ('S2', f))
    self.close(sd['wall_seconds'], s2['wall_seconds'], 'S2')
    self.assertEqual((s2['network'], s2['format'], s2['width'], s2['stream_scale']), ('student', 'fp32', 1, 1.0))
    self.assertEqual(d['checkpoint'], t['checkpoint'])  # the same student checkpoint in jobs D and E

  def test_comparisons(self):
    t, ev = _json('timing-dilated-r1-e/timing.json'), EVIDENCE['timing_job_E']
    for pair, c in ev['comparisons'].items():
      b, a = pair.split(' - ')
      ra, rb = t['runs'][a]['records'], t['runs'][b]['records']
      d = [_f1(y['precision'], y['recall']) - _f1(x['precision'], x['recall']) for x, y in zip(ra, rb)]
      self.close(sum(d) / len(d), c['mean_f1_change'], pair)
      self.assertEqual(sum(1 for v in d if v < -0.1), c['falls_gt_0.1'], pair)
      self.assertEqual(sum(1 for v in d if v > 0.1), c['rises_gt_0.1'], pair)
      self.assertEqual(sum(1 for x, y in zip(ra, rb) if _same(x, y)), c['identical_traces'], pair)

  def test_repeats_and_reproduced_traces(self):
    t, d = _json('timing-dilated-r1-e/timing.json'), _json('timing-dilated-r1-d/timing.json')
    for k, ev in EVIDENCE['timing_job_E']['repeats_of_job_D'].items():
      a, b = d['runs'][k]['records'], t['runs'][k]['records']
      self.assertEqual(len(b), ev['objects'])
      self.assertEqual(sum(1 for x, y in zip(a, b) if _same(x, y)), ev['identical_traces'], k)
      self.close(d['crop_seconds'][k], ev['crop_seconds_job_D'])
      self.close(t['crop_seconds'][k], ev['crop_seconds_job_E'])
    for pair, ev in EVIDENCE['timing_job_E']['timed_fp32_vs_trace_job'].items():
      run, trace = pair.split(' vs ')
      tr = _json(f'{trace}/student_crop11.json')['records']
      rr = t['runs'][run]['records']
      self.assertEqual([x['object_id'] for x in tr], [x['object_id'] for x in rr])
      self.assertEqual(len(rr), ev['objects'])
      self.assertEqual(sum(1 for x, y in zip(tr, rr) if x['steps'] == y['steps']), ev['same_steps'])
      diff = max(abs(_f1(x['precision'], x['recall']) - y['f1']) for x, y in zip(tr, rr))
      self.assertEqual(diff, ev['max_abs_f1_difference'])
      self.assertEqual(diff, 0.0)


if __name__ == '__main__':
  unittest.main()
