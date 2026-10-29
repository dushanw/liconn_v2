"""The recorded FFN-against-SegNeuron run (docs/headtohead_results.md) matches its evidence."""

import hashlib
import json
from pathlib import Path
import unittest

from liconn_v2 import data
from liconn_v2 import headtohead as hh

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'headtohead_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
RUN = RUNS / EVIDENCE['run']
TOL = EVIDENCE['tolerance']
STOL = EVIDENCE['seconds_tolerance']


def _sha(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 24), b''):
      h.update(chunk)
  return h.hexdigest()


def _json(name):
  return json.loads((RUN / name).read_text())


class HeadToHeadEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    staged = dict(reversed(l.split()) for l in (RUN / 'source_sha256.txt').read_text().splitlines())
    self.assertEqual(set(staged), set(EVIDENCE['run_source_sha256']))
    for rel in ('liconn_v2/headtohead.py', 'examples/run_headtohead.py', 'cluster/wahab/headtohead.sbatch',
                'liconn_v2/fasttrace.py', 'liconn_v2/benchmark.py', 'liconn_v2/vitseg.py'):
      self.assertIn(rel, staged)
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(staged[rel], expected, rel)
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_one_job_one_node(self):
    log = (RUN / f"{EVIDENCE['slurm_job']}.out").read_text().splitlines()
    self.assertEqual(log[0], f"headtohead on {EVIDENCE['node']}")
    self.assertIn('best.ckpt: OK', log[1])  # the fine-tuned checkpoint's sha256 was checked first
    f, s = _json('ffn_crop12.json'), _json('segneuron_crop12.json')
    for r in (f, s):
      self.assertEqual(r['platform']['node'], EVIDENCE['node'])
      self.assertEqual(r['platform']['slurm_job'], str(EVIDENCE['slurm_job']))
      self.assertEqual(r['crop'], hh.CROP)
    self.assertEqual(f['variant'], EVIDENCE['ffn_variant'])
    self.assertEqual(s['threshold'], EVIDENCE['segneuron_threshold'])
    self.assertIn('Tesla V100', (RUN / 'gpu.txt').read_text())
    self.assertEqual(s['inference'], _json('infer/inference.json'))
    ft = json.loads((ROOT / 'docs' / 'segneuron_finetune_evidence.json').read_text())
    self.assertEqual(s['inference']['checkpoint']['sha256'], ft['segneuron']['finetuned_checkpoint_sha256'])
    self.assertEqual(EVIDENCE['segneuron_checkpoint_sha256'], ft['segneuron']['finetuned_checkpoint_sha256'])
    self.assertEqual(s['threshold'], ft['ours']['threshold'])

  def test_same_objects_and_seeds(self):
    f, s = _json('ffn_crop12.json'), _json('segneuron_crop12.json')
    ids = [r['object_id'] for r in f['records']]
    self.assertEqual(len(ids), EVIDENCE['objects'])
    self.assertEqual(len(set(ids)), EVIDENCE['objects'])
    self.assertEqual(ids, [r['object_id'] for r in s['objects']])
    ft_run = json.loads((ROOT / 'docs' / 'segneuron_finetune_evidence.json').read_text())['run']
    ft = json.loads((RUNS / ft_run / 'segneuron_ft_crop12.json').read_text())
    self.assertEqual(ids, [r['object_id'] for r in ft['ours']['objects']])

  def test_accuracy(self):
    f, s = _json('ffn_crop12.json'), _json('segneuron_crop12.json')
    for r in f['records']:
      self.assertAlmostEqual(r['f1'], hh.f1(r['precision'], r['recall']), delta=1e-12)
    p, e = hh.paired(f['records'], s['objects']), EVIDENCE['accuracy']
    self.assertEqual(p, _json('headtohead_crop12.json')['accuracy'])
    for name in ('ffn', 'segneuron'):
      for k, v in e[name].items():
        self.assertAlmostEqual(p[name][k], v, delta=TOL, msg=f'{name} {k}')
    for k in ('ffn_higher', 'segneuron_higher', 'equal', 'segneuron_seeds_in_background', 'ffn_merges',
              'ffn_objects_with_merge', 'ffn_merged_voxels', 'segneuron_merges', 'segneuron_objects_with_merge',
              'segneuron_merged_voxels'):
      self.assertEqual(p[k], e[k], k)
    self.assertEqual(p['ffn_higher'] + p['segneuron_higher'] + p['equal'], EVIDENCE['objects'])
    self.assertAlmostEqual(p['mean_f1_difference'], e['mean_f1_difference'], delta=TOL)
    for got, want in zip(p['f1_difference_quantiles']['value'], e['f1_difference_quantiles']):
      self.assertAlmostEqual(got, want, delta=TOL)
    for k in ('ffn_mean_f1_where_segneuron_seed_found', 'segneuron_mean_f1_where_seed_found'):
      self.assertAlmostEqual(p[k], e[k], delta=TOL, msg=k)
    self.assertEqual(sum(r['f1'] >= 0.9 for r in f['records']), e['ffn_objects_f1_at_least_0.9'])
    self.assertEqual(sum(r['f1'] >= 0.9 for r in s['objects']), e['segneuron_objects_f1_at_least_0.9'])
    # merged_voxels counts every other-object voxel; the merge pieces only those of >= 1,000 voxels
    self.assertEqual(sum(v for r in f['records'] for _, v in r['merges']), e['ffn_voxels_in_merges_of_at_least_1000'])
    self.assertEqual(sum(v for r in s['objects'] for _, v in r['merges']),
                     e['segneuron_voxels_in_merges_of_at_least_1000'])
    self.assertEqual(f['overlap'], EVIDENCE['ffn_overlap'])
    self.assertEqual(f['overlap']['sum_of_mask_voxels'], sum(r['mask_voxels'] for r in f['records']))

  def test_speed(self):
    f, s, e = _json('ffn_crop12.json'), _json('segneuron_crop12.json'), EVIDENCE['speed']
    sp = hh.speed(f['records'], f['setup_seconds'], s['inference_wall_seconds'],
                  s['inference']['inference_seconds'], s['post_seconds'])
    self.assertEqual(sp, _json('headtohead_crop12.json')['speed'])
    for name in ('ffn', 'segneuron'):
      for k, v in e[name].items():
        self.assertAlmostEqual(sp[name][k], v, delta=STOL, msg=f'{name} {k}')
    for k, v in e['post_breakdown'].items():
      self.assertAlmostEqual(s[k], v, delta=STOL, msg=k)
    self.assertAlmostEqual(s['post_seconds'], sum(s[k] for k in e['post_breakdown']), delta=1e-9)
    self.assertAlmostEqual(sp['ffn_over_segneuron'], e['ffn_over_segneuron'], delta=TOL)
    for k, v in e['break_even_objects'].items():
      if k == 'end_to_end_with_ffn_setup':  # FFN's one-time setup counted: SN total - setup = n x mean wall
        got = (sp['segneuron']['total_seconds'] - sp['ffn']['setup_seconds']) / sp['ffn']['wall_seconds_per_object']
      else:
        got = sp['break_even_objects'][k]
      self.assertAlmostEqual(got, v, delta=0.05, msg=k)
    self.assertAlmostEqual(f['compile_seconds'], e['ffn_compile_seconds'], delta=STOL)
    for r in f['records']:  # the wall time contains the tracer's own clock
      self.assertLessEqual(r['trace_seconds'], r['wall_seconds'])

  def test_reproduces_the_recorded_runs(self):
    c, e = _json('headtohead_crop12.json')['checks'], EVIDENCE['checks']
    self.assertTrue(c['ffn_trace_matches_recorded'])
    self.assertEqual(c['segneuron_max_abs_affinity_change_vs_recorded'], 0.0)
    self.assertEqual(c['segneuron_max_abs_foreground_change_vs_recorded'], 0.0)
    self.assertEqual(c['segneuron_max_abs_f1_change_vs_recorded'], 0.0)
    for k, v in e.items():
      self.assertAlmostEqual(float(c[k]), float(v), delta=TOL, msg=k)
    # the check object's FFN trace against the recorded many-seed record, repeated here
    rec = [r for r in json.loads((RUNS / 'manyseed-fp16_head32-20261022' / 'crop-12' / 'benchmark.json').read_text())
           ['records'] if r['object_id'] == hh.CHECK_OBJECT][0]
    mine = [r for r in _json('ffn_crop12.json')['records'] if r['object_id'] == hh.CHECK_OBJECT][0]
    for k in ('steps', 'canvas_sha256', 'history_sha256', 'recall', 'precision', 'seed_zyx'):
      self.assertEqual(mine[k], rec[k], k)


if __name__ == '__main__':
  unittest.main()
