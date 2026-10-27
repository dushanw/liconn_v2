"""The recorded ViT pilot runs (docs/vitseg_results.md) match their evidence."""

import glob
import hashlib
import json
from pathlib import Path
import statistics
import unittest

from liconn_v2 import data
from liconn_v2 import fresh

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'vitseg_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
TOL = EVIDENCE['tolerance']


def _sha(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 24), b''):
      h.update(chunk)
  return h.hexdigest()


def _json(rel):
  return json.loads((RUNS / rel).read_text())


def _f1(p, r):
  return 2 * p * r / (p + r) if p + r else 0.0


class VitsegEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    staged = dict(reversed(l.split()) for l in
                  (RUNS / 'vitseg-train-r3-20261025' / 'source_sha256.txt').read_text().splitlines())
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(staged.get(rel), expected, rel)
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_training_record(self):
    log, ev = _json('vitseg-train-r3-20261025/train_log.json'), EVIDENCE['training']
    self.assertEqual(log['platform']['slurm_job'], str(EVIDENCE['slurm_jobs']['train']))
    self.assertEqual(log['steps'], ev['planned_steps'])
    self.assertEqual(log['stopped'], {'step': ev['stopped_step'], 'reason': ev['stopped_reason']})
    self.assertEqual(log['best_step'], ev['best_step'])
    self.assertAlmostEqual(log['best_validation_loss'], ev['best_validation_loss'], delta=TOL)
    self.assertEqual(len(log['skipped_steps']), ev['skipped_steps'])
    self.assertEqual(min(log['skipped_steps']), ev['first_skipped_step'])
    self.assertEqual(log['unstable'], ev['unstable'])
    best = min((v for v in log['validation'] if v['loss'] == v['loss']), key=lambda v: v['loss'])
    self.assertEqual(best['step'], ev['best_step'])  # the selected checkpoint is the lowest validation loss
    self.assertEqual(_sha(RUNS / 'vitseg-train-r3-20261025' / 'params.msgpack'), log['params_sha256'])

  def test_tuning_record(self):
    tu, ev = _json('vitseg-tune-r3-s66000-20261026/vitseg_tuning.json'), EVIDENCE['tuning']
    self.assertEqual(tu['platform']['slurm_job'], str(EVIDENCE['slurm_jobs']['tune']))
    self.assertTrue(tu['params'].endswith('vitseg-train-r3-20261025/params.msgpack'))
    self.assertEqual(tu['objects'], ev['objects'])
    self.assertEqual(tu['threshold'], ev['threshold'])
    best = max(tu['table'], key=lambda r: (r['mean_f1'], r['threshold']))
    self.assertEqual(best['threshold'], ev['threshold'])
    for k in ('mean_f1', 'mean_recall', 'mean_precision'):
      self.assertAlmostEqual(best[k], ev[k], delta=TOL, msg=k)
    self.assertAlmostEqual(tu['background']['precision'], ev['background_precision'], delta=TOL)
    self.assertAlmostEqual(tu['background']['recall'], ev['background_recall'], delta=TOL)
    self.assertEqual(tu['timing']['fragments'], ev['fragments'])

  def test_breakdown_record(self):
    b = _json('vitseg-breakdown-s66000-20261026/breakdown.json')
    self.assertEqual(len(b['objects']), EVIDENCE['tuning']['objects'])
    for k, v in EVIDENCE['breakdown'].items():
      self.assertAlmostEqual(b['per_object_mean'][k], v, delta=TOL, msg=k)
    self.assertEqual(b['per_object_mean']['seeds_in_background'], EVIDENCE['seeds_in_background'])
    # the breakdown's segmentation at 0.5 is the tuning run's
    self.assertAlmostEqual(b['per_object_mean']['f1'], EVIDENCE['tuning']['mean_f1'], delta=TOL)
    # recall, background and other segments account for each whole object
    for r in b['objects']:
      self.assertAlmostEqual(r['recall'] + r['lost_to_background'] + r['lost_to_other_segments'], 1.0, delta=1e-9)
    for k in ('precision', 'recall'):
      self.assertAlmostEqual(b['neuron_channel'][k], EVIDENCE['neuron_channel'][k], delta=TOL)
    for name, ev in EVIDENCE['hybrids'].items():
      self.assertEqual(b[name]['best_threshold'], ev['best_threshold'], name)
      for k in ('mean_f1', 'mean_recall', 'mean_precision'):
        self.assertAlmostEqual(b[name][k], ev[k], delta=TOL, msg=f'{name} {k}')

  def test_numbers_come_from_the_float_outputs(self):
    """The breakdown and the meshes used the float network outputs; the 1/255-rounded
    aff_uint8.npy is only written as an export and read by no recorded number."""
    d = RUNS / 'vitseg-breakdown-s66000-20261026'
    text = (d / 'breakdown.py').read_text()
    assigns = [l for l in text.splitlines() if l.startswith('aff = ')]
    self.assertEqual(assigns, ['aff = vitseg.predict_affinities(lambda x: jax.device_get(net(x)),'])
    uses = [l for l in text.splitlines() if 'uint8' in l]
    self.assertEqual(len(uses), 1)
    self.assertTrue(uses[0].startswith("np.save(out / 'aff_uint8.npy'"))
    meshes = (d / 'meshes.py').read_text()
    self.assertNotIn('aff_uint8', meshes)
    self.assertIn('vitseg.segment_crop(', meshes)
    # the two jobs agree exactly with each other and with the tuning run to within 1e-6
    # (observed 5e-7; the cause, possibly GPU non-determinism, was not isolated)
    line = [l for l in (d / 'mesh_7452015.out').read_text().splitlines() if l.startswith('mean F1 check')]
    self.assertEqual(len(line), 1)
    mesh_f1 = float(line[0].split()[-1])
    self.assertEqual(_json('vitseg-breakdown-s66000-20261026/breakdown.json')['per_object_mean']['f1'], mesh_f1)
    tuning = _json('vitseg-tune-r3-s66000-20261026/vitseg_tuning.json')
    self.assertAlmostEqual(max(r['mean_f1'] for r in tuning['table']), mesh_f1, delta=1e-6)

  def test_ffn_reference(self):
    ev = EVIDENCE['ffn_reference']
    recs = []
    for f in sorted(glob.glob(str(RUNS / ev['run'] / 'crop-*' / 'benchmark.json'))):
      recs += json.loads(Path(f).read_text())['records']
    self.assertEqual(len(recs), ev['seeds'])
    n = len(recs)
    self.assertAlmostEqual(sum(r['recall'] for r in recs) / n, ev['mean_recall'], delta=5e-4)
    self.assertAlmostEqual(sum(r['precision'] for r in recs) / n, ev['mean_precision'], delta=5e-4)
    self.assertAlmostEqual(sum(_f1(r['precision'], r['recall']) for r in recs) / n, ev['mean_f1'], delta=5e-4)
    self.assertAlmostEqual(sum(r['unlabeled_fraction'] for r in recs) / n, ev['mean_unlabeled_fraction'], delta=5e-4)

  def test_cited_details(self):
    """The secondary numbers cited in docs/vitseg_results.md."""
    d = EVIDENCE['details']
    near = lambda a, b, msg=None: self.assertAlmostEqual(a, b, delta=TOL, msg=msg)
    span = lambda vals, ev, msg: (near(min(vals), ev[0], msg), near(max(vals), ev[1], msg))
    self.assertIn('7452001.out', ' '.join(EVIDENCE['outputs_sha256']))
    self.assertIn(f"{EVIDENCE['slurm_jobs']['breakdown']}.out", ' '.join(EVIDENCE['outputs_sha256']))
    self.assertIn(f"mesh_{EVIDENCE['slurm_jobs']['meshes']}.out", ' '.join(EVIDENCE['outputs_sha256']))
    log = _json('vitseg-train-r3-20261025/train_log.json')
    val = {v['step']: v['loss'] for v in log['validation']}
    for step, loss in d['validation_loss'].items():
      near(val[int(step)], loss, step)
    tu = _json('vitseg-tune-r3-s66000-20261026/vitseg_tuning.json')
    table = {r['threshold']: r for r in tu['table']}
    for t, ev in d['tuning_points'].items():
      for k, v in ev.items():
        near(table[float(t)][k], v, f'{t} {k}')
    near(tu['background']['predicted_fraction'], d['background_predicted_fraction'])
    near(tu['background']['label0_fraction'], d['background_labelled_fraction'])
    b = _json('vitseg-breakdown-s66000-20261026/breakdown.json')
    bands = b['neuron_channel']['missed_neuron_by_distance']
    self.assertEqual(bands[0]['distance_voxels'], [0, 1])
    near(bands[0]['missed_fraction'], d['neuron_missed_at_boundary'])
    span([x['missed_fraction'] for x in bands[1:]], d['neuron_missed_further_in'], 'further in')
    near(bands[0]['false_neuron_fraction'], d['background_called_neuron_at_boundary'])
    self.assertEqual(bands[-1]['distance_voxels'][0], 4)
    near(bands[-1]['false_neuron_fraction'], d['deep_background_called_neuron'])
    self.assertEqual(bands[-1]['background_voxels'], d['deep_background_voxels'])
    nearest = [a for a in b['affinities'] if a['channel'] in ('z1', 'y1', 'x1')]
    longr = [a for a in b['affinities'] if a['channel'] not in ('z1', 'y1', 'x1')]
    self.assertEqual((len(nearest), len(longr)), (3, 3))
    span([a['same_below_0.5'] for a in nearest], d['nearest_same_below_half'], 'same below')
    span([a['diff_at_or_above_0.5'] for a in nearest], d['nearest_diff_at_or_above_half'], 'diff above')
    span([a['same_mean'] for a in nearest], d['nearest_same_mean'], 'same mean')
    span([a[k] for a in longr for k in ('same_below_0.5', 'diff_at_or_above_0.5')], d['long_range_errors'], 'long')
    near(b['interior_unconfident_fraction'], d['interior_unconfident_fraction'], 'unconfident')
    self.assertEqual(b['per_object_median_pieces'], d['median_pieces'])
    for got, ev in zip(b['by_thickness_tercile'], d['thickness_terciles'], strict=True):
      self.assertEqual(got['objects'], ev['objects'])
      near(got['f1'], ev['f1'])
    span([s['split_rate'] for s in b['split_edges']], d['split_rate'], 'split rate')
    for name in EVIDENCE['hybrids']:
      self.assertEqual([r['threshold'] for r in b[name]['table']], d['hybrid_thresholds'], name)
    scenes = {s['key']: s['stats'] for s in _json('vitseg-breakdown-s66000-20261026/vit_scenes.json')['scenes']}
    for key, ev in d['examples'].items():
      self.assertEqual(scenes[key]['f1_rank_of_457'], ev['rank'], key)
      self.assertEqual(scenes[key]['pieces_over_1pct'], ev['pieces'], key)
      near(scenes[key]['vit_f1'], ev['f1'], key)
    pe = d['paired_object']
    obj = [r for r in b['objects'] if r['object_id'] == pe['object_id']]
    self.assertEqual(len(obj), 1)
    for k in ('recall', 'precision', 'f1'):
      near(obj[0][k], pe['vit_' + k], k)
    self.assertEqual(scenes['vit12_ffn346093']['f1_rank_of_457'], pe['vit_rank'])
    ffn_dir = RUNS / EVIDENCE['ffn_reference']['run']
    ffn = [r for r in json.loads((ffn_dir / 'crop-12' / 'benchmark.json').read_text())['records']]
    self.assertEqual([r['object_id'] for r in ffn], [pe['object_id']])  # FFN's only crop-12 seed
    near(ffn[0]['recall'], pe['ffn_recall']); near(ffn[0]['precision'], pe['ffn_precision'])
    near(_f1(ffn[0]['precision'], ffn[0]['recall']), pe['ffn_f1'])
    unl = sorted(r['unlabeled_fraction'] for f in glob.glob(str(ffn_dir / 'crop-*' / 'benchmark.json'))
                 for r in json.loads(Path(f).read_text())['records'])
    q = statistics.quantiles(unl, n=4)
    near(q[0], d['ffn_unlabeled_fraction_quartiles'][0]); near(q[2], d['ffn_unlabeled_fraction_quartiles'][1])

  def test_test_crops_were_not_read(self):
    """No recorded job read a test crop: the tuning job ran run_vitseg.py --tune-only, which
    returns before the first test-crop access; its log and output hold crop 12 only; and the
    breakdown and mesh scripts load only the tuning crop."""
    test_crops = {c['index'] for c in fresh.load_manifest()['crops']}
    self.assertNotIn(12, test_crops)
    main = (ROOT / 'examples' / 'run_vitseg.py').read_text().split('def main():', 1)[1]
    self.assertLess(main.index('if a.tune_only:'), main.index('fresh.'))  # returns before any test crop
    self.assertIn('--tune-only', (ROOT / 'cluster' / 'wahab' / 'vitseg.sbatch').read_text().split('tune)', 1)[1].split(';;')[0])
    log = (RUNS / 'vitseg-tune-r3-s66000-20261026' / '7451650.out').read_text().splitlines()
    self.assertTrue(any(l.startswith('vitseg tune on ') for l in log))
    crops = [json.loads(l)['crop'] for l in log if l.startswith('{"crop"')]
    self.assertEqual(crops, [12])
    tuning = _json('vitseg-tune-r3-s66000-20261026/vitseg_tuning.json')
    self.assertEqual(tuning['crop'], 12)
    self.assertFalse({'records', 'crops', 'verdict'} & set(tuning))
    for script in ('breakdown.py', 'meshes.py'):
      text = (RUNS / 'vitseg-breakdown-s66000-20261026' / script).read_text()
      self.assertNotIn('fresh', text, script)
      loads = [l for l in text.splitlines() if 'load_raw_xyz(' in l or 'load_proofread_zyx(' in l]
      self.assertTrue(loads and all('TUNE_CROP' in l for l in loads), script)
    for d in ('vitseg-train-r3-20261025', 'vitseg-tune-r3-s66000-20261026', 'vitseg-breakdown-s66000-20261026'):
      self.assertFalse((RUNS / d / 'vitseg_run.json').exists(), d)

if __name__ == '__main__':
  unittest.main()
