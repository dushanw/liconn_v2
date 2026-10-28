"""The recorded SegNeuron zero-shot run (docs/segneuron_results.md) matches its evidence."""

import hashlib
import json
from pathlib import Path
import unittest

from liconn_v2 import data
from liconn_v2 import fresh
from liconn_v2 import segneuron

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'segneuron_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
RUN = RUNS / EVIDENCE['run']
TOL = EVIDENCE['tolerance']
SUMMARY_KEYS = ('recall', 'precision', 'f1', 'lost_to_background', 'lost_to_other_segments', 'pieces_over_1pct')


def _sha(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 24), b''):
      h.update(chunk)
  return h.hexdigest()


def _result():
  return json.loads((RUN / 'segneuron_crop12.json').read_text())


class SegneuronEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    staged = dict(reversed(l.split()) for l in (RUN / 'source_sha256.txt').read_text().splitlines())
    self.assertEqual(set(staged), set(EVIDENCE['run_source_sha256']))
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(staged[rel], expected, rel)
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_pinned_model_and_jobs(self):
    sbatch = (ROOT / 'cluster' / 'wahab' / 'segneuron.sbatch').read_text()
    self.assertIn(f"COMMIT={EVIDENCE['segneuron']['commit']}", sbatch)
    self.assertIn(f"CKPT_SHA={EVIDENCE['segneuron']['checkpoint_sha256']}", sbatch)
    for mode, job in EVIDENCE['slurm_jobs'].items():
      log = (RUN / f'{job}.out').read_text().splitlines()
      self.assertTrue(log[0].startswith(f'segneuron {mode} on '), mode)
      self.assertIn('SegNeuronModel.ckpt: OK', log[1])  # sha256sum -c ran before any work
    self.assertEqual(_result()['platform']['slurm_job'], str(EVIDENCE['slurm_jobs']['post']))
    for res, ev in EVIDENCE['resolutions'].items():
      inf = json.loads((RUN / res / 'infer' / 'inference.json').read_text())
      self.assertEqual(inf['checkpoint']['sha256'], EVIDENCE['segneuron']['checkpoint_sha256'])
      self.assertEqual(inf['input']['shape_zyx'], ev['inference']['input_shape_zyx'])
      self.assertEqual(inf['input']['sha256'], ev['inference']['input_sha256'])
      self.assertEqual(inf['patch_count'], ev['inference']['patch_count'])
      self.assertAlmostEqual(inf['inference_seconds'], ev['inference']['inference_seconds'], delta=0.05)
      self.assertEqual(_result()['resolutions'][res]['inference'], inf)

  def test_output_scores(self):
    for res, ev in EVIDENCE['resolutions'].items():
      o = _result()['resolutions'][res]['outputs']
      for k, v in ev['foreground'].items():
        got = o['foreground']['quantiles']['label_zero'][1] if k == 'label_zero_median' else o['foreground'][k]
        self.assertAlmostEqual(got, v, delta=TOL, msg=f'{res} {k}')
      chans = {a['channel']: a for a in o['affinities']}
      self.assertEqual(set(chans), set(ev['affinities']))
      for c, evc in ev['affinities'].items():
        for k, v in evc.items():
          got = chans[c]['quantiles'][k.split('_')[0]][1] if k.endswith('_median') else chans[c][k]
          self.assertAlmostEqual(got, v, delta=TOL, msg=f'{res} {c} {k}')

  def test_ours(self):
    for res, ev in EVIDENCE['resolutions'].items():
      ours, e = _result()['resolutions'][res]['ours'], ev['ours']
      best = max(ours['table'], key=lambda r: r['mean_f1'])
      self.assertEqual(best['threshold'], ours['threshold'])  # chosen by mean F1 on crop 12
      self.assertEqual(ours['threshold'], e['threshold'])
      self.assertEqual(ours['fragments'], e['fragments'])
      for k in SUMMARY_KEYS:
        self.assertAlmostEqual(ours['per_object_mean'][k], e[k], delta=TOL, msg=f'{res} {k}')
      self.assertAlmostEqual(best['mean_f1'], e['f1'], delta=TOL)
      self.assertEqual(ours['per_object_mean']['seeds_in_background'], e['seeds_in_background'])
      self.assertEqual(ours['median_pieces'], e['median_pieces'])
      self.assertAlmostEqual(ours['background']['precision'], e['background_precision'], delta=TOL)
      self.assertAlmostEqual(ours['background']['recall'], e['background_recall'], delta=TOL)
      at_half = [r for r in ours['table'] if r['threshold'] == 0.5]
      self.assertAlmostEqual(at_half[0]['mean_f1'], e['f1_at_0.5'], delta=TOL)
      self.assertEqual(len(ours['objects']), EVIDENCE['objects'])
      for r in ours['objects']:  # recall, background and other segments account for each object
        self.assertAlmostEqual(r['recall'] + r['lost_to_background'] + r['lost_to_other_segments'], 1.0, delta=1e-9)

  def test_frmc(self):
    log = (RUN / f"{EVIDENCE['slurm_jobs']['post']}.out").read_text()
    for res, ev in EVIDENCE['resolutions'].items():
      frmc, e = _result()['resolutions'][res]['frmc'], ev['frmc']
      self.assertEqual([t['beta'] for t in frmc['table']], list(segneuron.BETAS))
      best = max(frmc['table'], key=lambda t: (t['masked']['mean_f1'], t['beta']))
      self.assertEqual(best['beta'], frmc['beta'])  # chosen by masked mean F1 on crop 12
      self.assertEqual(frmc['beta'], e['beta'])
      for t in frmc['table']:
        et = e['table'][str(t['beta'])]
        self.assertEqual(t['segments'], et['segments'])
        self.assertAlmostEqual(t['masked']['mean_f1'], et['masked_f1'], delta=TOL)
        self.assertAlmostEqual(t['as_released']['mean_f1'], et['as_released_f1'], delta=TOL)
        self.assertEqual(t['as_released']['background']['predicted_fraction'], 0.0)
        # FRMC_post.py's own count (instances includes no label 0: FRMC labels every voxel)
        block = log.split(f'/{res}/frmc/beta_{t["beta"]}.npy"', 1)[1].split('}', 1)[0]
        self.assertIn(f'"instances": {t["segments"]},', block)
      for name, key in (('masked', 'masked_at_beta'), ('as_released', 'as_released_at_beta')):
        for k in SUMMARY_KEYS:
          self.assertAlmostEqual(frmc[key]['per_object_mean'][k], e[name][k], delta=TOL, msg=f'{res} {name} {k}')

  def test_references(self):
    vit = json.loads((ROOT / 'docs' / 'vitseg_evidence.json').read_text())
    ref = EVIDENCE['references']
    self.assertAlmostEqual(vit['tuning']['mean_f1'], ref['vit_crop12']['f1'], delta=TOL)
    self.assertAlmostEqual(vit['breakdown']['lost_to_other_segments'], ref['vit_crop12']['lost_to_other_segments'], delta=TOL)
    self.assertAlmostEqual(vit['breakdown']['lost_to_background'], ref['vit_crop12']['lost_to_background'], delta=TOL)
    oracle = json.loads((RUNS / ref['oracle_crop12']['run']).read_text())
    self.assertEqual(oracle['objects'], EVIDENCE['objects'])
    self.assertAlmostEqual(oracle['clean']['mean_f1'], ref['oracle_crop12']['f1'], delta=TOL)

  def test_test_crops_were_not_read(self):
    """The driver loads only the tuning crop, and the score file and logs hold crop 12 only."""
    test_crops = {c['index'] for c in fresh.load_manifest()['crops']}
    self.assertNotIn(12, test_crops)
    for rel in ('examples/run_segneuron.py', 'liconn_v2/segneuron.py'):
      text = (ROOT / rel).read_text()
      self.assertNotIn('fresh', text, rel)
      loads = [l for l in text.splitlines() if 'load_raw_xyz(' in l or 'load_proofread_zyx(' in l]
      self.assertTrue(all('TUNE_CROP' in l for l in loads), rel)
    self.assertEqual(len([l for l in (ROOT / 'examples/run_segneuron.py').read_text().splitlines()
                          if 'load_raw_xyz(' in l or 'load_proofread_zyx(' in l]), 2)
    r = _result()
    self.assertEqual(r['crop'], 12)
    self.assertEqual(r['objects'], EVIDENCE['objects'])
    self.assertEqual(set(r['resolutions']), set(segneuron.RESOLUTIONS))


if __name__ == '__main__':
  unittest.main()
