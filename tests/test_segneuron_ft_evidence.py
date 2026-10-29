"""The recorded SegNeuron fine-tuning run (docs/segneuron_finetune_results.md) matches its evidence."""

import hashlib
import json
from pathlib import Path
import unittest

from liconn_v2 import data
from liconn_v2 import fresh
from liconn_v2 import segneuron
from liconn_v2 import segneuron_ft as ft

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'segneuron_finetune_evidence.json').read_text())
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
  return json.loads((RUN / 'segneuron_ft_crop12.json').read_text())


def _log():
  return json.loads((RUN / 'train' / 'train_log.json').read_text())


class SegneuronFtEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    staged = dict(reversed(l.split()) for l in (RUN / 'source_sha256.txt').read_text().splitlines())
    self.assertEqual(set(staged), set(EVIDENCE['run_source_sha256']))
    for rel in ('liconn_v2/segneuron_ft.py', 'examples/train_segneuron.py', 'examples/run_segneuron_ft.py',
                'cluster/wahab/segneuron_ft.sbatch'):
      self.assertIn(rel, staged)
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      self.assertEqual(staged[rel], expected, rel)
      self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_pinned_model_and_jobs(self):
    sn = EVIDENCE['segneuron']
    sbatch = (ROOT / 'cluster' / 'wahab' / 'segneuron_ft.sbatch').read_text()
    self.assertIn(f"COMMIT={sn['commit']}", sbatch)
    self.assertIn(f"CKPT_SHA={sn['released_checkpoint_sha256']}", sbatch)
    for mode, job in EVIDENCE['slurm_jobs'].items():
      log = (RUN / f'{job}.out').read_text().splitlines()
      self.assertTrue(log[0].startswith(f'segneuron_ft {mode} on '), mode)
      self.assertIn('SegNeuronModel.ckpt: OK', log[1])  # sha256sum -c ran before any work
    r = _result()
    self.assertEqual(r['platform']['slurm_job'], str(EVIDENCE['slurm_jobs']['post']))
    inf, e = r['inference'], EVIDENCE['inference']
    self.assertEqual(inf, json.loads((RUN / 'native' / 'infer' / 'inference.json').read_text()))
    self.assertEqual(inf['checkpoint']['sha256'], sn['finetuned_checkpoint_sha256'])
    self.assertNotEqual(sn['finetuned_checkpoint_sha256'], sn['released_checkpoint_sha256'])
    self.assertTrue(inf['checkpoint']['path'].endswith('/train/best.ckpt'))
    self.assertEqual(inf['input']['shape_zyx'], e['input_shape_zyx'])
    self.assertEqual(inf['input']['sha256'], e['input_sha256'])
    self.assertEqual(inf['patch_count'], e['patch_count'])
    self.assertAlmostEqual(inf['inference_seconds'], e['inference_seconds'], delta=0.05)
    # the fine-tuned model saw the same crop-12 input as the zero-shot native run
    zs = json.loads((ROOT / 'docs' / 'segneuron_evidence.json').read_text())
    self.assertEqual(inf['input']['sha256'], zs['resolutions']['native']['inference']['input_sha256'])

  def test_training(self):
    log, e = _log(), EVIDENCE['training']
    self.assertEqual(_result()['training'], {k: log[k] for k in ('best', 'steps_done', 'seconds', 'stopped', 'args')})
    self.assertEqual(log['args']['iters'], e['iters'])
    self.assertEqual(log['args']['batch'], e['batch'])
    self.assertEqual(log['steps_done'], e['steps_done'])
    self.assertEqual(log['steps_done'], log['args']['iters'])  # ran to the end, not the time budget
    self.assertIsNone(log['stopped'])
    self.assertIsNone(e['stopped'])
    self.assertAlmostEqual(log['seconds'], e['seconds'], delta=0.05)
    self.assertEqual(log['gpu'], e['gpu'])
    self.assertEqual(log['torch'], e['torch'])
    self.assertEqual((log['base_lr'], log['end_lr'], log['power']), (ft.BASE_LR, ft.END_LR, ft.POWER))
    self.assertEqual((e['base_lr'], e['end_lr'], e['power']), (ft.BASE_LR, ft.END_LR, ft.POWER))
    self.assertEqual(log['train_crops'], list(ft.TRAIN_CROPS))
    self.assertEqual(log['valid_crop'], ft.VALID_CROP)
    self.assertEqual(log['valid_patches'], ft.VALID_PATCHES)
    self.assertAlmostEqual(max(t['max_memory_gb'] for t in log['train']), e['max_memory_gb'], delta=TOL)
    v = log['validation']
    self.assertEqual(len(v), e['validations'])
    self.assertEqual([x['step'] for x in v], list(range(0, log['steps_done'] + 1, ft.VALID_EVERY)))
    self.assertEqual(ft.best_step(v), log['best'])  # the checkpoint chosen by crop-12 validation loss
    self.assertEqual(log['best']['step'], e['best_step'])
    self.assertEqual(log['best']['step'], v[-1]['step'])  # the lowest loss was the last validation
    by_step = {x['step']: x for x in v}
    for step, ev in e['validation'].items():
      for k, val in ev.items():
        self.assertAlmostEqual(by_step[int(step)][k], val, delta=TOL, msg=f'{step} {k}')
    self.assertEqual(len({x['diff_pairs'] for x in v}), 1)  # the same fixed patches each time

  def test_output_scores(self):
    o = _result()['outputs']
    for k, v in EVIDENCE['foreground'].items():
      got = o['foreground']['quantiles']['label_zero'][1] if k == 'label_zero_median' else o['foreground'][k]
      self.assertAlmostEqual(got, v, delta=TOL, msg=k)
    chans = {a['channel']: a for a in o['affinities']}
    self.assertEqual(set(chans), set(EVIDENCE['affinities']))
    for c, evc in EVIDENCE['affinities'].items():
      for k, v in evc.items():
        got = chans[c]['quantiles'][k.split('_')[0]][1] if k.endswith('_median') else chans[c][k]
        self.assertAlmostEqual(got, v, delta=TOL, msg=f'{c} {k}')

  def test_ours(self):
    ours, e = _result()['ours'], EVIDENCE['ours']
    best = max(ours['table'], key=lambda r: r['mean_f1'])
    self.assertEqual(best['threshold'], ours['threshold'])  # chosen by mean F1 on crop 12
    self.assertEqual(ours['threshold'], e['threshold'])
    self.assertEqual(ours['fragments'], e['fragments'])
    for k in SUMMARY_KEYS:
      self.assertAlmostEqual(ours['per_object_mean'][k], e[k], delta=TOL, msg=k)
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
    frmc, e = _result()['frmc'], EVIDENCE['frmc']
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
      block = log.split(f'/native/frmc/beta_{t["beta"]}.npy"', 1)[1].split('}', 1)[0]
      self.assertIn(f'"instances": {t["segments"]},', block)
    for name, key in (('masked', 'masked_at_beta'), ('as_released', 'as_released_at_beta')):
      for k in SUMMARY_KEYS:
        self.assertAlmostEqual(frmc[key]['per_object_mean'][k], e[name][k], delta=TOL, msg=f'{name} {k}')

  def test_same_objects_as_the_references(self):
    """The comparison rows in the results come from the checked zero-shot and ViT evidence."""
    zs = json.loads((ROOT / EVIDENCE['references']['zeroshot_native']['source']).read_text())
    vit = json.loads((ROOT / EVIDENCE['references']['vit_crop12']['source']).read_text())
    self.assertEqual(zs['objects'], EVIDENCE['objects'])
    self.assertEqual(vit['tuning']['objects'], EVIDENCE['objects'])
    self.assertEqual(zs['segneuron']['commit'], EVIDENCE['segneuron']['commit'])
    self.assertEqual(zs['segneuron']['checkpoint_sha256'], EVIDENCE['segneuron']['released_checkpoint_sha256'])
    ids = [r['object_id'] for r in _result()['ours']['objects']]
    self.assertEqual(len(set(ids)), EVIDENCE['objects'])
    zs_run = json.loads((RUNS / zs['run'] / 'segneuron_crop12.json').read_text())
    self.assertEqual(ids, [r['object_id'] for r in zs_run['resolutions']['native']['ours']['objects']])
    # paired per-object F1, fine-tuned against zero-shot native, each at its own crop-12 threshold
    pairs = list(zip(_result()['ours']['objects'], zs_run['resolutions']['native']['ours']['objects']))
    change = sorted(a['f1'] - b['f1'] for a, b in pairs)
    e = EVIDENCE['paired_vs_zeroshot_native']
    self.assertEqual(sum(c > 0 for c in change), e['objects_f1_higher'])
    self.assertEqual(sum(c < 0 for c in change), e['objects_f1_lower'])
    self.assertAlmostEqual(change[len(change) // 2], e['median_f1_change'], delta=TOL)  # 457: odd
    self.assertAlmostEqual(change[0], e['largest_drop'], delta=TOL)
    self.assertEqual(sum(a['f1'] >= 0.9 for a, _ in pairs), e['objects_f1_at_least_0.9'])
    self.assertEqual(sum(b['f1'] >= 0.9 for _, b in pairs), e['zeroshot_objects_f1_at_least_0.9'])
    self.assertAlmostEqual(zs['resolutions']['native']['ours']['f1'], 0.4999, delta=TOL)
    self.assertAlmostEqual(vit['tuning']['mean_f1'], 0.5314, delta=TOL)

  def test_test_crops_were_not_read(self):
    """The drivers load only crops 0-12, and the score file holds crop 12 only."""
    test_crops = {c['index'] for c in fresh.load_manifest()['crops']}
    self.assertTrue(test_crops.isdisjoint(set(ft.TRAIN_CROPS) | {ft.VALID_CROP}))
    for rel in ('examples/run_segneuron_ft.py', 'examples/train_segneuron.py', 'liconn_v2/segneuron_ft.py'):
      self.assertNotIn('fresh', (ROOT / rel).read_text(), rel)
    manifest = json.loads((RUN / 'data' / 'manifest.json').read_text())
    self.assertEqual(sorted(int(k) for k in manifest['crops']), sorted(set(ft.TRAIN_CROPS) | {ft.VALID_CROP}))
    r = _result()
    self.assertEqual(r['crop'], ft.VALID_CROP)
    self.assertEqual(r['objects'], EVIDENCE['objects'])
    self.assertEqual(r['resolution'], 'native')


if __name__ == '__main__':
  unittest.main()
