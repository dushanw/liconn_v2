"""Fine-tuning SegNeuron on newer GPUs (docs/gpu_training_results.md) matches its evidence and its logs.

The evidence writer (examples/gpu_training_evidence.py) is re-run here on the training logs in the data
cache, and docs/gpu_training_evidence.json must equal its output. The logs and the job scripts kept in
the run folders are checked by sha256, and the repository's copies of those scripts must be the ones
that ran. docs/gpu_training_results.md must be exactly the output of examples/write_gpu_training_results.py,
that writer must take every number from the evidence, and the manuscript's statements about this study must
be the evidence's.
"""

import copy
import json
import math
import re
from pathlib import Path
import sys
import unittest

from liconn_v2 import data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'examples'))
import gpu_training_evidence  # pylint: disable=g-import-not-at-top,wrong-import-position
import write_gpu_training_results  # pylint: disable=g-import-not-at-top,wrong-import-position

EVIDENCE = json.loads((ROOT / 'docs' / 'gpu_training_evidence.json').read_text())
RESULTS = (ROOT / 'docs' / 'gpu_training_results.md').read_text()
RUNS = data.cache_dir() / 'runs'
# The repository's copies of the job scripts and the modified training script that ran on MIT.
NUMBER = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?')


def altered(x, seed: int, path=''):
  """Every number of x changed by a factor that depends on where it is, so sums and ratios change as well."""
  if isinstance(x, bool) or x is None or isinstance(x, str):
    return x
  f = 1.13 + ((sum(map(ord, path)) + 31 * seed) % 89) / 53
  if isinstance(x, float):
    return x * f + 0.0317
  if isinstance(x, int):
    return int(x * f) + 3
  if isinstance(x, dict):
    return {k: altered(v, seed, f'{path}/{k}') for k, v in x.items()}
  return [altered(v, seed, f'{path}[{i}]') for i, v in enumerate(x)]


COPIES = {'segneuron-b200-batch-20261030/train_segneuron_bs.py': 'cluster/mit/segneuron_gpu/train_segneuron_bs.py',
          'segneuron-b200-batch-20261030/run.sbatch': 'cluster/mit/segneuron_gpu/run.sbatch',
          'segneuron-mit-probe-20261030/probe.sbatch': 'cluster/mit/segneuron_gpu/probe.sbatch'}


class EvidenceTest(unittest.TestCase):

  def test_evidence_is_the_writer_s_output(self):
    self.assertEqual(json.loads(json.dumps(gpu_training_evidence.build())), EVIDENCE)

  def test_logs_and_scripts_are_pinned(self):
    for run, sha in EVIDENCE['logs_sha256'].items():
      self.assertEqual(data.sha256_file(RUNS / run / 'train_log.json'), sha, run)
    self.assertEqual(set(EVIDENCE['scripts_sha256']), set(COPIES))
    for run_file, sha in EVIDENCE['scripts_sha256'].items():
      self.assertEqual(data.sha256_file(RUNS / run_file), sha, run_file)
      self.assertEqual(data.sha256_file(ROOT / COPIES[run_file]), sha, COPIES[run_file])

  def test_each_run_is_what_it_is_listed_as(self):
    gpus = {'V100': 'V100', 'A100': 'A100', 'H100': 'H100', 'B200': 'B200'}
    for name, p in {**EVIDENCE['probes'], **EVIDENCE['workers_16']}.items():
      self.assertIn(gpus[name.split()[0]], p['gpu'], name)
      self.assertEqual(p['batch'], 4, name)
    self.assertEqual({p['workers'] for p in EVIDENCE['workers_16'].values()}, {16})
    for b, r in EVIDENCE['batch'].items():
      self.assertEqual(r['batch'], int(b))
      self.assertIn('B200', r['gpu'])
      self.assertEqual(r['patches'], EVIDENCE['batch']['4']['patches'])  # equal data

  def test_every_run_trained_on_the_study_s_data(self):
    m = EVIDENCE['data_manifest_sha256']
    self.assertEqual(len(set(m.values())), 1)
    links = EVIDENCE['data_links']
    self.assertTrue(links['segneuron-b200-batch-20261030'].endswith('/runs/segneuron-mit-probe-20261030/data'))
    self.assertTrue(links['segneuron-a100-probe-20261030'].endswith('/liconn_runs/segneuron-ft-20261028/data'))
    self.assertEqual(EVIDENCE['b200_batch4_time_from'], EVIDENCE['runs']['batch']['4'])

  def test_learning_rate_rule_as_stated(self):
    """The page states the batch study's rule; the logged settings must follow it, batch 48 included."""
    for b, r in [*EVIDENCE['batch'].items(), ('48', EVIDENCE['batch48'])]:
      self.assertTrue(math.isclose(r['base_lr'], 1e-3 * (int(b) / 4) ** 0.5, rel_tol=1e-3), b)
      self.assertAlmostEqual(r['warmup_steps'] / r['steps_done'], 0.0 if b == '4' else 0.05, places=3)


class ResultsPageTest(unittest.TestCase):

  def test_page_is_exactly_the_writer_s_output(self):
    self.assertEqual(RESULTS, write_gpu_training_results.render(EVIDENCE))

  def test_writer_takes_every_number_from_the_evidence(self):
    """Rendered twice from evidence in which every number is changed (each by its own factor, so that ratios
    change too, and differently each time), the pages share no number with the real page except fixed labels:
    batch sizes, job numbers and run dates, the crop-12 validation crop, the torch version, the 0.5 affinity
    threshold and the square-root exponent, the data manifest's sha256, the number of probes (5) and the V100
    against itself (1.00)."""
    pages = [write_gpu_training_results.render(altered(EVIDENCE, seed)) for seed in (1, 2)]
    page = pages[0]
    real = set(NUMBER.findall(RESULTS))
    labels = {'4', '8', '16', '32', '12', '2.7', '0.5', '1', '1.00', '5'} | {
        n.rstrip(',') for n in real if len(n.rstrip(',')) >= 7}
    # A number found on both altered pages is fixed in the writer; one found on only one is a chance match.
    shared = {n.rstrip(',') for n in real & set(NUMBER.findall(pages[0])) & set(NUMBER.findall(pages[1]))} - labels
    self.assertEqual(shared, set())
    self.assertNotEqual(page, RESULTS)
    # '5' is also the number of probes, which no alteration changes; the warm-up share must still follow the evidence.
    warm_up = r'(?<![\d.])5% linear warm-up'
    self.assertRegex(RESULTS, warm_up)
    for p in pages:
      self.assertNotRegex(p, warm_up)

  def test_manuscript_numbers(self):
    """The manuscript's statements about this study (manuscript/body.md and supplement.md), formatted from the
    evidence and looked up in the text. How far the validation losses after 200 steps agree is checked by the
    manuscript's own test, tests/test_manuscript_unseen.py, together with its rewording."""
    text = (ROOT / 'manuscript' / 'body.md').read_text() + (ROOT / 'manuscript' / 'supplement.md').read_text()
    per = EVIDENCE['seconds_per_step_batch4']
    v100, a100, h100, b200 = EVIDENCE['probe_order'][0], EVIDENCE['probe_order'][1], EVIDENCE['probe_order'][3], \
        EVIDENCE['probe_order'][4]
    ratio = f"{EVIDENCE['b200_over_v100']:.1f}×"
    batch, b48, full = EVIDENCE['batch'], EVIDENCE['batch48'], EVIDENCE['v100_full']
    statements = [
        f'{ratio} faster per step on a B200',
        f'than on the V100 ({per[b200]:.2f} against {per[v100]:.2f} s at batch 4',
        f"projected at about {EVIDENCE['b200_batch4_projected_hours']:.1f} h on a B200 against "
        f"{EVIDENCE['v100_full_hours']:.1f} h on the V100",
        f'a step took {per[v100]:.2f} s on a V100, {per[a100]:.2f} s on an A100 (PCIe) and {per[h100]:.2f} s on an '
        'H100',
        f"{per[b200]:.2f} s on a B200 (steady state over {batch['4']['steps_done']:,} steps), {ratio} faster than "
        'the V100',
        f"from {batch['4']['ms_per_patch']:.1f} ms (batch 4) to {batch['32']['ms_per_patch']:.1f} ms (batch 32; "
        f"{batch['4']['ms_per_patch'] / batch['32']['ms_per_patch']:.1f}×",
        '(' + ', '.join(f"{batch[b]['final_valid_loss']:.3f}" for b in ('4', '8', '16')) +
        f" and {batch['32']['final_valid_loss']:.3f} for batch 4, 8, 16 and 32",
        f"finished in {b48['seconds'] / 3600:.1f} h but ended at a validation loss of {b48['final_valid_loss']:.3f} "
        f"against {full['final_valid_loss']:.3f}",
        f"({b48['final_diff_at_or_above_0.5']:.3f} against {full['final_diff_at_or_above_0.5']:.3f})",
        f"was still declining slowly at {full['final_valid_loss']:.3f} when training ended after "
        f"{full['steps_done']:,} steps ({EVIDENCE['v100_full_hours']:.1f} h",
    ]
    for x in statements:
      self.assertIn(x, text)

if __name__ == '__main__':
  unittest.main()
