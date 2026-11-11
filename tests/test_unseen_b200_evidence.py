"""The SegNeuron-on-B200 addendum (docs/unseen_b200_results.md) matches its evidence and its result files.

The judge (examples/judge_unseen_b200.py) is re-run here on the copied result files in the data cache,
and docs/unseen_b200_evidence.json is compared with its output, not with the extract itself. The
result files are checked by sha256. docs/unseen_b200_results.md must be exactly the output of
examples/write_unseen_b200_results.py, and its key numbers are also checked one by one against the evidence.
"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from liconn_v2 import data
from liconn_v2 import unseen
from liconn_v2 import unseen_b200

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'examples'))
import judge_unseen_b200  # pylint: disable=g-import-not-at-top,wrong-import-position
import write_unseen_b200_results  # pylint: disable=g-import-not-at-top,wrong-import-position

EVIDENCE = json.loads((ROOT / 'docs' / 'unseen_b200_evidence.json').read_text())
RESULTS = (ROOT / 'docs' / 'unseen_b200_results.md').read_text()
B200 = data.cache_dir() / 'runs' / 'unseen-b200-segneuron-20261106'
V100 = data.cache_dir() / 'runs' / 'unseen-20261104' / 'v100'
CROPS = [str(k) for k in unseen.CROPS]
RUNS = ('F5', 'F1', 'S10', 'P2', 'Q2')
# F5's crop seconds as the test recorded them, read independently of the addendum's evidence.
EVIDENCE_F5 = json.loads((ROOT / 'docs' / 'unseen_evidence.json').read_text())['judged']['speed_b200']['F5'][
    'crop_seconds']


def section(start: str, end: str) -> str:
  """The page's text from the heading start (which occurs once) to the next heading end."""
  assert RESULTS.count(start) == 1, start
  i = RESULTS.index(start)
  return RESULTS[i:RESULTS.index(end, i)]


def host(k: str) -> str:
  return EVIDENCE['devices'][k]['job'].split()[1]


class EvidenceTest(unittest.TestCase):

  def test_evidence_is_the_judge_s_output(self):
    rebuilt = json.loads(json.dumps(judge_unseen_b200.build(B200, V100)))
    self.assertEqual(rebuilt, EVIDENCE)

  def test_result_files_are_pinned(self):
    files = EVIDENCE['result_files_sha256']
    self.assertEqual(sorted(files), sorted(f'crop-{k}/{f}' for k in CROPS for f in ('segneuron.json', 'env-infer.txt')))
    for f, sha in files.items():
      self.assertEqual(data.sha256_file(B200 / f), sha, f)

  def test_design_and_test_evidence_are_current(self):
    self.assertEqual(EVIDENCE['design_sha256'], unseen_b200.design_sha256())
    self.assertEqual(EVIDENCE['test_evidence_sha256'], data.sha256_file(ROOT / 'docs' / 'unseen_evidence.json'))

  def test_every_crop_judged_on_a_b200(self):
    j = EVIDENCE['judged']
    self.assertEqual(j['failures'], {})
    self.assertEqual(sorted(j['segneuron_b200']), CROPS)
    self.assertEqual(set(d['gpu'] for d in EVIDENCE['devices'].values()), {'NVIDIA B200, 590.48.01'})
    for k in CROPS:
      self.assertIn(f'job 28228697_{k} ', EVIDENCE['devices'][k]['job'])
      for way in unseen_b200.WAYS:
        for r in RUNS:
          c = EVIDENCE['judged']['comparisons'][way][r]
          self.assertEqual(c['crops_judged'], 10)
          self.assertEqual(c['crops_segneuron_cheaper'] + c['crops_segneuron_dearer'], 10)


class ResultsPageTest(unittest.TestCase):
  """Every number of docs/unseen_b200_results.md, formatted from the evidence."""

  j = EVIDENCE['judged']

  def assertStated(self, text):
    self.assertIn(text, RESULTS)

  def test_page_is_exactly_the_writer_s_output(self):
    """The whole page, character for character, is rendered from the evidence and the run records, so it
    holds no number, row or claim beyond those checked below."""
    self.assertEqual(RESULTS, write_unseen_b200_results.render(EVIDENCE, B200, V100))

  def test_writer_takes_every_number_from_its_inputs(self):
    """Rendered from altered evidence and records holding only the smoke run (crop 11) and crops 41-50, the
    page carries the altered numbers, not the real ones: none of them is written into the writer."""
    e = copy.deepcopy(EVIDENCE)
    for k in CROPS:
      e['devices'][k]['job'] = f'host node-x job 77777777_{k} cpus 4'
    for way in unseen_b200.WAYS:
      e['judged']['comparisons'][way]['F5']['crops_segneuron_cheaper'] = 3
    with tempfile.TemporaryDirectory() as tmp:
      b200, v100 = Path(tmp) / 'b200', Path(tmp) / 'v100'
      smoke = b200 / 'smoke20' / 'crop-11'
      smoke.mkdir(parents=True)
      record = json.loads((B200 / 'smoke20' / 'crop-11' / 'segneuron.json').read_text())
      record['smoke_objects'] = 9
      (smoke / 'segneuron.json').write_text(json.dumps(record))
      (smoke / 'gpu-segneuron.txt').write_text('name, driver_version\nGPU\nhost node-y job 55555555_11 cpus 4\n')
      for k in CROPS:
        (b200 / f'crop-{k}' / 'infer').mkdir(parents=True)
        (b200 / f'crop-{k}' / 'env-infer.txt').write_text((B200 / f'crop-{k}' / 'env-infer.txt').read_text())
        (b200 / f'crop-{k}' / 'infer' / 'affinities.npy').write_bytes(b'x' * 1234)
        (b200 / f'crop-{k}' / 'segneuron.json').write_text(json.dumps({'inference': {'crop_size_zyx': [7, 64, 96]}}))
        (v100 / f'crop-{k}').mkdir(parents=True)
        (v100 / f'crop-{k}' / 'env-infer.txt').write_text((V100 / f'crop-{k}' / 'env-infer.txt').read_text())
      page = write_unseen_b200_results.render(e, b200, v100)
    for text in ('MIT job 77777777 in', '(job 77777777, one B200 per crop, 4 CPUs)',
                 '(job 55555555_11, 9 objects scored)', 'Whether the count of 3 depends', '(1,234 bytes per crop)',
                 'node-x: crops 41, 42', 'runs one 7 x 64 x 96 patch at a time'):
      self.assertIn(text, page)
    for text in ('28228697', '28225144', '8 CPUs', '20 objects scored', 'count of 7', '1,500,000,128',
                 '20 x 128 x 128'):
      self.assertNotIn(text, page)

  def test_runs(self):
    """The run facts on the page, each read from the run's own records."""
    self.assertStated(f"sha256 `{EVIDENCE['design_sha256'][:16]}...`")
    self.assertStated('failures: 0; non-finite SegNeuron outputs: none on any crop.')
    for n in sorted({host(k) for k in CROPS}):
      self.assertStated(f"- {n}: crops {', '.join(k for k in CROPS if host(k) == n)}.")
    smoke = json.loads((B200 / 'smoke20' / 'crop-11' / 'segneuron.json').read_text())
    job = (B200 / 'smoke20' / 'crop-11' / 'gpu-segneuron.txt').read_text().splitlines()[2].split()[3]
    self.assertStated(f"Smoke run on crop {smoke['crop']} (job {job}, {smoke['smoke_objects']} objects scored): "
                      'completed. It checked only that the job runs and is not a result.')
    cpus = {EVIDENCE['devices'][k]['job'].split()[-1] for k in CROPS}
    self.assertEqual(cpus, {'8'})
    self.assertStated('(job 28228697, one B200 per crop, 8 CPUs)')
    gpus = {EVIDENCE['devices'][k]['gpu'] for k in CROPS}
    self.assertEqual(len(gpus), 1)
    self.assertStated(f'{gpus.pop()} (GPU and driver)')
    for k in CROPS:
      env = (B200 / f'crop-{k}' / 'env-infer.txt').read_text().split()
      for pin in unseen_b200.INFER_PINS:
        self.assertIn(pin, env, k)
    self.assertStated('the inference versions ' + ', '.join(sorted(unseen_b200.INFER_PINS)) + ',')
    v100_torch = {line for k in CROPS for line in (V100 / f'crop-{k}' / 'env-infer.txt').read_text().split()
                  if line.startswith('torch==')}
    self.assertEqual(len(v100_torch), 1)
    self.assertStated(f"has torch==2.7.1+cu128; the V100 run's has {v100_torch.pop()}")
    sizes = {(B200 / f'crop-{k}' / 'infer' / 'affinities.npy').stat().st_size for k in CROPS}
    self.assertStated(f"affinities ({', '.join(f'{x:,}' for x in sorted(sizes))} bytes per crop)")
    design = (ROOT / 'docs' / 'unseen_b200_segneuron_design.md').read_text()
    sizes = {tuple(json.loads((B200 / f'crop-{k}' / 'segneuron.json').read_text())['inference']['crop_size_zyx'])
             for k in CROPS}
    self.assertEqual(sizes, {(20, 128, 128)})
    self.assertIn('runs one 20 x 128 x 128 patch at a time', design)
    self.assertStated('runs one 20 x 128 x 128 patch at a time')
    objects = [self.j['objects_per_crop'][k] for k in CROPS]
    self.assertStated(f'own object count ({min(objects)}-{max(objects)})')

  def test_statements(self):
    """Both ways of timing, every run: the fixed statement, crops cheaper and the break-evens, each row in
    the section of its own timing definition."""
    sections = {'end_to_end': section('End to end (primary):', 'Without start-up:'),
                'without_startup': section('Without start-up:', 'Crops on which SegNeuron was cheaper, end to end:')}
    for way in unseen_b200.WAYS:
      for r in RUNS:
        c = self.j['comparisons'][way][r]
        b = c['break_even']
        self.assertIn(f"| {r} | {c['statement']} | {c['crops_segneuron_cheaper']} of 10 | {b['median']:.0f} "
                      f"({b['min']:.0f}-{b['max']:.0f}) | {c['totals']['break_even_per_crop']:.0f} |", sections[way],
                      (way, r))
    e2e = self.j['comparisons']['end_to_end']
    listed = section('Crops on which SegNeuron was cheaper, end to end:', 'Totals over all crops, end to end:')
    for r in RUNS:
      on = ', '.join(str(k) for k in e2e[r]['cheaper_on']) or 'none'
      self.assertIn(f"{unseen_b200.NAMES[r]} ({r}): {on}.", listed, r)
    self.assertEqual(e2e['F5']['crops_segneuron_cheaper'], 7)
    self.assertStated('Totals over all crops, end to end: SegNeuron '
                      f"{e2e['F5']['totals']['segneuron_seconds']:,.0f} s; "
                      + '; '.join(f"{r} {e2e[r]['totals']['run_seconds']:,.0f} s"
                                  for r in ('F1', 'F5', 'S10', 'P2', 'Q2')) + '.')

  def test_per_crop(self):
    e2e = self.j['comparisons']['end_to_end']['F5']
    for k in CROPS:
      x = self.j['segneuron_b200'][k]
      f5 = x['end_to_end'] / e2e['segneuron_over_run'][k]
      self.assertAlmostEqual(f5, EVIDENCE_F5[k], places=3)
      self.assertStated(f"| {k} | {self.j['objects_per_crop'][k]} | {host(k)} | {x['end_to_end']:.0f} | "
                        f"{x['inference_wall_seconds']:.0f} | {x['inference_seconds']:.0f} | {x['post_seconds']:.0f} | "
                        f"{f5:.0f} | {e2e['break_even_per_crop'][k]:.0f} |")

  def test_b200_against_v100(self):
    for label, d in (('B200', self.j['segneuron_b200']), ("V100 (the test's run)", self.j['segneuron_v100'])):
      t = {key: sum(d[k][key] for k in CROPS) for key in
           ('end_to_end', 'inference_wall_seconds', 'inference_seconds', 'post_seconds')}
      self.assertStated(f"| {label} | {t['end_to_end']:,.0f} s | {t['inference_wall_seconds']:,.0f} s | "
                        f"{t['inference_seconds']:,.0f} s | {t['post_seconds']:,.0f} s |")

  def test_b200_against_v100_per_crop(self):
    sb, sv = self.j['segneuron_b200'], self.j['segneuron_v100']
    for k in CROPS:
      self.assertStated(f"| {k} | {sb[k]['inference_seconds']:.0f} s | {sv[k]['inference_seconds']:.0f} s | "
                        f"{sb[k]['post_seconds']:.0f} s | {sv[k]['post_seconds']:.0f} s |")

  def test_agreement(self):
    a = self.j['agreement_pooled']
    self.assertEqual(a['objects'], sum(self.j['objects_per_crop'][k] for k in CROPS))
    self.assertStated(f"over all {a['objects']:,} objects: mean {a['mean_f1_difference']:+.6f}, largest absolute "
                      f"{a['max_abs_f1_difference']:.3f}, {a['objects_abs_difference_gt_0.1']} objects differing "
                      'by more than 0.1')
    for k in CROPS:
      g = self.j['agreement'][k]
      self.assertStated(f"| {k} | {g['mean_f1_b200']:.4f} | {g['mean_f1_v100']:.4f} | "
                        f"{g['objects_abs_difference_gt_0.1']} |")

  def test_node_disclosure(self):
    sb = self.j['segneuron_b200']
    cheap = set(self.j['comparisons']['end_to_end']['F5']['cheaper_on'])
    nodes = sorted({host(k) for k in CROPS})
    post = {n: [sb[k]['post_seconds'] for k in CROPS if host(k) == n] for n in nodes}
    loop = [sb[k]['inference_seconds'] for k in CROPS]
    self.assertStated('; '.join(f'{n}: {min(v):.0f}-{max(v):.0f} s' for n, v in post.items())
                      + f' (the GPU loop: {min(loop):.0f}-{max(loop):.0f} s on every crop)')
    self.assertStated('; '.join(
        f'{sum(1 for k in CROPS if host(k) == n and int(k) in cheap)} of '
        f'{sum(1 for k in CROPS if host(k) == n)} crops on {n}'
        for n in nodes))
    v100 = [self.j['segneuron_v100'][k]['post_seconds'] for k in CROPS]
    self.assertStated(f"On the V100 the same crops' post-processing took {min(v100):.0f}-{max(v100):.0f} s, which is "
                      'consistent with a node effect but does not show')
    self.assertStated('Whether the count of 7 depends on node assignment was not measured.')
    self.assertNotIn('follows the node', RESULTS)
    self.assertStated('Crops 41-50 had been read by every method of the test')


if __name__ == '__main__':
  unittest.main()
