"""Structural checks of the manuscript source (manuscript/): citations, reference cap, figures.

These check form only (every citation resolves, at most 50 references and a 300-word abstract as Light: Science &
Applications allows, every figure has a legend and figures are first cited in order); they do not
check that the numbers in the text match the results documents. The exception is test_fig2c_inputs,
which checks Fig. 2c's source runs against job D's record.
"""

import json
from pathlib import Path
import re
import unittest

from liconn_v2 import data

ROOT = Path(__file__).resolve().parent.parent
MS = ROOT / 'manuscript'
MAX_REFERENCES = 50
MAX_ABSTRACT_WORDS = 300


def text():
  return (MS / 'front.md').read_text() + '\n\n' + (MS / 'body.md').read_text()


def supplement():
  return (MS / 'supplement.md').read_text()


def bib_keys():
  return set(re.findall(r'^@\w+\{([^,\s]+),', (MS / 'refs.bib').read_text(), flags=re.M))


def cited_keys(src):
  keys = []
  for group in re.findall(r'\[(@[^\]]+)\]', src):
    for k in group.split(';'):
      k = k.strip().lstrip('@')
      if k not in keys:
        keys.append(k)
  return keys


def main_text(src):
  """Everything before the figure legends."""
  return src.split('# Figure legends', 1)[0]


class ManuscriptTest(unittest.TestCase):

  def test_every_citation_resolves(self):
    missing = [k for k in cited_keys(text() + supplement()) if k not in bib_keys()]
    self.assertEqual(missing, [])

  def test_reference_cap(self):  # main text and supplement share one numbered list
    self.assertLessEqual(len(cited_keys(text() + supplement())), MAX_REFERENCES)

  def test_bib_keys_unique(self):
    keys = re.findall(r'^@\w+\{([^,\s]+),', (MS / 'refs.bib').read_text(), flags=re.M)
    self.assertEqual(len(keys), len(set(keys)))

  def test_abstract_length(self):
    abstract = (MS / 'front.md').read_text().split('# Abstract', 1)[1].split('\n# ', 1)[0]
    self.assertLessEqual(len(abstract.split()), MAX_ABSTRACT_WORDS)

  def test_figures_have_legends_and_are_cited_in_order(self):
    src = text()
    legends = [int(n) for n in re.findall(r'^\*\*Fig\. (\d+) \|', src, flags=re.M)]
    self.assertEqual(legends, list(range(1, len(legends) + 1)))
    first = []
    for m in re.finditer(r'(?<!Supplementary )Figs?\. ?(\d+)', main_text(src)):
      n = int(m.group(1))
      if n not in first:
        first.append(n)
    self.assertEqual(first, sorted(first))
    self.assertEqual(set(first), set(legends))

  def test_supplementary_figures_and_notes_exist(self):
    sup = supplement()
    legends = [int(n) for n in re.findall(r'^\*\*Supplementary Fig\. (\d+) \|', sup, flags=re.M)]
    self.assertEqual(legends, list(range(1, len(legends) + 1)))
    notes = [int(n) for n in re.findall(r'^## Supplementary Note (\d+) \|', sup, flags=re.M)]
    self.assertEqual(notes, list(range(1, len(notes) + 1)))
    cited_figs = {int(n) for n in re.findall(r'Supplementary Figs?\. ?(\d+)', text() + sup)}
    self.assertEqual(cited_figs, set(legends))
    cited_notes = {int(n) for n in re.findall(r'Supplementary Note (\d+)', text())}
    self.assertEqual(cited_notes, set(notes))

  def test_fig2c_inputs(self):
    """Fig. 2c plots FFN runs F1, F7, F6 and F5 of job D from docs/distil_timing_evidence.json, with the
    settings its legend states, and those values are job D's recorded ones."""
    job = json.loads((ROOT / 'docs' / 'distil_timing_evidence.json').read_text())['jobs']['D (27684986)']
    settings = {'F1': ('fp32', 1, 1.0), 'F7': ('fp16', 1, 0.0625), 'F6': ('fp16', 32, 0.0625),
                'F5': ('fp16_head32', 32, 0.0625)}  # float16 runs divide the residual stream by 16
    raw = json.loads((data.cache_dir() / 'runs/distil-20261031/timing-dilated-r1-d/timing.json').read_text())
    for k, (fmt, width, scale) in settings.items():
      r = job['runs'][k]
      self.assertEqual((r['network'], r['format'], r['width'], r['stream_scale']), ('ffn', fmt, width, scale), k)
      self.assertEqual(raw['runs'][k]['summary']['format'], fmt, k)
      self.assertEqual(raw['runs'][k]['summary']['width'], width, k)
      self.assertAlmostEqual(job['crop_seconds'][k], raw['crop_seconds'][k], delta=1e-5 * raw['crop_seconds'][k])
      self.assertAlmostEqual(r['mean_f1'], raw['runs'][k]['summary']['mean_f1'], delta=1e-6)
    self.assertIn("tim = ev('distil_timing')['jobs']['D (27684986)']", (MS / 'figures' / 'make_figures.py').read_text())

  def test_build_inlines_the_figures_the_script_saves(self):
    build = (MS / 'build.py').read_text()
    saved = set(re.findall(r"save\(fig, '([\w]+)'\)", (MS / 'figures' / 'make_figures.py').read_text()))
    inlined = set(re.findall(r"\d+: '([\w]+)'", build))
    self.assertEqual(inlined, saved)
    figs = {int(n) for n in re.findall(r'^\*\*Fig\. (\d+) \|', text(), flags=re.M)}
    sfigs = {int(n) for n in re.findall(r'^\*\*Supplementary Fig\. (\d+) \|', supplement(), flags=re.M)}
    self.assertEqual(len(inlined), len(figs) + len(sfigs))

if __name__ == '__main__':
  unittest.main()
