"""The repository's indexes (README.md and docs/README.md) are complete and point at real records.

Every link resolves; every design, results and evidence document in docs/ is listed in the index; and every
number README.md states (outside code blocks, commit hashes and the version and licence phrases) is a number (whole, not a substring) of a
results document cited in the same table row, bullet or paragraph. That check catches a number found in none
of the cited results documents; it cannot catch a wrong number that happens to occur elsewhere in them, so
README.md quotes each number in the form its results document uses.
"""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
INDEX = (DOCS / 'README.md').read_text()
README = (ROOT / 'README.md').read_text()
# A design draft the owner keeps out of the repository (not committed), so it is not indexed.
UNINDEXED = {'correction_catalogue_design.md', 'README.md'}
# Documents that report measured results; a README number must come from one of these.
RESULTS_DOCS = {f.resolve() for f in DOCS.glob('*_results.md')} | {
    (DOCS / f).resolve() for f in ('gpu_baseline.md', 'benchmark_baseline.md', 'method_source.md')}
# Version and licence phrases, which are not results; only these exact phrases are exempt, so the same number
# elsewhere ("4.0x faster") is still checked.
# A phrase is exempt only as a whole token: not followed by a letter, digit or decimal part ('Python 3.12e3' and
# 'CC BY 4.05' are checked).
END = r'(?![\w]|[.,]\d)'
NOT_RESULTS = re.compile(r'(?<![\w.])(?:Python 3\.12|CC BY 4\.0|Apache-2\.0)' + END)
# Commit hashes in code spans (`12d680e`, at least one hex letter, so a job number is not one); other code spans are
# checked like text.
HASH = re.compile(r'`(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}`')
# The literature citations, exempt only in exactly these words (whitespace aside); any other number is checked.
CITATIONS = ('Tavakoli et al., *Nature* 642, 398–410, 2025', 'doi:10.1038/s41586-025-08985-1',
             'Tavakoli et al. 2025',
             'Januszewski et al., *Nature Methods* 15, 605–610, 2018')
LINK = re.compile(r'\[[^\]]*\]\(([^)]+)\)')
# Signed (- or −) and without backtracking, so '16GB' gives 16, not 1; only a dimension ('3D') is not a number.
NUMBER = re.compile(r'(?<![\w.])[−-]?(?>\d[\d,]*(?:\.\d+)?(?:e-?\d+)?)(?!D\b)')


def links(text: str) -> list:
  return [t for t in LINK.findall(text) if not t.startswith(('http://', 'https://', '#'))]


def units(text: str) -> list:
  """Headings, table rows, list items (bullets, sub-bullets, numbered items, with their continuation lines) and
  paragraphs of text; only a table's separator line is not a unit."""
  out = []
  for block in re.split(r'\n\s*\n', text):
    for line in block.splitlines():
      if re.match(r'\|\s*-', line):  # a table's separator line
        continue
      if re.match(r'(#|\| |\s*[-*] |\s*\d+\. )', line) or not out or re.match(r'(#|\| )', out[-1]):
        out.append(line)
      else:
        out[-1] += ' ' + line
    out.append('')
  return [u for u in out if u.strip()]


def numbers(text: str) -> list:
  """Numbers in text, including those in code spans (`0.999`; digits inside a name such as `fp16_head32` are not
  numbers), ignoring commit hashes, the version and licence phrases, link targets, the literature citations in
  CITATIONS and row labels (R1-R6)."""
  text = HASH.sub('', text)
  text = NOT_RESULTS.sub('', text)
  text = re.sub(r'\]\([^)]*\)', ']', text)
  text = re.sub(r'<https?://[^>]*>', '', text)
  text = ' '.join(text.split())
  for c in CITATIONS:
    text = re.sub(r'(?<![\w.])' + re.escape(c) + END, '', text)
  return [n.rstrip(',').replace('−', '-') for n in NUMBER.findall(text) if n.strip('−-,')]


class LinksTest(unittest.TestCase):

  def test_every_link_resolves(self):
    for base, text in ((DOCS, INDEX), (ROOT, README)):
      for target in links(text):
        self.assertTrue((base / target).resolve().exists(), f'{base.name}: {target}')

  def test_every_record_is_indexed(self):
    listed = {(DOCS / t).resolve() for t in links(INDEX)}
    for f in sorted(DOCS.glob('*')):
      if f.is_file() and f.suffix in ('.md', '.json') and f.name not in UNINDEXED:
        self.assertIn(f.resolve(), listed, f.name)

  def test_every_evidence_test_listed_exists(self):
    for t in links(INDEX):
      if t.startswith('../tests/'):
        self.assertTrue((DOCS / t).resolve().is_file(), t)


class ReadmeNumbersTest(unittest.TestCase):

  def test_numbers_come_from_the_cited_results(self):
    """Every number in README.md, outside code blocks, is a number of a results document cited in the same unit (a
    table row, a bullet with its continuation lines, or a paragraph), except commit hashes and the version and
    licence phrases in NOT_RESULTS. Designs and other documents do not count as support: a planned value cannot back a result."""
    rows = [line for line in README.splitlines() if re.match(r'\| R\d \|', line)]
    self.assertEqual(sorted({r.split('|')[1].strip() for r in rows}), [f'R{i}' for i in range(1, 7)])
    text = re.sub(r'```.*?```', '', README, flags=re.S)
    checked = 0
    for unit in units(text):
      found = numbers(unit)
      if not found:
        continue
      cited = [t for t in links(unit) if (ROOT / t).resolve() in RESULTS_DOCS]
      self.assertTrue(cited, f'states numbers but cites no results document: {unit[:80]}')
      values = set()
      for target in cited:
        values.update(numbers((ROOT / target).read_text()))
      for n in found:
        self.assertTrue(n in values, f'{n} is not a number of {cited}: {unit[:80]}')
        checked += 1
    self.assertGreater(checked, 60)

  def test_index_numbers_come_from_the_cited_documents(self):
    """Every number docs/README.md states (crop roles, seed counts, rules) is a number of a document cited in the
    same unit; here designs count, since these are the designs' own facts."""
    checked = 0
    for unit in units(INDEX):
      found = numbers(unit)
      if not found:
        continue
      cited = links(unit)
      self.assertTrue(cited, f'states numbers but cites nothing: {unit[:80]}')
      values = set()
      for target in cited:
        values.update(numbers((DOCS / target).read_text()))
      for n in found:
        self.assertTrue(n in values, f'{n} is not a number of {cited}: {unit[:80]}')
        checked += 1
    self.assertGreater(checked, 10)

  def test_exemptions_are_bound_to_their_phrases(self):
    self.assertEqual(numbers('Python 3.12, CC BY 4.0, Apache-2.0, commit `12d680e`'), [])
    self.assertEqual(numbers('4.0x faster; 2.0 s; 3.12 objects'), ['4.0', '2.0', '3.12'])
    self.assertEqual(numbers('Python 3.12. CC BY 4.0, Apache-2.0)'), [])
    for extended in ('Python 3.12e3', 'Python 3.125', 'CC BY 4.05', 'Apache-2.0.1', 'Apache-2.0x',
                     'Tavakoli et al. 20251', 'Tavakoli et al. 2025.5'):
      self.assertTrue(numbers(extended), extended)
    self.assertEqual(numbers('F1 `0.999` with `fp16_head32-s16`'), ['0.999'])
    self.assertEqual(numbers('job `7448381`, 16GB, the 90th, −0.028 in 3D'), ['7448381', '16', '90', '-0.028'])
    self.assertEqual(numbers('(Smith et al. 2020) and (Tavakoli et al. 2025)'), ['2020'])

  def test_headings_and_list_items_are_units(self):
    self.assertEqual(units('# Speed 2.4x\nText 1 [a](b)\n  - sub 3\n1. item 4\n* star 5'),
                     ['# Speed 2.4x', 'Text 1 [a](b)', '  - sub 3', '1. item 4', '* star 5'])

  def test_results_documents_are_results(self):
    self.assertGreater(len(RESULTS_DOCS), 20)
    for f in RESULTS_DOCS:
      self.assertFalse(f.name.endswith(('_design.md', '_sources.md')), f.name)


if __name__ == '__main__':
  unittest.main()
