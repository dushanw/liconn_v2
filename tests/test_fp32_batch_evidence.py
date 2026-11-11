"""The float32 batching results (docs/fp32_batch_results.md) match their evidence and their result files.

The judge (examples/judge_fp32_batch.py) is re-run here on the copied result files in the data cache,
and docs/fp32_batch_evidence.json is compared with its output. The page's numbers are checked one by
one against the evidence.
"""

import json
from pathlib import Path
import sys
import unittest

from liconn_v2 import data
from liconn_v2 import unseen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'examples'))
import judge_fp32_batch  # pylint: disable=g-import-not-at-top,wrong-import-position

EVIDENCE = json.loads((ROOT / 'docs' / 'fp32_batch_evidence.json').read_text())
PAGE = (ROOT / 'docs' / 'fp32_batch_results.md').read_text()
RUN = data.cache_dir() / 'runs' / 'fp32batch-20261109'
NODE5601 = RUN / 'widths-each-28611579-node5601'
MAIN = [NODE5601 / n / 'widths.json' for n in ('F32x1', 'F32x8', 'F32x32', 'F32x64', 'H16x1', 'H16x8', 'H16x32')] + [
    RUN / 'widths-each' / 'H16x64' / 'widths.json']
FIRST_JOB = [RUN / 'widths' / 'widths.json']
OTHER_NODE = [RUN / 'widths-each' / 'H16x32' / 'widths.json']
UNSEEN_B200 = data.cache_dir() / 'runs' / 'unseen-20261104' / 'b200'
W = EVIDENCE['widths']['report']
T = EVIDENCE['test']['report']


def num(x: float, places: int) -> str:
  return f'{x:,.{places}f}'


class EvidenceTest(unittest.TestCase):

  def test_evidence_is_the_judge_s_output(self):
    rebuilt = judge_fp32_batch.build(MAIN, RUN / 'test', UNSEEN_B200, FIRST_JOB, OTHER_NODE)
    self.assertEqual(rebuilt, EVIDENCE)

  def test_all_eight_settings_and_ten_crops(self):
    self.assertEqual(W['missing_runs'], [])
    self.assertEqual(T['missing_crops'], [])
    self.assertEqual(EVIDENCE['widths']['replaced'], {})
    self.assertEqual(EVIDENCE['widths']['run_source']['H16x64'], 'widths-each/H16x64/widths.json')
    for n in ('F32x1', 'H16x32'):
      self.assertTrue(W['same_as_timing_study'][n]['same'], n)


class PageTest(unittest.TestCase):

  def assert_on_page(self, text: str):
    self.assertIn(text, PAGE)

  def test_crop_11_table(self):
    for n, r in W['runs'].items():
      ident = W['identity_vs_width_1'].get(n)
      row = (f"({n}) | {num(r['crop_seconds'], 1)} | {r['ms_per_step']:.2f} | {r['steps']:,} | "
             f"{r['mean_f1']:.3f} | {r['merges_ge_1000']} |")
      self.assert_on_page(row)
      if ident is not None:
        self.assertEqual(ident['identical'], 512, n)
    self.assertEqual({r['nonfinite_outputs'] for n, r in W['runs'].items() if n.startswith('H16')}, {0})

  def test_fingerprints_and_fastest_width(self):
    for n in ('F32x1', 'H16x32'):
      self.assert_on_page(f"(`{W['fingerprints'][n][:8]}...`)")
    self.assertEqual(W['fastest_fp32_width'], 64)
    self.assert_on_page(f"({num(W['runs']['F32x64']['crop_seconds'], 1)} s against {num(W['runs']['F32x32']['crop_seconds'], 1)} s at width 32)")

  def test_smoke_runs_quoted(self):
    first = json.loads((RUN / 'widths-smoke20' / 'widths.json').read_text())['runs']['H16x1']['summary']
    again = json.loads((RUN / 'widths-each-smoke20' / 'H16x1' / 'widths.json').read_text())['runs']['H16x1']['summary']
    self.assertEqual((first['objects'], again['objects']), (20, 20))
    self.assert_on_page(f"took {first['ms_per_step']:.2f} ms per step on node5601 (`widths-smoke20`) and "
                        f"{again['ms_per_step']:.2f} ms in a fresh process on\nnode5501")

  def test_split(self):
    self.assert_on_page(f"({W['fast_ffn_over_fp32_one_at_a_time_per_step']:.2f}x in all; "
                        f"{W['batching_speedup_per_step']['F32x32']:.2f}x then {W['precision_speedup_per_step_at_width']['32']:.2f}x, or "
                        f"{W['precision_speedup_per_step_at_width']['1']:.2f}x then {W['batching_speedup_per_step']['H16x32']:.2f}x)")
    self.assert_on_page(f"{W['fast_ffn_over_fp32_one_at_a_time']:.2f}x faster")
    self.assert_on_page(f"float32 x1 -> x32: {W['batching_speedup']['F32x32']:.2f}x")
    self.assert_on_page(f"float32 x32 -> fast FFN x32: {W['precision_speedup_at_width']['32']:.2f}x")
    self.assert_on_page(f"float32 x1 -> fast FFN x1: {W['precision_speedup_at_width']['1']:.2f}x")
    self.assert_on_page(f"fast FFN x1 -> x32: {W['batching_speedup']['H16x32']:.2f}x")
    self.assert_on_page(f"float32 {W['batching_speedup']['F32x8']:.2f}x, fast FFN {W['batching_speedup']['H16x8']:.2f}x")
    gain = [W['batching_speedup'][f'{p}x{w}'] / W['batching_speedup'][f'{p}x8'] - 1 for p in ('F32', 'H16') for w in (32, 64)]
    self.assert_on_page(f'add {100 * min(gain):.0f}% to {100 * max(gain):.0f}% more')

  def test_replaced_and_other_node(self):
    first = EVIDENCE['first_job']['report']['runs']
    self.assertEqual(sorted(first), ['F32x1', 'F32x32', 'F32x64', 'F32x8', 'H16x1', 'H16x8'])
    self.assert_on_page(f"{first['H16x1']['ms_per_step']:.2f} ms per step, against {W['runs']['H16x1']['ms_per_step']:.2f} above")
    worst = max(first[n]['crop_seconds'] / W['runs'][n]['crop_seconds'] - 1 for n in first if n.startswith('F32'))
    self.assertLess(worst, 0.03)
    other = EVIDENCE['other_node']['report']
    self.assert_on_page(f"{other['runs']['H16x32']['ms_per_step']:.2f} ms per step against {W['runs']['H16x32']['ms_per_step']:.2f}")
    self.assertEqual(other['fingerprints']['H16x32'], W['fingerprints']['H16x32'])
    same_node = other['runs']['H16x32']['crop_seconds'] / W['runs']['H16x64']['crop_seconds']
    self.assert_on_page(f"H16x32 -> H16x64 is {same_node:.2f}x")

  def test_test_crops(self):
    tot, per = T['total_seconds'], T['seconds_per_object']
    for r in ('F1', 'F32x32', 'F5'):
      self.assert_on_page(f"({r}) | {num(tot[r], 0)} | {per[r]:.2f} |")
    self.assertEqual(T['objects'], 4579)
    self.assertEqual(T['identical_objects'], 4579)
    pooled, counts = T['pooled']['F32x32'], T['counts']['F32x32']
    self.assertEqual(T['pooled']['F1'], pooled)
    self.assertEqual(T['counts']['F1'], counts)
    self.assert_on_page(f"pooled F1 {pooled['f1']:.3f}, recall {pooled['recall']:.3f}, precision {pooled['precision']:.3f}; "
                        f"{counts['objects_with_merge']} objects with a merge, {counts['merged_voxels'] / 1e6:.1f} million merged")
    self.assertEqual(T['changes_vs_F1']['new_merges'] + T['changes_vs_F1']['new_early_stops'], 0)
    self.assertEqual(T['runaways_vs_F1']['runaway_traces'], 0)
    b = [v['batching_speedup_fp32'] for v in T['per_crop'].values()]
    p = [v['precision_speedup_at_32'] for v in T['per_crop'].values()]
    self.assert_on_page(f"gave {T['batching_speedup_fp32']:.2f}x ({min(b):.2f}x to {max(b):.2f}x per crop)")
    self.assert_on_page(f"a further {T['precision_speedup_at_32']:.2f}x ({min(p):.2f}x to {max(p):.2f}x)")
    self.assert_on_page(f"the test's {T['fast_ffn_over_fp32_one_at_a_time']:.2f}x")
    e2e, ws = T['segneuron_break_even']['end_to_end'], T['segneuron_break_even']['without_startup']
    self.assertEqual(e2e['crops_segneuron_cheaper'], 10)
    self.assert_on_page(f"{e2e['break_even']['median']:.0f} objects per crop (median; "
                        f"{e2e['break_even']['min']:.0f} to {e2e['break_even']['max']:.0f})")
    self.assert_on_page(f"{ws['break_even']['median']:.0f}\n({ws['break_even']['min']:.0f} to {ws['break_even']['max']:.0f})")
    self.assert_on_page(f"break-even is {e2e['totals']['break_even_per_crop']:.0f} objects per crop end\n"
                        f"to end and {ws['totals']['break_even_per_crop']:.0f} without start-up")

  def test_per_crop_table(self):
    hosts = EVIDENCE['test']['hosts']
    sec, ms = T['crop_seconds'], T['ms_per_step']
    for k, v in T['per_crop'].items():
      self.assert_on_page(f"| {k} | {hosts[k].replace('-c1', '')} | {num(sec['F1'][k], 1)} | {num(sec['F32x32'][k], 1)} | "
                          f"{num(sec['F5'][k], 1)} | {ms['F32x32'][k]:.2f} | {v['batching_speedup_fp32']:.2f} | "
                          f"{v['precision_speedup_at_32']:.2f} | {v['mean_f1']:.3f} |")
    slowest = max(ms['F32x32'], key=ms['F32x32'].get)
    self.assertEqual((slowest, hosts[slowest]), ('41', 'node5501-c1'))
    rest = [v for k, v in ms['F32x32'].items() if k != '41']
    self.assert_on_page(f"slowest F32x32 step ({ms['F32x32']['41']:.2f} ms, against {min(rest):.2f} to {max(rest):.2f}")

  def test_hosts(self):
    hosts = EVIDENCE['test']['hosts']
    self.assertEqual(sorted(k for k, h in hosts.items() if h == 'node5501-c1'), ['41', '42'])
    self.assertEqual(sorted(hosts), sorted(str(k) for k in unseen.CROPS))


if __name__ == '__main__':
  unittest.main()
