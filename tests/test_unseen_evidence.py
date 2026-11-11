"""The unseen-crop test (docs/unseen_results.md) matches its evidence and its result files.

The judge (examples/judge_unseen.py) is re-run here on the copied result files in the data cache,
and every judged number in docs/unseen_evidence.json is compared with it, not with the extract
itself. The result files and the staged source are checked by sha256, and the numbers stated in
docs/unseen_results.md are checked against the evidence.
"""

import hashlib
import json
from pathlib import Path
import sys
import unittest

from liconn_v2 import data
from liconn_v2 import unseen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'examples'))
import unseen_evidence  # pylint: disable=g-import-not-at-top,wrong-import-position

EVIDENCE = json.loads((ROOT / 'docs' / 'unseen_evidence.json').read_text())
RESULTS = (ROOT / 'docs' / 'unseen_results.md').read_text()
RUN = unseen_evidence.run_dir()
REL = 1e-5  # evidence values are rounded to 6 significant digits


def close(a, b, where=''):
  """a (recomputed) matches b (evidence) to 6 significant digits, recursively."""
  if isinstance(b, dict):
    assert isinstance(a, dict) and set(map(str, a)) == set(b), where
    a = {str(k): v for k, v in a.items()}
    for k in b:
      close(a[k], b[k], f'{where}.{k}')
  elif isinstance(b, list):
    assert isinstance(a, list) and len(a) == len(b), where
    for i, (x, y) in enumerate(zip(a, b)):
      close(x, y, f'{where}[{i}]')
  elif isinstance(b, float):
    assert abs(a - b) <= REL * max(abs(b), 1e-12), f'{where}: {a} != {b}'
  else:
    assert a == b, f'{where}: {a} != {b}'


class EvidenceTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.rebuilt = unseen_evidence.build(RUN)

  def test_judged_numbers_are_the_judge_s(self):
    close(self.rebuilt['judged'], EVIDENCE['judged'], 'judged')

  def test_result_files_and_source_are_pinned(self):
    self.assertEqual(len(EVIDENCE['result_files_sha256']), 7 * len(unseen.CROPS))
    for f, sha in EVIDENCE['result_files_sha256'].items():
      self.assertEqual(data.sha256_file(RUN / f), sha, f)
    for k in ('source_list_sha256', 'source_files', 'all_tasks_same_source', 'design_sha256', 'manifest_sha256',
              'judge_output_sha256', 'v100_devices', 'source_sha256'):
      self.assertEqual(self.rebuilt[k], EVIDENCE[k], k)

  def test_design_and_crops_are_the_ones_fetched(self):
    manifest = unseen.load_manifest()
    self.assertEqual(manifest['design_sha256'], unseen.design_sha256())
    self.assertEqual(EVIDENCE['design_sha256'], unseen.design_sha256())
    self.assertEqual([c['index'] for c in manifest['crops']], list(unseen.CROPS))
    self.assertEqual({c['index']: tuple(c['xyz']) for c in manifest['crops']}, unseen.CROP_XYZ)

  def test_the_code_that_ran_is_this_code(self):
    """The evidence's file-by-file source is the list every job recorded, and every file still has
    the sha256 it had when the jobs ran."""
    mapping = EVIDENCE['source_sha256']
    self.assertEqual(len(mapping), EVIDENCE['source_files'])
    listed = unseen_evidence.source_list(mapping)
    self.assertEqual(hashlib.sha256(listed.encode()).hexdigest(), EVIDENCE['source_list_sha256'])
    self.assertEqual(listed, (RUN / 'v100' / 'source_sha256.txt').read_text())
    for rel, sha in mapping.items():
      # README.md was rewritten after the test (owner's decision, 2026-11-07); the version that ran is kept, exactly,
      # in docs/records/README_as_staged_for_unseen_test.md and checked there. Every other file is checked live.
      path = ROOT / 'docs' / 'records' / 'README_as_staged_for_unseen_test.md' if rel == 'README.md' else ROOT / rel
      self.assertEqual(data.sha256_file(path), sha, rel)


def pct(x):
  return f'{x:+.4f}'


class ResultsPageTest(unittest.TestCase):
  """Key numbers of docs/unseen_results.md, formatted from the evidence."""

  j = EVIDENCE['judged']

  def assertStated(self, text):
    self.assertIn(text, RESULTS)

  def test_question_1(self):
    q = self.j['q1_fast_mode']
    c = q['counts']
    self.assertTrue(q['passes'])
    self.assertEqual(self.j['failures'], {})
    self.assertStated(f"| objects (N) | {c['objects']:,} |")
    self.assertStated(f"| new merges | {c['new_merges']} | {q['limit_objects']} | yes |")
    self.assertStated(f"| new early stops | {c['new_early_stops']} | {q['limit_objects']} | yes |")
    self.assertStated(f"| mean recall change | {pct(c['mean_recall_change'])} |")
    self.assertStated(f"| mean precision change | {pct(c['mean_precision_change'])} |")
    self.assertStated(f"Avoided merges {c['avoided_merges']}, extensions {c['extensions']}")
    for name, row in q['by_class'].items():
      self.assertStated(f"| {name} | {row['objects']:,} | {row['new_merges']} | {row['new_early_stops']} | "
                        f"{row['avoided_merges']} | {row['extensions']} |")
    for k, row in q['per_crop'].items():
      self.assertStated(f"| {k} | {row['objects']} | {row['new_merges']} | {row['new_early_stops']} | "
                        f"{row['avoided_merges']} | {row['extensions']} |")
    m = q['reference_already_merged']
    self.assertStated(f"Among the {m['objects']} objects on which F1 already merged")
    self.assertEqual(m['new_merges_among_them'], 1)

  def test_non_finite_outputs(self):
    """Zero wherever counted (F5, S10, P2, Q2 on every crop; SegNeuron would be a failure), and
    the page claims nothing for F1 and V, which do not count them."""
    for k, runs in self.j['nonfinite_outputs'].items():
      for m in unseen.COUNTED_NONFINITE:
        self.assertEqual(runs[m], 0, (k, m))
      self.assertIsNone(runs['F1'], k)
    self.assertNotIn('segneuron', self.j['failures'])
    self.assertStated('The float32 run F1 and run V (float16 on the V100) do not count them')

  def test_crop_means(self):
    rows = {'F1': 'FFN float32 (F1)', 'F5': 'FFN fast (F5)', 'V': 'FFN on the V100 (V)', 'S10': 'student (S10)',
            'P2': 'P8 (P2)', 'Q2': 'P4 (Q2)', 'segneuron': 'SegNeuron'}
    for m, label in rows.items():
      v = self.j['crop_mean_f1'][m]
      self.assertStated(f"| {label} | " + ' | '.join(f'{v[str(k)]:.3f}' for k in unseen.CROPS) + ' |')

  def test_comparisons(self):
    q2, q3 = self.j['q2_ffn_vs_segneuron'], self.j['q3_smaller_ffns']
    rows = [('F5 - SegNeuron', q2['F5_minus_segneuron'], 'FFN more accurate'),
            ('V - SegNeuron', q2['V_minus_segneuron'], 'FFN more accurate'),
            ('student - F5', q3['student_minus_ffn'], 'worse than FFN'),
            ('P8 - F5', q3['P8_minus_ffn'], 'worse than FFN'),
            ('P4 - F5', q3['P4_minus_ffn'], 'worse than FFN'),
            ('P8 - student', self.j['q3_p8_minus_student'], 'no difference claimed')]
    for label, c, claim in rows:
      t, b = c['t_interval'], c['bootstrap_interval']
      self.assertStated(f"| {label} | {pct(c['mean'])} | [{t[0]:.4f}, {t[1]:.4f}] | [{b[0]:.4f}, {b[1]:.4f}] | "
                        f"{c['crops_a_higher']} / {c['crops_b_higher']} | {claim} |")
    for c in (q3['student_minus_ffn'], q3['P8_minus_ffn'], q3['P4_minus_ffn']):
      self.assertEqual(c['verdict'], 'worse than FFN')
    self.assertTrue(q2['F5_minus_segneuron']['difference_claimed'] and q2['V_minus_segneuron']['difference_claimed'])
    self.assertTrue(q2['claims_agree'])
    self.assertFalse(self.j['q3_p8_minus_student']['difference_claimed'])
    a = q2['V_vs_F5_agreement']
    self.assertStated(f"differ by more than 0.1 on {a['objects_abs_difference_gt_0.1']} objects "
                      f"(largest {a['max_abs_f1_difference']:.3f})")
    o = q2['F5_minus_segneuron']['objects']
    self.assertStated(f"FFN higher on {o['ffn_higher']:,} (F5) and {q2['V_minus_segneuron']['objects']['ffn_higher']:,} (V)")
    self.assertStated(f"background for {o['segneuron_seeds_in_background']} objects")

  def test_pooled_and_counts(self):
    for m, label in (('F1', 'F1'), ('F5', 'F5'), ('V', 'V'), ('S10', 'S10'), ('P2', 'P2'), ('Q2', 'Q2'),
                     ('segneuron', 'SegNeuron')):
      p, c = self.j['pooled'][m], self.j['counts'][m]
      self.assertStated(f"| {label} | {p['recall']:.3f} | {p['precision']:.3f} | {p['f1']:.3f} | {c['merges']:,} | "
                        f"{c['objects_with_merge']} | {c['merged_voxels'] / 1e6:.1f} M |")
    r = self.j['runaways']
    self.assertStated(f"| F5 | {r['F5_vs_F1']['runaway_traces']} | - |")
    for m in ('S10', 'P2', 'Q2'):
      self.assertStated(f"| {m} | {r[f'{m}_vs_F1']['runaway_traces']} | {r[f'{m}_vs_F5']['runaway_traces']} |")
    self.assertStated(f"{self.j['object_ids_in_more_than_one_crop']} object IDs")
    for k, v in r.items():
      self.assertEqual(v['reference_stopped_at_first_step'], [206992, 820108], k)
      self.assertEqual(v['of_them_counted_as_runaways'], [] if k == 'F5_vs_F1' else [206992, 820108], k)
    self.assertStated('206992 (crop 42) and 820108 (crop 49)')

  def test_per_crop_differences(self):
    rows = [('F5 - SegNeuron', self.j['q2_ffn_vs_segneuron']['F5_minus_segneuron']),
            ('V - SegNeuron', self.j['q2_ffn_vs_segneuron']['V_minus_segneuron']),
            ('student - F5', self.j['q3_smaller_ffns']['student_minus_ffn']),
            ('P8 - F5', self.j['q3_smaller_ffns']['P8_minus_ffn']),
            ('P4 - F5', self.j['q3_smaller_ffns']['P4_minus_ffn']),
            ('P8 - student', self.j['q3_p8_minus_student'])]
    for label, c in rows:
      self.assertStated(f'| {label} | ' + ' | '.join(pct(c['per_crop'][str(k)]) for k in unseen.CROPS) + ' |')

  def test_v100_devices(self):
    d = EVIDENCE['v100_devices']
    self.assertEqual(d['gpus'], ['Tesla V100-SXM2-16GB, 580.178.04'])
    self.assertEqual(len(set(d['jobs'].values())), 7)
    self.assertStated('different V100 nodes (seven in all')

  def test_speed(self):
    for m, s in self.j['speed_b200'].items():
      self.assertStated(f"| {m} | {s['total_seconds']:,.0f} s | {s['mean_crop_seconds']:,.0f} s | ")
      self.assertStated(f" | {s['speedup_vs_F1']:.1f}x |")
      self.assertStated(f'| {m} | ' + ' | '.join(f"{s['crop_seconds'][str(k)]:,.0f}" for k in unseen.CROPS) + ' |')
    v = self.j['speed_v100']
    for way, label in (('end_to_end', 'end to end (primary)'), ('gpu_only', 'GPU only')):
      b = v[f'break_even_{way}']
      self.assertStated(f"| {label} | {b['median']:.0f} ({b['min']:.0f}-{b['max']:.0f}) | "
                        f"{v['totals'][f'break_even_{way}_per_crop']:.0f} |")
    self.assertEqual(len(self.j['devices']['b200_hosts']), 4)


if __name__ == '__main__':
  unittest.main()
