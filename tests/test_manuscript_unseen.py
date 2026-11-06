"""The unseen-crop test numbers in the manuscript (manuscript/front.md, body.md) match the evidence.

Every number the manuscript states about crops 41-50 is formatted here from docs/unseen_evidence.json
(itself checked against the result files by tests/test_unseen_evidence.py) and looked up in the text.
The claims use the design's pre-registered labels, so the labels are checked against the evidence too.
"""

import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent
MS = ROOT / 'manuscript'
J = json.loads((ROOT / 'docs' / 'unseen_evidence.json').read_text())['judged']
FRONT = (MS / 'front.md').read_text()
BODY = (MS / 'body.md').read_text()
ABSTRACT = FRONT.split('# Abstract', 1)[1].split('\n# ', 1)[0]
TEXT = FRONT + '\n' + BODY
EVIDENCE = json.loads((ROOT / 'docs' / 'unseen_evidence.json').read_text())
# The passages that state test results; each check looks only in its own passage.
SECTION = BODY[BODY.index('## A pre-registered test on ten unseen crops'):BODY.index('# Discussion')]
DISCUSSION = BODY[BODY.index('# Discussion'):BODY.index('# Materials and methods')]
METHODS = BODY[BODY.index('## Test on unseen crops'):BODY.index('## Reproducibility')]
TEST_ABSTRACT = ABSTRACT[ABSTRACT.index('We then froze'):ABSTRACT.index('Fine-tuning the generalist')]
DATA_ROLE = re.search(r'Crops 41–50[^.]*\.', BODY).group(0)  # the test crops' role in Methods, Data


def sentences(text):
  return re.split(r'(?<=[.;])\s+(?=[A-Z(])', text)


TEST_DISCUSSION = ' '.join(x for x in sentences(DISCUSSION)
                           if re.search(r'unseen|test crop|test object|ten-crop|ten crops|418', x))


def n(x):
  return f'{x:,}'


def r(x, d):
  return f'{x:.{d}f}'


class Numbers:
  q1 = J['q1_fast_mode']
  c = q1['counts']
  objects = c['objects']
  limit = q1['limit_objects']  # 0.02 N
  q2 = J['q2_ffn_vs_segneuron']
  q3 = J['q3_smaller_ffns']
  b200 = J['speed_b200']
  v100 = J['speed_v100']
  smaller = {'student': q3['student_minus_ffn'], 'P8': q3['P8_minus_ffn'], 'P4': q3['P4_minus_ffn']}


N = Numbers


class AbstractTest(unittest.TestCase):

  def test_question_1(self):
    self.assertTrue(N.q1['passes'])
    self.assertIn(f'ten crops ({n(N.objects)} neurons)', ABSTRACT)
    d = N.q1['by_class']['dendrites']
    self.assertEqual(d['new_merges'], 1)
    self.assertIn(f"pooled over all neurons ({N.c['new_merges']} new merges and {N.c['new_early_stops']} new early "
                  f"stops, limit {N.limit}; one new merge among {d['objects']} classified dendrites), and was "
                  f'{r(N.b200["F5"]["speedup_vs_F1"], 1)} times faster', ABSTRACT)
    self.assertIn('a 2% tolerance set after its earlier failure', ABSTRACT)

  def test_segneuron(self):
    f5 = N.q2['F5_minus_segneuron']
    self.assertTrue(f5['difference_claimed'] and f5['crops_a_higher'] == 10)
    self.assertIn(f"on every test crop (mean F1 lower by {r(f5['mean'], 3)})", ABSTRACT)
    be = N.v100['totals']['break_even_end_to_end_per_crop']
    self.assertEqual(round(be, -1), 180)
    self.assertIn('cheaper than flood-filling networks tracing one neuron at a time on the same processor above about '
                  '180 neurons per 500³-voxel crop, which every test crop exceeded', ABSTRACT)
    self.assertGreater(min(J['objects_per_crop'].values()), be)

  def test_smaller(self):
    for c in N.smaller.values():
      self.assertEqual(c['verdict'], 'worse than FFN')
      self.assertEqual(c['crops_b_higher'], 10)
    speed = [N.b200['F5']['total_seconds'] / N.b200[m]['total_seconds'] for m in ('S10', 'P2', 'Q2')]
    falls = sorted(-c['mean'] for c in N.smaller.values())
    self.assertIn(f'{r(min(speed), 1)} to {r(max(speed), 1)} times faster than the fastest flood-filling setting',
                  ABSTRACT)
    self.assertIn(f'by {r(falls[0], 3)} to {r(falls[-1], 3)} mean F1, and none was shown to be within 0.02 of it',
                  ABSTRACT)
    for c in N.smaller.values():  # "worse than FFN": interval below 0, lower end at or below -0.02
      self.assertTrue(c['t_interval'][1] < 0 and c['t_interval'][0] <= -0.02)
    self.assertNotIn('beyond a margin', TEXT)
    self.assertNotIn('outside the 0.02 margin', TEXT)


class ResultsTest(unittest.TestCase):

  def test_setup(self):
    self.assertEqual(J['failures'], {})
    per = J['objects_per_crop'].values()
    self.assertIn(f'{min(per)}–{max(per)} large labelled objects per crop, {n(N.objects)} in all', SECTION)

  def test_question_1(self):
    c = N.c
    self.assertIn(f"{c['new_merges']} new merges and {c['new_early_stops']} new early stops against a limit of "
                  f'{N.limit} (2% of objects) each, mean recall {100 * c["mean_recall_change"]:+.2f} and mean precision '
                  f'{100 * c["mean_precision_change"]:+.2f} points'.replace('-', '−'), SECTION)
    self.assertIn(f"avoided {c['avoided_merges']} merges and extended {c['extensions']} traces", SECTION)
    self.assertIn(f"{r(N.b200['F5']['speedup_vs_F1'], 1)}× faster than float32 one object at a time "
                  f"({n(round(N.b200['F5']['total_seconds']))} against {n(round(N.b200['F1']['total_seconds']))} s",
                  SECTION)
    d, a = N.q1['by_class']['dendrites'], N.q1['by_class']['axons']
    self.assertEqual(d['new_merges'], 1)
    self.assertIn(f"among the {d['objects']} classified dendrites, where all five fresh-seed merges had occurred, "
                  "there was one new merge", SECTION)
    self.assertIn(f"among the {a['objects']} classified axons seven traces stopped early "
                  f"({100 * a['new_early_stops'] / a['objects']:.1f}%)", SECTION)
    self.assertEqual(a['new_early_stops'], 7)

  def test_segneuron(self):
    f5, v = N.q2['F5_minus_segneuron'], N.q2['V_minus_segneuron']
    for c in (f5, v):  # "with either FFN implementation": the same rounded values
      self.assertIn(f"+{r(c['mean'], 3)} F1 (95% interval {r(c['t_interval'][0], 3)} to {r(c['t_interval'][1], 3)})",
                    SECTION)
    self.assertIn(f"higher on {n(f5['objects']['ffn_higher'])} (B200) and {n(v['objects']['ffn_higher'])} (V100) of "
                  f"the {n(N.objects)} objects", SECTION)
    o = f5['objects']
    self.assertIn(f"Without the {o['segneuron_seeds_in_background']} objects whose seed SegNeuron placed in predicted "
                  f"background, mean F1 was {r(o['ffn_mean_f1_where_segneuron_seed_found'], 3)} against "
                  f"{r(o['segneuron_mean_f1_where_seed_found'], 3)}", SECTION)
    self.assertEqual(r(v['objects']['ffn_mean_f1_where_segneuron_seed_found'], 3),
                     r(o['ffn_mean_f1_where_segneuron_seed_found'], 3))
    p, c = J['pooled'], J['counts']
    self.assertIn(f"mostly in recall ({r(p['segneuron']['recall'], 3)} against {r(p['F5']['recall'], 3)}); it had fewer "
                  f"merged voxels than FFN ({c['segneuron']['merged_voxels'] / 1e6:.1f} M against "
                  f"{c['F5']['merged_voxels'] / 1e6:.1f} M), though more objects with a merge "
                  f"({c['segneuron']['objects_with_merge']} against {c['F5']['objects_with_merge']})", SECTION)
    self.assertIn(f"Batched fast FFN on the B200 took {n(round(N.b200['F5']['total_seconds']))} s", SECTION)
    t = N.v100['totals']
    pc = N.v100['break_even_end_to_end']
    self.assertIn('Against FFN traced one object at a time on the same V100', SECTION)
    self.assertIn(f"{round(t['break_even_end_to_end_per_crop'])} objects per crop end to end "
                  f"({round(pc['min'])}–{round(pc['max'])} across crops; "
                  f"{round(t['break_even_gpu_only_per_crop'])} for network time alone), below the smallest crop's "
                  f"{min(J['objects_per_crop'].values())} objects", SECTION)
    self.assertIn(f"it took {n(round(t['segneuron_total_seconds']))} s against FFN's "
                  f"{n(round(t['ffn_wall_seconds']))} s", SECTION)

  def test_smaller(self):
    s = N.smaller
    text = (f"the factorised student by {r(-s['student']['mean'], 3)} (95% interval "
            f"{r(-s['student']['t_interval'][1], 3)} to {r(-s['student']['t_interval'][0], 3)}), "
            f"P8 by {r(-s['P8']['mean'], 3)} ({r(-s['P8']['t_interval'][1], 3)} to {r(-s['P8']['t_interval'][0], 3)}) "
            f"and P4 by {r(-s['P4']['mean'], 3)} ({r(-s['P4']['t_interval'][1], 3)} to {r(-s['P4']['t_interval'][0], 3)})")
    self.assertIn(text, SECTION)
    p = J['q3_p8_minus_student']
    self.assertFalse(p['difference_claimed'])
    self.assertIn(f"(+{r(p['mean'], 3)}, {r(p['t_interval'][0], 3)} to +{r(p['t_interval'][1], 3)})".replace('-', '−'),
                  SECTION)
    vs_f5 = ', '.join(f"{r(N.b200['F5']['total_seconds'] / N.b200[m]['total_seconds'], 1)}×" for m in ('S10', 'P2'))
    vs_f1 = ', '.join(f"{r(N.b200[m]['speedup_vs_F1'], 1)}×" for m in ('S10', 'P2'))
    self.assertIn(f"They were {vs_f5} and {r(N.b200['F5']['total_seconds'] / N.b200['Q2']['total_seconds'], 1)}× faster "
                  f"than fast FFN ({vs_f1} and {r(N.b200['Q2']['speedup_vs_F1'], 1)}× faster than float32 FFN)", SECTION)
    ra = J['runaways']
    self.assertIn(f"{ra['S10_vs_F1']['runaway_traces']}, {ra['P2_vs_F1']['runaway_traces']} and "
                  f"{ra['Q2_vs_F1']['runaway_traces']} traces took at least three times float32 FFN's steps, against "
                  f"{ra['F5_vs_F1']['runaway_traces']} for fast FFN", SECTION)
    # "for the student and P8 the intervals end close to the margin (P8's just inside it), and on some crops the
    # difference was smaller than 0.02"
    self.assertIn("(P8's just inside it), and on some crops the difference was smaller than 0.02", SECTION)
    self.assertTrue(-0.02 < s['P8']['t_interval'][1] < 0)
    self.assertTrue(s['student']['t_interval'][1] > -0.022)
    for m in ('student', 'P8'):
      self.assertTrue(any(-0.02 < v < 0 for v in s[m]['per_crop'].values()), m)
    self.assertIn(f"held {n(J['counts']['S10']['merges'])} merges against fast FFN's {n(J['counts']['F5']['merges'])}",
                  SECTION)

  def test_bootstrap_gives_the_same_claims(self):
    self.assertIn('was computed as a check and gave the same claim for every comparison', METHODS)
    for c in (N.q2['F5_minus_segneuron'], N.q2['V_minus_segneuron']):
      self.assertGreater(c['bootstrap_interval'][0], 0)
    for c in N.smaller.values():
      self.assertTrue(c['bootstrap_interval'][1] < 0 and c['bootstrap_interval'][0] <= -0.02)
    b = J['q3_p8_minus_student']['bootstrap_interval']
    self.assertTrue(b[0] < 0 < b[1])

  def test_table_1(self):
    rows = [('FFN, float32, one at a time', 'B200', 'F1'), ('FFN, `fp16_head32`, 32 at once', 'B200', 'F5'),
            ('factorised student', 'B200', 'S10'), ('P8', 'B200', 'P2'), ('P4', 'B200', 'Q2'),
            ('FFN, `fp16_head32`, fast tracer', 'V100', 'V'), ('fine-tuned SegNeuron', 'V100', 'segneuron')]
    t = N.v100['totals']
    times = {m: N.b200[m]['total_seconds'] for m in N.b200}
    times.update({'V': t['ffn_wall_seconds'], 'segneuron': t['segneuron_total_seconds']})
    for label, gpu, m in rows:
      p, c = J['pooled'][m], J['counts'][m]
      key = f'{m}_vs_F1'
      run = str(J['runaways'][key]['runaway_traces']) if key in J['runaways'] else '–'
      self.assertIn(f"| {label} | {gpu} | {r(p['f1'], 3)} | {r(p['recall'], 3)} | {r(p['precision'], 3)} | "
                    f"{n(c['merges'])} | {run} | {n(round(times[m]))} s |", SECTION)
    self.assertIn(f'**Table 1 | Ten unseen crops (41–50), {n(N.objects)} objects.**', SECTION)


class DiscussionTest(unittest.TestCase):

  def test_numbers(self):
    c, s = N.c, N.smaller
    self.assertIn(f"pooled over all large objects ({c['new_merges']} new merges and {c['new_early_stops']} new early "
                  f"stops against {N.limit} allowed), with one new merge among {N.q1['by_class']['dendrites']['objects']} classified dendrites", DISCUSSION)
    self.assertIn(f"they were {r(-s['student']['mean'], 3)} and {r(-s['P8']['mean'], 3)} below FFN and P4 "
                  f"{r(-s['P4']['mean'], 3)}, each lower on every crop", DISCUSSION)
    f5 = N.b200['F5']['total_seconds']
    self.assertIn(f"{r(f5 / N.b200['S10']['total_seconds'], 1)}×, {r(f5 / N.b200['P2']['total_seconds'], 1)}× and "
                  f"{r(f5 / N.b200['Q2']['total_seconds'], 1)}× faster than FFN itself "
                  f"({r(N.b200['P2']['speedup_vs_F1'], 1)}× faster than single-precision FFN", DISCUSSION)
    t = N.v100['totals']
    self.assertIn(f"{r(t['ffn_wall_seconds'] / t['objects'], 2)} s per test object", DISCUSSION)
    self.assertIn(f"({r(N.b200['F5']['seconds_per_object'], 2)} and {r(N.b200['P2']['seconds_per_object'], 2)} s per "
                  "test object)", DISCUSSION)
    o = J['ffn_v100_overlap'].values()
    share = sum(x['voxels_in_two_or_more'] for x in o) / sum(x['voxels_in_any_mask'] for x in o)
    self.assertIn(f"{100 * share:.1f}% on the test crops", DISCUSSION)
    self.assertIn(f"the gap was {r(N.q2['F5_minus_segneuron']['mean'], 3)}, with FFN ahead on every crop, and the "
                  f"break-even against FFN traced one object at a time on the V100 was "
                  f"{round(N.v100['totals']['break_even_end_to_end_per_crop'])} objects per crop", DISCUSSION)
    self.assertIn(f"{J['object_ids_in_more_than_one_crop']} object IDs were scored in more than one crop", DISCUSSION)
    self.assertIn(f"more than 0.1 F1 on {N.q2['V_vs_F5_agreement']['objects_abs_difference_gt_0.1']} objects", DISCUSSION)

  def test_nodes_and_non_finite(self):
    hosts_b200 = len(J['devices']['b200_hosts'])
    hosts_v100 = len(set(EVIDENCE['v100_devices']['jobs'].values()))
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    self.assertIn(f'{words[hosts_b200]} B200 and {words[hosts_v100]} V100 nodes', DISCUSSION)
    for crop, runs in J['nonfinite_outputs'].items():
      for m in ('F5', 'S10', 'P2', 'Q2'):
        self.assertEqual(runs[m], 0, (crop, m))
    self.assertNotIn('segneuron', J['failures'])
    self.assertIn('non-finite outputs were counted for the four float16 B200 runs and SegNeuron, and there were none, '
                  'while the float32 run and the V100 FFN do not count them', SECTION)

  def test_conclusion(self):
    self.assertIn('On ten crops that no part of the work had read, the accelerated FFN passed a rule fixed for them that '
                  'allows at most 2% of neurons to gain a new merge and 2% to stop early, and mean recall and precision to '
                  'fall by at most one point, pooled over all neurons, though the rule was set after an earlier failure',
                  DISCUSSION)
    self.assertTrue(N.q1['passes'])

  def test_data_role(self):
    self.assertIn('the last ten positions of the order, were the test set for every method; their role was fixed and the '
                  'analysis written before they were fetched.', DATA_ROLE)
    self.assertEqual(list(map(int, J['objects_per_crop'])), list(range(41, 51)))

  def test_no_claim_that_the_test_is_still_to_come(self):
    for stale in ('remains to be made', 'has not yet been made', 'still have to be compared',
                  'reserved as the test set'):
      self.assertNotIn(stale, TEXT)


def leaves(x):
  if isinstance(x, bool) or x is None or isinstance(x, str):
    return []
  if isinstance(x, (int, float)):
    return [float(x)]
  if isinstance(x, dict):
    return [v for y in x.values() for v in leaves(y)]
  return [v for y in x for v in leaves(y)]


def derived():
  """Numbers the text derives from the evidence: speed ratios, per-object times, shares, counts."""
  b, t = J['speed_b200'], J['speed_v100']['totals']
  out = [b['F5']['total_seconds'] / b[m]['total_seconds'] for m in b]
  out += [t['ffn_wall_seconds'] / t['objects'], 100 * J['q1_fast_mode']['by_class']['axons']['new_early_stops'] /
          J['q1_fast_mode']['by_class']['axons']['objects']]
  o = J['ffn_v100_overlap'].values()
  out.append(100 * sum(x['voxels_in_two_or_more'] for x in o) / sum(x['voxels_in_any_mask'] for x in o))
  out += [len(J['crop_mean_f1']['F1'])]  # ten crops
  out.append(round(t['break_even_end_to_end_per_crop'], -1))  # 'about 180' in the abstract
  return out


VALUES = leaves(J) + derived()
# Constants of the design and protocol (docs/unseen_design.md), crop and date labels, and the tuning-crop
# numbers that test sentences quote for comparison (docs/headtohead_results.md, docs/distil_results.md):
CONSTANTS = {
    '0.02', '−0.02', '2', '1', '−1', '0.1', '1,000', '95', '9', '10,000', '32', '16', '0.6',  # rules, settings
    '41', '50', '11', '12', '300', '500', '2026', '4', '256',  # crops, seed sets, crop size, date, SHA-256
    '0.07', '2.53', '0.45', '0.25', '2.8', '3.2',  # tuning: crop-12 gap and V100 time; crop-11 B200 times; overlap
}


def explained(token: str) -> bool:
  if token in CONSTANTS:
    return True
  t = token.replace('−', '-').replace('+', '').replace(',', '')
  x = float(t)
  decimals = len(t.split('.')[1]) if '.' in t else 0
  for v in VALUES:
    for scale in (1, 100, 1e-6):  # plain, percentage points or percent, millions
      for sign in (1, -1):
        if f'{sign * v * scale:.{decimals}f}' == f'{x:.{decimals}f}':
          return True
  return False


class EveryNumberTest(unittest.TestCase):
  """Every number in the passages that state test results comes from the evidence or is a stated constant."""

  def test_every_number_is_explained(self):
    number = re.compile(r'(?<![\w.])[−+-]?\d[\d,]*(?:\.\d+)?')
    for name, text in (('abstract', TEST_ABSTRACT), ('results', SECTION), ('discussion', TEST_DISCUSSION),
                       ('methods', METHODS), ('data', DATA_ROLE)):
      for token in number.findall(text):
        token = token.rstrip(',')
        self.assertTrue(explained(token), f'{name}: {token} is not in the evidence')


if __name__ == '__main__':
  unittest.main()
