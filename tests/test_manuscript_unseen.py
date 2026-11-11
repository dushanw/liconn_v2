"""The unseen-crop test numbers in the manuscript (manuscript/front.md, body.md) match the evidence.

Every number the manuscript states about crops 41-50 is formatted here from docs/unseen_evidence.json and,
for SegNeuron timed on the B200, docs/unseen_b200_evidence.json (each checked against its result files by
tests/test_unseen_evidence.py and tests/test_unseen_b200_evidence.py) and looked up in the text. The claims
use the designs' pre-registered labels, so the labels are checked against the evidence too. Each Results
section ends with its test result; those paragraphs, the section comparing all methods (with Table 1), the
test sentences of the abstract and discussion, and the test's methods are the passages checked here.
"""

import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent
MS = ROOT / 'manuscript'
EVIDENCE = json.loads((ROOT / 'docs' / 'unseen_evidence.json').read_text())
J = EVIDENCE['judged']
B = json.loads((ROOT / 'docs' / 'unseen_b200_evidence.json').read_text())['judged']
FRONT = (MS / 'front.md').read_text()
BODY = (MS / 'body.md').read_text()
ABSTRACT = FRONT.split('# Abstract', 1)[1].split('\n# ', 1)[0]
TEXT = FRONT + '\n' + BODY
RESULTS = BODY[BODY.index('# Results'):BODY.index('# Discussion')]


def paragraphs(text):
  return [p for p in text.split('\n\n') if p.strip()]


def section(title):
  start = RESULTS.index(f'## {title}')
  end = RESULTS.find('\n## ', start + 1)
  return RESULTS[start:end if end > 0 else len(RESULTS)]


# Each method's test result: the paragraph of its section that begins "On the ten test crops".
TEST_PARAS = {name: next(p for p in paragraphs(section(title)) if p.startswith('On the ten test crops'))
              for name, title in (('arithmetic', 'Speed-ups that change the arithmetic'),
                                  ('smaller', 'Smaller flood-filling networks by distillation'),
                                  ('dense', 'Dense segmentation against FFN'))}
EXACT = section('Speed-ups that keep every trace')
EXACT_TEST = EXACT[EXACT.index('On the ten test crops'):]
DESIGN = section('The reference, the cost and the yardstick')
DESIGN_TEST = DESIGN[DESIGN.index('The crops of the volume had fixed roles'):]
COMPARISON = section('Accuracy against cost on the unseen crops')
DISCUSSION = BODY[BODY.index('# Discussion'):BODY.index('# Materials and methods')]
METHODS = BODY[BODY.index('## Test on unseen crops'):BODY.index('## Reproducibility')]
TEST_ABSTRACT = ABSTRACT[ABSTRACT.index('We then froze'):ABSTRACT.index('Fine-tuning the generalist')]
DATA_ROLE = re.search(r'Crops 41–50[^.]*\.', BODY).group(0)  # the test crops' role in Methods, Data


def sentences(text):
  return re.split(r'(?<=[.;])\s+(?=[A-Z(*])', text)


TEST_DISCUSSION = ' '.join(x for x in sentences(DISCUSSION)
                           if re.search(r'unseen|test crop|test object|ten-crop|ten crops|418|B200', x))


def n(x):
  return f'{x:,}'


def r(x, d):
  return f'{x:.{d}f}'


def minus(s):
  return s.replace('-', '−')


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
  e2e = B['comparisons']['end_to_end']


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
    one_at_a_time = N.e2e['F1']['totals']['break_even_per_crop']
    fastest = N.e2e['F5']['totals']['break_even_per_crop']
    per = J['objects_per_crop'].values()
    b = N.e2e['F5']['break_even']
    self.assertIn(f'cheaper than flood-filling networks tracing one neuron at a time above about '
                  f'{round(one_at_a_time)} neurons per 500³-voxel crop, but against the fastest flood-filling setting it '
                  f"was cheaper on only {N.e2e['F5']['crops_segneuron_cheaper']} of the 10 test crops, with a break-even "
                  f"of about {round(fastest, -1):.0f} neurons per crop ({round(b['min'])} to {round(b['max'])} across "
                  f'crops) against the {min(per)} to {max(per)} that the crops held', ABSTRACT)
    self.assertTrue(b['min'] <= max(per) and b['max'] >= min(per))  # the per-crop break-evens reach the crops' counts
    self.assertEqual(N.e2e['F1']['crops_segneuron_cheaper'], 10)

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
    self.assertIn(f'{min(per)}–{max(per)} large labelled objects per crop, {n(N.objects)} in all', DESIGN_TEST)
    self.assertIn('No method failed on any crop: every run finished, and no run that counts non-finite outputs had one '
                  '(the four float16 B200 runs and SegNeuron count them; the float32 run and the V100 FFN do not)',
                  COMPARISON)
    for crop, runs in J['nonfinite_outputs'].items():
      for m in ('F5', 'S10', 'P2', 'Q2'):
        self.assertEqual(runs[m], 0, (crop, m))

  def test_float32_cost(self):
    f1 = N.b200['F1']
    self.assertIn(f"took {n(round(f1['total_seconds']))} s for the {n(N.objects)} objects "
                  f"({r(f1['seconds_per_object'], 2)} s per object)", EXACT_TEST)

  def test_question_1(self):
    c, text = N.c, TEST_PARAS['arithmetic']
    self.assertIn(minus(f"{c['new_merges']} new merges and {c['new_early_stops']} new early stops against a limit of "
                        f'{N.limit} (2% of objects) each, mean recall {100 * c["mean_recall_change"]:+.2f} and mean '
                        f'precision {100 * c["mean_precision_change"]:+.2f} points'), text)
    self.assertIn(f"avoided {c['avoided_merges']} merges and extended {c['extensions']} traces", text)
    self.assertIn(f"{r(N.b200['F5']['speedup_vs_F1'], 1)}× faster than float32 one object at a time "
                  f"({n(round(N.b200['F5']['total_seconds']))} against {n(round(N.b200['F1']['total_seconds']))} s", text)
    d, a = N.q1['by_class']['dendrites'], N.q1['by_class']['axons']
    self.assertEqual(d['new_merges'], 1)
    self.assertIn(f"among the {d['objects']} classified dendrites, where all five fresh-seed merges had occurred, "
                  'there was one new merge', text)
    self.assertIn(f"among the {a['objects']} classified axons seven traces stopped early "
                  f"({100 * a['new_early_stops'] / a['objects']:.1f}%)", text)
    self.assertEqual(a['new_early_stops'], 7)
    p, agree = J['pooled'], N.q2['V_vs_F5_agreement']
    self.assertEqual(r(p['V']['f1'], 3), r(p['F5']['f1'], 3))
    self.assertIn(f"with the same mean F1 ({r(p['F5']['f1'], 3)} for both), but the two forms differed by more than 0.1 "
                  f"F1 on {agree['objects_abs_difference_gt_0.1']} of the {n(N.objects)} objects", text)

  def test_smaller(self):
    s, text = N.smaller, TEST_PARAS['smaller']
    self.assertIn(f"lower by {r(-s['student']['mean'], 4)} for the factorised student (95% interval "
                  f"{r(-s['student']['t_interval'][1], 4)} to {r(-s['student']['t_interval'][0], 4)}), "
                  f"by {r(-s['P8']['mean'], 4)} for P8 ({r(-s['P8']['t_interval'][1], 4)} to "
                  f"{r(-s['P8']['t_interval'][0], 4)}) and by {r(-s['P4']['mean'], 4)} for P4 "
                  f"({r(-s['P4']['t_interval'][1], 4)} to {r(-s['P4']['t_interval'][0], 4)})", text)
    p = J['q3_p8_minus_student']
    self.assertFalse(p['difference_claimed'])
    self.assertIn(minus(f"(+{r(p['mean'], 4)}, {r(p['t_interval'][0], 4)} to +{r(p['t_interval'][1], 4)})"), text)
    vs_f5 = ', '.join(f"{r(N.b200['F5']['total_seconds'] / N.b200[m]['total_seconds'], 1)}×" for m in ('S10', 'P2'))
    vs_f1 = ', '.join(f"{r(N.b200[m]['speedup_vs_F1'], 1)}×" for m in ('S10', 'P2'))
    self.assertIn(f"They were {vs_f5} and {r(N.b200['F5']['total_seconds'] / N.b200['Q2']['total_seconds'], 1)}× faster "
                  f"than `fp16_head32-s16` FFN ({vs_f1} and {r(N.b200['Q2']['speedup_vs_F1'], 1)}× faster than float32 "
                  'FFN)', text)
    ra = J['runaways']
    self.assertIn(f"{ra['S10_vs_F1']['runaway_traces']}, {ra['P2_vs_F1']['runaway_traces']} and "
                  f"{ra['Q2_vs_F1']['runaway_traces']} traces, against {ra['F5_vs_F1']['runaway_traces']} for "
                  '`fp16_head32-s16` FFN', text)
    self.assertIn('for the student and P8 the intervals end close to the margin, and on some crops the difference was '
                  'smaller than 0.02', text)
    self.assertTrue(-0.02 < s['P8']['t_interval'][1] < 0)
    self.assertTrue(s['student']['t_interval'][1] > -0.022)
    for m in ('student', 'P8'):
      self.assertTrue(any(-0.02 < v < 0 for v in s[m]['per_crop'].values()), m)

  def test_segneuron(self):
    f5, v, text = N.q2['F5_minus_segneuron'], N.q2['V_minus_segneuron'], TEST_PARAS['dense']
    for c in (f5, v):  # "with either FFN implementation": the same rounded values
      self.assertIn(f"+{r(c['mean'], 3)} F1 (95% interval {r(c['t_interval'][0], 3)} to {r(c['t_interval'][1], 3)})", text)
    self.assertIn(f"higher on {n(f5['objects']['ffn_higher'])} (B200) and {n(v['objects']['ffn_higher'])} (V100) of "
                  f"the {n(N.objects)} objects", text)
    o = f5['objects']
    self.assertIn(f"Without the {o['segneuron_seeds_in_background']} objects whose seed SegNeuron placed in predicted "
                  f"background, mean F1 was {r(o['ffn_mean_f1_where_segneuron_seed_found'], 3)} against "
                  f"{r(o['segneuron_mean_f1_where_seed_found'], 3)}", text)
    p, c = J['pooled'], J['counts']
    self.assertIn(f"mostly in recall ({r(p['segneuron']['recall'], 3)} against {r(p['F5']['recall'], 3)}); it had fewer "
                  f"merged voxels than FFN ({c['segneuron']['merged_voxels'] / 1e6:.1f} M against "
                  f"{c['F5']['merged_voxels'] / 1e6:.1f} M), though more objects with a merge "
                  f"({c['segneuron']['objects_with_merge']} against {c['F5']['objects_with_merge']})", text)

  def test_comparison(self):
    """The section comparing all methods: orders, B200 break-evens and the V100 view."""
    p, t = J['pooled'], COMPARISON
    self.assertIn(f"FFN (float32 and `fp16_head32-s16`, {r(p['F5']['f1'], 3)}), P8 ({r(p['P2']['f1'], 3)}), the "
                  f"factorised student ({r(p['S10']['f1'], 3)}), P4 ({r(p['Q2']['f1'], 3)}) and fine-tuned SegNeuron "
                  f"({r(p['segneuron']['f1'], 3)})", t)
    self.assertEqual(r(p['F1']['f1'], 3), r(p['F5']['f1'], 3))
    sn = N.e2e['F5']['totals']['segneuron_seconds']
    self.assertIn(f"P4 ({n(round(N.b200['Q2']['total_seconds']))} s for the ten crops), P8 "
                  f"({n(round(N.b200['P2']['total_seconds']))} s), the student ({n(round(N.b200['S10']['total_seconds']))} s), "
                  f"SegNeuron ({n(round(sn))} s), `fp16_head32-s16` FFN ({n(round(N.b200['F5']['total_seconds']))} s) "
                  f"and float32 FFN ({n(round(N.b200['F1']['total_seconds']))} s)", t)
    e = N.e2e
    self.assertEqual(e['F1']['crops_segneuron_cheaper'], 10)
    self.assertIn(f"on every test crop (break-even {round(e['F1']['totals']['break_even_per_crop'])} objects per crop "
                  'from the totals)', t)
    b = e['F5']['break_even']
    self.assertIn(f"cheaper than `fp16_head32-s16` FFN on {e['F5']['crops_segneuron_cheaper']} of 10 crops (break-even "
                  f"{round(e['F5']['totals']['break_even_per_crop'])}; {round(b['min'])}–{round(b['max'])} across crops)", t)
    self.assertEqual(e['S10']['crops_segneuron_cheaper'], e['P2']['crops_segneuron_cheaper'])
    self.assertIn(f"cheaper than the student and P8 on {e['S10']['crops_segneuron_cheaper']} of 10 "
                  f"({round(e['S10']['totals']['break_even_per_crop'])} and {round(e['P2']['totals']['break_even_per_crop'])})",
                  t)
    self.assertEqual(e['Q2']['crops_segneuron_cheaper'], 0)
    self.assertIn(f"dearer than P4 on every crop ({n(round(e['Q2']['totals']['break_even_per_crop']))})", t)
    per = J['objects_per_crop'].values()
    self.assertIn(f'The test crops held {min(per)}–{max(per)} large labelled objects', t)
    v = N.v100['totals']
    self.assertIn(f"above {round(v['break_even_end_to_end_per_crop'])} objects per crop "
                  f"({n(round(v['segneuron_total_seconds']))} against {n(round(v['ffn_wall_seconds']))} s for the ten crops)",
                  t)

  def test_b200_disclosures(self):
    host = {k: x['job'].split()[1] for k, x in json.loads(
        (ROOT / 'docs' / 'unseen_b200_evidence.json').read_text())['devices'].items()}
    nodes = sorted(set(host.values()))
    post = {h: [B['segneuron_b200'][k]['post_seconds'] for k in host if host[k] == h] for h in nodes}
    first, second = sorted(nodes, key=lambda h: min(post[h]))
    loop = [x['inference_seconds'] for x in B['segneuron_b200'].values()]
    self.assertIn(f"{min(post[first]):.0f}–{max(post[first]):.0f} s per crop on one node and "
                  f"{min(post[second]):.0f}–{max(post[second]):.0f} s on the other, against "
                  f"{min(loop):.0f}–{max(loop):.0f} s of GPU time", COMPARISON)
    cheap = set(map(str, N.e2e['F5']['cheaper_on']))
    on = {h: sum(1 for k in host if host[k] == h and k in cheap) for h in nodes}
    self.assertEqual((on[first], on[second]), (5, 2))
    self.assertIn('all five crops on the first node, and two of five on the second, were cheaper', COMPARISON)
    a = B['agreement_pooled']
    self.assertIn(f"(mean F1 difference {a['mean_f1_difference']:.5f}; {a['objects_abs_difference_gt_0.1']} of "
                  f"{n(a['objects'])} objects differed by more than 0.1)", COMPARISON)
    worst = max(abs(g['mean_f1_b200'] - g['mean_f1_v100']) for g in B['agreement'].values())
    self.assertIn(f"agreed with the V100 run's to within {r(worst + 0.00005, 4)}", COMPARISON)
    self.assertLess(worst, 0.0004)

  def test_table_1(self):
    rows = [('FFN, float32, one at a time', 'B200', 'F1'), ('FFN, `fp16_head32-s16`', 'B200', 'F5'),
            ('factorised student', 'B200', 'S10'), ('P8', 'B200', 'P2'), ('P4', 'B200', 'Q2'),
            ('fine-tuned SegNeuron', 'B200', 'segneuron'),
            ('FFN, `fp16_head32`, fast tracer', 'V100', 'V'), ('fine-tuned SegNeuron', 'V100', 'segneuron')]
    t = N.v100['totals']
    times = {('B200', m): N.b200[m]['total_seconds'] for m in N.b200}
    times[('B200', 'segneuron')] = N.e2e['F5']['totals']['segneuron_seconds']
    times[('V100', 'V')] = t['ffn_wall_seconds']
    times[('V100', 'segneuron')] = t['segneuron_total_seconds']
    delta = {'S10': N.smaller['student'], 'P2': N.smaller['P8'], 'Q2': N.smaller['P4']}
    for label, gpu, m in rows:
      p, c = J['pooled'][m], J['counts'][m]
      if m == 'F5':
        d = 'reference'
      elif gpu == 'B200' and m in delta:
        x = delta[m]
        d = minus(f"{r(x['mean'], 3)} ({r(x['t_interval'][0], 3)}, {r(x['t_interval'][1], 3)})")
      elif gpu == 'B200' and m == 'segneuron':
        x = N.q2['F5_minus_segneuron']
        d = minus(f"{r(-x['mean'], 3)} ({r(-x['t_interval'][1], 3)}, {r(-x['t_interval'][0], 3)})")
      else:
        d = '–'
      key = f'{m}_vs_F1'
      run = str(J['runaways'][key]['runaway_traces']) if key in J['runaways'] and gpu == 'B200' else '–'
      tm = times[(gpu, m)]
      self.assertIn(f"| {label} | {gpu} | {r(p['f1'], 3)} | {r(p['recall'], 3)} | {r(p['precision'], 3)} | {d} | "
                    f"{c['objects_with_merge']} | {c['merged_voxels'] / 1e6:.1f} M | {run} | {n(round(tm))} s | "
                    f"{r(tm / N.objects, 2)} |", COMPARISON)
    self.assertIn(f'**Table 1 | Ten unseen crops (41–50), {n(N.objects)} objects.**', COMPARISON)

  def test_bootstrap_gives_the_same_claims(self):
    self.assertIn('was computed as a check and gave the same claim for every comparison', METHODS)
    for c in (N.q2['F5_minus_segneuron'], N.q2['V_minus_segneuron']):
      self.assertGreater(c['bootstrap_interval'][0], 0)
    for c in N.smaller.values():
      self.assertTrue(c['bootstrap_interval'][1] < 0 and c['bootstrap_interval'][0] <= -0.02)
    b = J['q3_p8_minus_student']['bootstrap_interval']
    self.assertTrue(b[0] < 0 < b[1])


class GpuTrainingWordingTest(unittest.TestCase):
  """The agreement of validation losses across GPUs, reworded in Draft 5 (the spread is 0.000302, not within
  0.0003), as docs/gpu_training_evidence.json gives it; tests/test_gpu_training_evidence.py checks the rest."""

  def test_validation_loss_range(self):
    g = json.loads((ROOT / 'docs' / 'gpu_training_evidence.json').read_text())
    losses = [p['valid_loss_200'] for p in g['probes'].values()]
    lo, hi = f'{min(losses):.4f}', f'{max(losses):.4f}'
    supplement = (MS / 'supplement.md').read_text()
    self.assertIn(f'validation loss after 200 steps ranged only from {lo} to {hi} across V100, A100, H100 and B200',
                  BODY)
    self.assertIn(f'Validation loss after 200 steps ranged only from {lo} to {hi} across all four GPU types',
                  supplement)
    self.assertNotIn('within 0.0003', TEXT + supplement)


class ScopeWordingTest(unittest.TestCase):
  """Claims reworded after review: batching and the fast tracer were identical in the settings tested, not shown
  exact for every input; the cost comparison says which fixed costs each method's time includes, as
  docs/unseen_b200_results.md does; and two supplementary overstatements stay removed."""

  def test_identity_is_scoped_to_the_settings_tested(self):
    for overclaim in ('exact again', 'accuracy argument', 'without changing a single trace', 'exact by construction',
                      'settled exactly', "keep the published method's output exactly",
                      'batching produce bitwise-identical traces', 'most of the gap to perfect segmentation'):
      self.assertNotIn(overclaim, TEXT)
    self.assertIn('This shows identity for the widths, formats and objects tested (Methods), not for every input.',
                  EXACT)
    self.assertIn('identity was not shown for every width, format or volume: for FFN, batches of 8 were checked only '
                  'in `fp16_head32` and batches of 32 only in plain float16 with the scaled stream', DISCUSSION)
    methods = BODY[BODY.index('## PyTorch port and batched tracing'):BODY.index('## Variants, number formats')]
    self.assertIn('for FFN at widths 1 and 8 (`fp16_head32`) and 1 and 32 (float16, scaled stream)', methods)
    self.assertIn('batching produced bitwise-identical traces in every setting tested', BODY)

  def test_fixed_costs_are_disclosed(self):
    source = ' '.join((ROOT / 'docs' / 'unseen_b200_results.md').read_text().split())
    self.assertIn("SegNeuron's end-to-end time includes process start-up, model loading and reading and writing its "
                  "affinities", source)
    self.assertIn("FFN's crop time leaves out loading the network and follows a warm-up", source)
    self.assertIn("SegNeuron's crop time includes process start-up, model loading and reading and writing its "
                  "affinities, whereas each FFN crop time leaves out loading the network and follows a warm-up",
                  COMPARISON)
    legend = COMPARISON[COMPARISON.index('**Table 1 |'):]
    self.assertIn('including start-up, model loading and affinity input and output', legend)
    self.assertIn("FFN's crop time leaves out loading the network and follows a warm-up", legend)
    self.assertIn('measured after the network was loaded and a warm-up had run; SegNeuron\'s is measured as on the '
                  'V100 and so includes process start-up, model loading and reading and writing its affinities',
                  METHODS)

  def test_supplement_overstatements_removed(self):
    supplement = (MS / 'supplement.md').read_text()
    self.assertNotIn('visited different boxes from then on', supplement)
    self.assertIn('It did not establish whether the diverging traces later rejoined', supplement)
    self.assertNotIn('not the post-processing, limited accuracy', supplement)
    h = json.loads((ROOT / 'docs' / 'vitseg_evidence.json').read_text())['hybrids']
    mask, aff = (h['label_neuron_mask__predicted_affinities']['mean_f1'],
                 h['predicted_neuron_mask__label_affinities']['mean_f1'])
    self.assertIn(f'replacing only the foreground or only the affinities by the labels reached {mask:.3f} and '
                  f'{aff:.3f}, so neither substitution alone came close to the oracle', supplement)

class DiscussionTest(unittest.TestCase):

  def test_numbers(self):
    s, d = N.smaller, DISCUSSION
    self.assertIn(f"pooled over all large objects with one new merge among {N.q1['by_class']['dendrites']['objects']} "
                  f"classified dendrites", d)
    self.assertIn(f"{r(N.b200['F5']['speedup_vs_F1'], 1)} times faster tracing than float32", d)
    self.assertIn(f"within {r(-s['student']['mean'], 3)} and {r(-s['P8']['mean'], 3)} on the test crops, where P4 was "
                  f"{r(-s['P4']['mean'], 3)} below", d)
    f5 = N.b200['F5']['total_seconds']
    speed = [f5 / N.b200[m]['total_seconds'] for m in ('S10', 'P2', 'Q2')]
    self.assertIn(f'{r(min(speed), 1)} to {r(max(speed), 1)} times faster than FFN itself', d)
    self.assertIn(f"was {r(N.q2['F5_minus_segneuron']['mean'], 3)} mean F1 below FFN on unseen crops", d)
    e = N.e2e
    self.assertIn(f"above {round(e['F1']['totals']['break_even_per_crop'])} objects per crop, cheaper than the fastest FFN "
                  f"setting above about {round(e['F5']['totals']['break_even_per_crop'], -1):.0f}, and dearer than the "
                  f"smallest distilled tracer, P4, below about {n(round(e['Q2']['totals']['break_even_per_crop']))}", d)
    o = J['ffn_v100_overlap'].values()
    share = sum(x['voxels_in_two_or_more'] for x in o) / sum(x['voxels_in_any_mask'] for x in o)
    self.assertIn(f'{100 * share:.1f}% on the test crops', d)
    self.assertIn(f"{J['object_ids_in_more_than_one_crop']} object IDs were scored in more than one crop", d)

  def test_nodes_and_non_finite(self):
    for crop, runs in J['nonfinite_outputs'].items():
      for m in ('F5', 'S10', 'P2', 'Q2'):
        self.assertEqual(runs[m], 0, (crop, m))
    self.assertNotIn('segneuron', J['failures'])
    hosts_b200 = len(J['devices']['b200_hosts'])
    hosts_v100 = len(set(EVIDENCE['v100_devices']['jobs'].values()))
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    self.assertIn(f"({words[hosts_b200]} B200 and {words[hosts_v100]} V100 nodes in the test's own jobs)", DISCUSSION)

  def test_conclusion(self):
    self.assertIn('On ten crops that no part of the work had read, the accelerated FFN passed a rule fixed for them that '
                  'allows at most 2% of neurons to gain a new merge and 2% to stop early, pooled over all neurons, '
                  'though the rule was set after an earlier failure', DISCUSSION)
    self.assertTrue(N.q1['passes'])

  def test_data_role(self):
    self.assertIn('the last ten positions of the order, were the test set for every method; their role was fixed and the '
                  'analysis written before they were fetched.', DATA_ROLE)
    self.assertEqual(list(map(int, J['objects_per_crop'])), list(range(41, 51)))

  def test_no_claim_that_the_test_is_still_to_come(self):
    for stale in ('remains to be made', 'has not yet been made', 'still have to be compared',
                  'reserved as the test set', 'was not timed on that GPU', 'has not been timed on the same GPU'):
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
  sn = B['comparisons']['end_to_end']['F5']['totals']['segneuron_seconds']
  out = [b['F5']['total_seconds'] / b[m]['total_seconds'] for m in b]
  out += [t['ffn_wall_seconds'] / t['objects'], t['segneuron_total_seconds'] / t['objects'], sn / t['objects'],
          100 * J['q1_fast_mode']['by_class']['axons']['new_early_stops'] /
          J['q1_fast_mode']['by_class']['axons']['objects']]
  o = J['ffn_v100_overlap'].values()
  out.append(100 * sum(x['voxels_in_two_or_more'] for x in o) / sum(x['voxels_in_any_mask'] for x in o))
  out += [len(J['crop_mean_f1']['F1'])]  # ten crops
  out.append(round(B['comparisons']['end_to_end']['F5']['totals']['break_even_per_crop'], -1))  # 'about 390'
  out.append(max(abs(g['mean_f1_b200'] - g['mean_f1_v100']) for g in B['agreement'].values()) + 0.00005)
  return out


VALUES = leaves(J) + leaves(B) + derived()
# Constants of the designs and protocol (docs/unseen_design.md, docs/unseen_b200_segneuron_design.md), crop and
# date labels, and node counts; nothing else in the test passages may be a number not in the evidence.
CONSTANTS = {
    '0.02', '−0.02', '2', '1', '−1', '0.1', '1,000', '95', '9', '10,000', '32', '16', '0.6', '8',  # rules, settings
    '41', '50', '11', '12', '300', '500', '2026', '4', '256', '13', '40', '0', '28',  # crops, seed sets, sizes, dates
    '12.8',  # the CUDA build of PyTorch for the B200 (a version label)
    '0.07',  # the crop-12 gap the dense test paragraph quotes for comparison (docs/headtohead_results.md)
    '23',  # one dendrite in 23 (5 of 114) on the fresh seeds, quoted beside the test (docs/fresh_results.md)
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
    passages = [('abstract', TEST_ABSTRACT), ('exact', EXACT_TEST), ('design', DESIGN_TEST),
                ('comparison', COMPARISON), ('discussion', TEST_DISCUSSION), ('methods', METHODS), ('data', DATA_ROLE)]
    passages += list(TEST_PARAS.items())
    for name, text in passages:
      text = re.sub(r'`[^`]*`', '', text)  # format names such as fp16_head32-s16
      for token in number.findall(text):
        token = token.rstrip(',')
        self.assertTrue(explained(token), f'{name}: {token} is not in the evidence')


if __name__ == '__main__':
  unittest.main()
