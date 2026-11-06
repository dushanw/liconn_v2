"""Factorised FFN, part 2: the PyTorch tracer and the distillation pieces (CPU, float32).

See docs/distil_design.md. The tracer must follow fasttrace.FastTracer exactly; the torch teacher
must reproduce the notebook's reference trace.
"""

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import torch

from liconn_v2 import data
from liconn_v2 import distil
from liconn_v2 import factorised as fz
from liconn_v2 import fasttrace
from liconn_v2 import reference


def _teacher():
  from clu import checkpoint  # pylint: disable=g-import-not-at-top
  params = checkpoint.Checkpoint('').load_state(state=None, checkpoint=data.checkpoint_path())['params']
  return fz.teacher_from_params(params)


class OptionsTest(unittest.TestCase):

  def test_thresholds_equal_ffn_canvas_options(self):
    from ffn.inference import inference_pb2  # pylint: disable=g-import-not-at-top
    opts = fasttrace.canvas_options(inference_pb2.InferenceOptions(
        init_activation=reference.INIT_ACTIVATION, pad_value=reference.PAD_VALUE,
        move_threshold=reference.MOVE_THRESHOLD, segment_threshold=reference.SEGMENT_THRESHOLD))
    for name in fasttrace.LOGIT_OPTIONS:
      self.assertEqual(getattr(distil.OPTIONS, name), getattr(opts, name), name)


class UpdateTest(unittest.TestCase):
  """The torch canvas update, face maxima and positions equal fasttrace's JAX functions."""

  def setUp(self):
    self.jax_fns = fasttrace._device_fns(reference.FOV_SIZE, distil.OPTIONS.pad_value,  # pylint: disable=protected-access
                                         distil.OPTIONS.move_threshold)

  def _case(self, rng, nan_logits=False, all_low=False):
    shape = (45, 50, 41)
    canvas = (rng.normal(size=shape) * 3).astype(np.float32)
    canvas[rng.random(shape) < 0.4] = np.nan
    canvas[rng.random(shape) < 0.05] = 0.0  # the seed-consistency rule's boundary (old < 0)
    logits = np.round(rng.normal(size=reference.FOV_SIZE) * 4) / 2  # many ties
    if all_low:
      logits = -np.abs(logits) - 1
    if nan_logits:
      logits[rng.random(logits.shape) < 0.01] = np.nan
    corner = np.array([int(rng.integers(0, s - 32)) for s in shape])
    return canvas, logits.astype(np.float32), corner

  def _compare(self, canvas, logits, corner):
    _, update, _, _, _ = self.jax_fns
    j_canvas, j_scores, j_pos = update(np.array(canvas), logits[None, ..., None], corner.astype(np.int32))
    t_canvas = torch.from_numpy(canvas.copy())
    moves = distil.update(t_canvas, torch.from_numpy(logits), corner).numpy()
    np.testing.assert_array_equal(t_canvas.numpy(), np.asarray(j_canvas))  # NaNs equal too
    np.testing.assert_array_equal(moves[:, 0], np.asarray(j_scores))
    np.testing.assert_array_equal(moves[:, 1:].astype(np.int64), np.asarray(j_pos))
    m = distil.assemble_mask(torch.from_numpy(canvas), corner).numpy()
    assemble = self.jax_fns[0]
    np.testing.assert_array_equal(
        m[0, 0], np.asarray(assemble(canvas, np.zeros(canvas.shape, np.float32), corner.astype(np.int32)))[0, ..., 1])

  def test_update_equals_fasttrace(self):
    rng = np.random.default_rng(0)
    for _ in range(20):
      self._compare(*self._case(rng))

  def test_update_equals_fasttrace_with_nan_logits_and_no_move(self):
    rng = np.random.default_rng(1)
    for _ in range(5):
      self._compare(*self._case(rng, nan_logits=True))
      self._compare(*self._case(rng, all_low=True))


class TeacherTraceTest(unittest.TestCase):

  def test_torch_teacher_reproduces_the_reference_trace(self):
    image = reference.normalize(data.load_raw_xyz())
    tracer = distil.TorchTracer(distil.TeacherNet(_teacher(), image, 'cpu'), image.shape, 'cpu')
    trace = tracer.trace(reference.SEED_XYZ[::-1])
    got, expected = reference.fingerprint(trace), reference.load_expected()
    for key in ('iterations', 'history_sha256', 'history_first', 'predicted_voxels',
                'predicted_mask_sha256'):
      self.assertEqual(got[key], expected[key], key)
    # The canvas values differ from the JAX run by float rounding, not by the segment.
    self.assertLessEqual(abs(got['seed_component_voxels'] - expected['seed_component_voxels']), 50)


class TeacherFileTest(unittest.TestCase):

  def test_saved_teacher_equals_checkpoint_port(self):
    t = _teacher()
    with tempfile.TemporaryDirectory() as d:
      distil.save_teacher(t, Path(d) / 'teacher.pt')
      u = distil.load_teacher(Path(d) / 'teacher.pt')
    for (k, a), (k2, b) in zip(t.state_dict().items(), u.state_dict().items()):
      self.assertEqual(k, k2)
      self.assertTrue(torch.equal(a, b), k)


class SampleTest(unittest.TestCase):

  def _record(self, crop, trace, n):
    rec = distil.SampleRecorder(crop, trace)
    z = torch.zeros(reference.FOV_SIZE)
    for s in range(n):
      rec(s, np.array([s, 0, 1]), z[None, None] + s, z - s)
    return rec.arrays()

  def test_early_steps_always_kept_and_rule_is_deterministic(self):
    a, b = self._record(3, 7, 4000), self._record(3, 7, 4000)
    for k in distil.SAMPLE_KEYS:
      np.testing.assert_array_equal(a[k], b[k])
    self.assertEqual(a['steps'][:4].tolist(), [0, 1, 2, 3])
    frac = (len(a['steps']) - 4) / (4000 - 4)
    self.assertTrue(0.22 < frac < 0.28, frac)
    c = self._record(3, 8, 4000)
    self.assertNotEqual(a['steps'].tolist(), c['steps'].tolist())

  def test_teacher_sample_stream_is_unchanged(self):
    """The round-0 teacher samples were drawn from default_rng([crop, trace]), one draw per step."""
    rng = np.random.default_rng([3, 7])
    expected = [s for s in range(4000) if (rng.random() < distil.KEEP_FRACTION) or s < distil.EARLY_STEPS]
    self.assertEqual(self._record(3, 7, 4000)['steps'].tolist(), expected)

  def test_on_policy_recorder_stores_the_teachers_answer_on_the_students_input(self):
    calls = []

    def teacher(corner, mask):
      calls.append(int(corner[0]))
      return torch.full(reference.FOV_SIZE, 10.0) + mask[0, 0]  # depends on the student's mask

    rec = distil.OnPolicyRecorder(3, 7, teacher)
    z = torch.zeros(reference.FOV_SIZE)
    for s in range(400):
      rec(s, np.array([s, 0, 1]), z[None, None] + s, z - s)  # student output: -s
    a = rec.arrays()
    self.assertEqual(calls, a['steps'].tolist())  # the teacher runs on kept steps only
    self.assertEqual(a['steps'][:4].tolist(), [0, 1, 2, 3])
    for i, s in enumerate(a['steps']):
      self.assertTrue(np.all(a['masks'][i] == s) and np.all(a['targets'][i] == 10 + s))
    self.assertNotEqual(a['steps'].tolist(), self._record(3, 7, 400)['steps'].tolist())  # own stream
    self.assertEqual(rec.moves.steps, len(a['steps']))

  def test_move_counts(self):
    t = np.float32(distil.OPTIONS.move_threshold)
    teacher = torch.full(reference.FOV_SIZE, -5.0)
    student = teacher.clone()
    teacher[24, 16, 16] = 5.0  # z+: the teacher moves; the student falls 0.25 short (near miss)
    student[24, 16, 16] = float(t) - 0.25
    teacher[16, 8, 16] = student[16, 8, 16] = 5.0  # y-: both move
    teacher[16, 16, 24] = 5.0  # x+: the teacher moves, the student is far below
    student[16, 16, 8] = 5.0  # x-: the student moves alone
    m = distil.MoveCounts()
    m.count(student, teacher)
    s = m.summary()
    self.assertEqual((s['steps'], s['teacher_moves'], s['student_moves'], s['both_move']), (1, 3, 2, 1))
    self.assertEqual((s['missed_moves'], s['extra_moves'], s['missed_within_near_miss']), (2, 1, 1))
    self.assertAlmostEqual(s['missed_gap_median'], (0.25 + float(t) + 5.0) / 2, places=5)
    total = distil.MoveCounts()
    total.add(m)
    total.add(m)
    self.assertEqual(total.summary()['missed_moves'], 4)

  def test_written_equals_the_canvas_update(self):
    g = torch.Generator().manual_seed(0)
    for active in (True, False):
      canvas = torch.randn((40, 40, 40), generator=g) * 3
      canvas[torch.rand((40, 40, 40), generator=g) < 0.3] = float('nan')
      canvas[5, 5, 5] = 0.0
      corner = np.array([2, 3, 4])
      mask = distil.assemble_mask(canvas, corner)
      logits = torch.randn(reference.FOV_SIZE, generator=g) * 3
      if not active:
        logits = logits.clamp(max=float(distil.OPTIONS.move_threshold) - 1)
      got = distil.written(logits[None], mask[0])[0]
      distil.update(canvas, logits, corner)
      torch.testing.assert_close(got, canvas[distil.box(corner)], rtol=0, atol=0, equal_nan=False)

  def test_move_gaps_apply_the_canvas_rule_and_bin_by_step(self):
    t = float(distil.OPTIONS.move_threshold)
    mask = torch.zeros((2, *reference.FOV_SIZE))
    teacher = torch.full((2, *reference.FOV_SIZE), -5.0)
    student = teacher.clone()
    teacher[:, 24, 16, 16] = 5.0  # z+: the teacher moves
    student[:, 24, 16, 16] = t - 0.25  # the student falls 0.25 short
    mask[1, 16, 8, 16] = -1.0  # y- in sample 1: the old canvas is negative there
    teacher[1, 16, 8, 16] = 4.0  # raw teacher output would move, but FFN's rule keeps -1
    m = distil.MoveGaps()
    m.add(student, teacher, mask, steps=[0, 7])
    s = m.summary()
    a0, a7 = s['0']['all'], s['4-49']['all']
    self.assertEqual((a0['teacher_moves'], a0['missed_moves']), (1, 1))
    self.assertEqual((a7['teacher_moves'], a7['missed_moves']), (1, 1))  # y- not counted
    self.assertAlmostEqual(a0['teacher_minus_student_quantiles']['0.5'], 5.0 - (t - 0.25), places=5)
    self.assertAlmostEqual(a7['missed_below_threshold_quantiles']['0.5'], 0.25, places=5)
    self.assertEqual(s['1-3']['all']['samples'], 0)
    self.assertIsNone(s['50+']['all']['teacher_minus_student_quantiles'])
    # near the threshold: mean of 5.0 and t - 0.25 is more than 1 above t, so not counted
    self.assertEqual(a0['near_threshold_faces'], 0)
    self.assertEqual(s['0']['frontier']['teacher_moves'], 1)  # the mask is 0 on z+: a frontier face

  def test_move_gaps_frontier_and_near_threshold_faces(self):
    t = float(distil.OPTIONS.move_threshold)
    mask = torch.zeros((1, *reference.FOV_SIZE))
    mask[0, 8, 8:25, 8:25] = 3.0  # z- face already traced: not a frontier face
    teacher = torch.full((1, *reference.FOV_SIZE), -5.0)
    student = teacher.clone()
    teacher[0, 8, 16, 16], student[0, 8, 16, 16] = 6.0, 6.0  # z-: both move (backward face)
    teacher[0, 24, 16, 16], student[0, 24, 16, 16] = t + 0.3, t - 0.3  # z+: near the threshold
    m = distil.MoveGaps()
    m.add(student, teacher, mask, steps=[2])
    a, f = m.summary()['1-3']['all'], m.summary()['1-3']['frontier']
    self.assertEqual((a['teacher_moves'], f['teacher_moves']), (2, 1))
    self.assertEqual((f['missed_moves'], f['near_threshold_faces']), (1, 1))
    self.assertAlmostEqual(f['near_threshold_teacher_minus_student_quantiles']['0.5'], 0.6, places=5)

  def test_sample_content_and_round_trip(self):
    a = self._record(0, 1, 50)
    i = 5
    s = int(a['steps'][i])
    self.assertEqual(a['centers'][i].tolist(), [s + 16, 16, 17])
    self.assertTrue(np.all(a['masks'][i] == s) and np.all(a['targets'][i] == -s))
    self.assertTrue(np.all(a['traces'] == 1))
    with tempfile.TemporaryDirectory() as d:
      distil.save_samples(Path(d) / 's', a)
      b = distil.load_samples(Path(d) / 's')
    for k in distil.SAMPLE_KEYS:
      np.testing.assert_array_equal(a[k], b[k])
      self.assertEqual(a[k].dtype, b[k].dtype)


class TrainingPiecesTest(unittest.TestCase):

  def test_gathered_features_equal_box_features(self):
    g = torch.Generator().manual_seed(0)
    feats = [torch.randn(4, 40, 45, 50, generator=g) for _ in range(2)]
    centers = np.array([[16, 16, 16], [23, 28, 33], [20, 20, 30]])
    crops = [0, 1, 1]
    got = distil.gather_features(feats, crops, centers)
    for j, (k, c) in enumerate(zip(crops, centers)):
      self.assertTrue(torch.equal(got[j], fz.box_features(feats[k], c)[0]))

  def test_loss_is_soft_bce_and_stationary_at_the_teacher(self):
    t = torch.randn(2, 5, 5, 5) * 3
    s = t.clone().requires_grad_(True)
    loss = distil.distil_loss(s, t)
    p = torch.sigmoid(t)
    manual = -(p * torch.log(torch.sigmoid(t)) + (1 - p) * torch.log(1 - torch.sigmoid(t))).mean()
    self.assertAlmostEqual(float(loss), float(manual), places=5)
    loss.backward()
    self.assertLess(float(s.grad.abs().max()), 1e-7)

  def test_learning_rate_schedule(self):
    self.assertAlmostEqual(distil.learning_rate(distil.WARMUP - 1, 10_000), distil.PEAK_LR)
    self.assertLess(distil.learning_rate(0, 10_000), distil.PEAK_LR / 100)
    self.assertAlmostEqual(distil.learning_rate(10_000, 10_000), 0.0)

  def test_agreement_counts(self):
    t = torch.full((2, *reference.FOV_SIZE), -5.0)
    s = t.clone()
    t[0, 16 + 8, 16, 16] = 5.0  # the teacher moves through face z+ only, at step 0
    s[0, 16 - 8, 16, 16] = 5.0  # the student through z- instead
    s[1, 0, 0, 0] = 5.0  # a shell voxel at a later step
    a = distil.agreement(s, t, steps=[0, 3])
    self.assertEqual(a['first_steps'], 1)
    self.assertEqual((a['teacher_moves'], a['both_move'], a['student_only_moves']), (1, 0, 1))
    s[0, 16 + 8, 16, 18] = 5.0  # now the student also moves through z+, 2 voxels off: near
    a = distil.agreement(s, t, steps=[0, 3])
    self.assertEqual((a['both_move'], a['both_move_near'], a['student_only_moves']), (1, 1, 1))
    s[0, 16 + 8, 16, 18] = -5.0
    s[0, 16 + 8, 16, 19] = 5.0  # 3 voxels off: a move, but not the same one
    a = distil.agreement(s, t, steps=[0, 3])
    self.assertEqual((a['both_move'], a['both_move_near']), (1, 0))
    s[0, 16 + 8, 16, 19] = -5.0
    a = distil.agreement(s, t, steps=[0, 3])
    self.assertEqual(a['interior_voxels'] - a['interior_same'], 2)
    self.assertEqual(a['shell_voxels'] - a['shell_same'], 1)
    self.assertEqual(a['interior_voxels'] + a['shell_voxels'], 2 * 33 ** 3)
    self.assertEqual(a['interior_voxels'], 2 * 25 ** 3)
    self.assertGreater(a['interior_loss_sum'], 0)

  def test_agreement_without_first_steps(self):
    """A validation batch with no step-0 sample (most of them) counts no first-step moves."""
    t = torch.full((2, *reference.FOV_SIZE), -5.0)
    t[0, 16 + 8, 16, 16] = 5.0
    a = distil.agreement(t.clone(), t, steps=[2, 3])
    self.assertEqual((a['first_steps'], a['teacher_moves'], a['both_move'], a['both_move_near'],
                      a['student_only_moves']), (0, 0, 0, 0, 0))
    self.assertEqual(a['interior_same'] + a['shell_same'], 2 * 33 ** 3)

  def test_a_few_steps_lower_the_loss(self):
    """The loss above its floor (the teacher's own entropy) falls by more than half."""
    torch.manual_seed(0)
    target_net = fz.Decoder((1, 1, 1, 1))
    torch.nn.init.normal_(target_net.mask_b.weight, std=0.05)
    student = fz.Decoder((1, 2, 4, 1))
    feats, masks = torch.randn(4, 32, 17, 17, 17), torch.randn(4, 1, 17, 17, 17) * 3
    with torch.no_grad():
      targets = target_net(feats, masks)[:, 0]
      floor = float(distil.distil_loss(targets, targets))
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.0)
    excess = []
    for _ in range(30):
      loss = distil.distil_loss(student(feats, masks)[:, 0], targets)
      opt.zero_grad()
      loss.backward()
      opt.step()
      excess.append(float(loss) - floor)
    self.assertLess(excess[-1], 0.5 * excess[0])


class ConstantNet:
  """A stand-in network that predicts one logit everywhere."""

  def __init__(self, value):
    self.value = value

  def __call__(self, corner, mask):
    return torch.full(reference.FOV_SIZE, self.value)


class TracerLoopTest(unittest.TestCase):

  def test_trace_stops_when_the_seed_falls_below_the_move_threshold(self):
    trace = distil.TorchTracer(ConstantNet(-5.0), (80, 80, 80), 'cpu').trace((40, 40, 40))
    self.assertEqual(trace.iterations, 1)  # the first step overwrites the seed (old > 0) with -5
    self.assertEqual(trace.seed[40, 40, 40], -5.0)

  def test_recorded_step_sees_the_padded_input_and_the_raw_output(self):
    seen = []
    def on_step(step, corner, mask, logits):
      seen.append((step, corner.copy(), mask.clone(), logits.clone()))
    trace = distil.TorchTracer(ConstantNet(1.0), (60, 60, 60), 'cpu').trace((30, 30, 30), on_step=on_step)
    self.assertEqual(len(seen), trace.iterations)
    self.assertGreater(trace.iterations, 1)
    step, corner, mask, logits = seen[0]
    self.assertEqual((step, corner.tolist()), (0, [14, 14, 14]))
    expected = torch.zeros(1, 1, *reference.FOV_SIZE)
    expected[0, 0, 16, 16, 16] = distil.OPTIONS.init_activation
    self.assertTrue(torch.equal(mask, expected))  # NaN -> pad logit 0, the seed at logit(0.95)
    self.assertTrue(torch.equal(logits, torch.ones(reference.FOV_SIZE)))
    self.assertEqual([s[0] for s in seen], list(range(len(seen))))


class StudentTraceTest(unittest.TestCase):

  def test_student_net_is_the_decoder_on_the_box_and_traces(self):
    torch.manual_seed(0)
    decoder = fz.Decoder((1, 2, 4, 1)).eval()
    feats = torch.randn(32, 60, 60, 60)
    net = distil.StudentNet(decoder, feats)
    corner, mask = np.array([3, 10, 20]), torch.randn(1, 1, *reference.FOV_SIZE)
    with torch.no_grad():
      direct = decoder(fz.box_features(feats, corner + 16), mask)[0, 0]
    self.assertTrue(torch.equal(net(corner, mask), direct))
    trace = distil.TorchTracer(net, (60, 60, 60), 'cpu').trace((30, 30, 30))
    self.assertGreaterEqual(trace.iterations, 1)
    self.assertEqual(trace.history[0], (30, 30, 30))
    self.assertEqual(trace.seed.shape, (60, 60, 60))


class CropSummaryTest(unittest.TestCase):

  @staticmethod
  def _rec(i, p, r, steps=10, wall=1.0, merges=()):
    return {'object_id': i, 'precision': p, 'recall': r, 'steps': steps, 'trace_seconds': wall / 2,
            'wall_seconds': wall, 'merges': [list(m) for m in merges], 'merged_voxels': sum(m[1] for m in merges),
            'unlabeled_fraction': 0.1}

  def test_means_paired_times_and_signal(self):
    ffn = [self._rec(1, 1.0, 1.0, steps=1, wall=4.0), self._rec(2, 0.5, 0.5, wall=6.0, merges=[(9, 2000)])]
    st = [self._rec(1, 1.0, 0.5, wall=1.0), self._rec(2, 1.0, 1.0, steps=1, wall=1.0)]
    s = distil.crop_summary(ffn, st, ffn_setup_seconds=2.0, encode_seconds=1.0, required_objects=2)
    self.assertAlmostEqual(s['ffn']['mean_f1'], 0.75)
    self.assertAlmostEqual(s['student']['mean_f1'], (2 / 3 + 1.0) / 2)
    self.assertEqual(s['ffn']['merges_ge_1000'], 1)
    self.assertEqual(s['ffn']['merged_voxels'], 2000)
    self.assertEqual(s['ffn']['stopped_at_first_step'], 1)
    self.assertAlmostEqual(s['student']['mean_unlabeled_fraction'], 0.1)
    self.assertEqual((s['paired']['student_higher'], s['paired']['ffn_higher']), (1, 1))
    self.assertAlmostEqual(s['crop_seconds']['ffn'], 12.0)
    self.assertAlmostEqual(s['crop_seconds']['student'], 3.0)
    self.assertAlmostEqual(s['speedup'], 4.0)
    self.assertFalse(s['signal']['f1_met'])  # 0.833 < 0.86
    self.assertTrue(s['signal']['speedup_met'])
    self.assertFalse(s['signal']['passed'])

  def test_speedup_boundary_is_inclusive(self):
    ffn = [self._rec(1, 1.0, 1.0, wall=1.0)]
    st = [self._rec(1, 0.875, 0.875, wall=0.5)]  # F1 exactly 0.875 in binary
    s = distil.crop_summary(ffn, st, ffn_setup_seconds=1.0, encode_seconds=0.5, required_objects=1)
    self.assertAlmostEqual(s['speedup'], 2.0)
    self.assertTrue(s['signal']['passed'])
    st = [self._rec(1, 0.75, 0.75, wall=0.5)]
    self.assertFalse(distil.crop_summary(ffn, st, 1.0, 0.5, required_objects=1)['signal']['f1_met'])

  def test_f1_boundary_is_inclusive_despite_float_error(self):
    n = distil.SCORE_OBJECTS
    ffn = [self._rec(i, 1.0, 1.0, wall=1.0) for i in range(n)]
    at = [self._rec(i, 0.86, 0.86, wall=0.1) for i in range(n)]  # each F1 computes as 0.8599999999999999
    s = distil.crop_summary(ffn, at, 0.0, 0.0)
    self.assertLess(s['student']['mean_f1'], 0.86)
    self.assertTrue(s['signal']['f1_met'])
    self.assertTrue(s['signal']['passed'])
    below = [self._rec(i, 0.8599, 0.8599, wall=0.1) for i in range(n)]
    self.assertFalse(distil.crop_summary(ffn, below, 0.0, 0.0)['signal']['f1_met'])

  def test_no_verdict_for_an_incomplete_run(self):
    s = distil.crop_summary([self._rec(1, 1, 1, wall=9.0)], [self._rec(1, 1, 1, wall=1.0)], 0.0, 0.0)
    self.assertFalse(s['signal']['eligible'])
    self.assertIsNone(s['signal']['passed'])
    self.assertTrue(s['signal']['f1_met'] and s['signal']['speedup_met'])

  def test_objects_must_match(self):
    with self.assertRaises(ValueError):
      distil.crop_summary([self._rec(1, 1, 1)], [self._rec(2, 1, 1)], 0.0, 0.0)
    with self.assertRaises(ValueError):
      distil.crop_summary([self._rec(1, 1, 1)] * 2, [self._rec(1, 1, 1)] * 2, 0.0, 0.0)


class BatchableNet:
  """A stand-in network whose output for a box does not depend on the batch: the mask, scaled,
  plus a corner-dependent offset that makes some moves fail."""

  def _one(self, corner, mask):
    bias = 1.0 if int(np.sum(corner)) % 7 < 5 else -3.0
    return 0.9 * mask[0] + bias

  def __call__(self, corner, mask):
    return self._one(corner, mask[0])

  def batch(self, corners, masks):
    return torch.stack([self._one(c, m) for c, m in zip(corners, masks)])


class TimingPiecesTest(unittest.TestCase):

  def test_box_copy_rebuilds_the_full_canvas(self):
    for net, shape, seed in ((ConstantNet(1.0), (60, 60, 60), (30, 30, 30)),
                             (ConstantNet(-5.0), (80, 80, 80), (40, 40, 40)),
                             (BatchableNet(), (70, 70, 70), (35, 30, 40))):
      full = distil.TorchTracer(net, shape, 'cpu').trace(seed)
      boxed = distil.TorchTracer(net, shape, 'cpu').trace(seed, box_only=True)
      self.assertIsNone(boxed.seed)
      self.assertEqual((boxed.history, boxed.iterations), (full.history, full.iterations))
      np.testing.assert_array_equal(distil.full_canvas(boxed, shape), full.seed)  # NaN positions included
      self.assertIs(distil.full_canvas(full, shape), full.seed)

  def test_batch_tracer_equals_one_at_a_time(self):
    shape = (70, 70, 70)
    seeds = [(35, 35, 35), (30, 40, 35), (40, 30, 30), (20, 20, 20), (45, 25, 33)]
    one = [distil.TorchTracer(BatchableNet(), shape, 'cpu').trace(s) for s in seeds]
    self.assertGreater(len({t.iterations for t in one}), 1)
    seeds = seeds + [(25, 45, 40), (33, 33, 20)]
    one += [distil.TorchTracer(BatchableNet(), shape, 'cpu').trace(s) for s in seeds[5:]]
    for width in (1, 2, 3, 8):
      got = distil.BatchTracer(BatchableNet(), shape, 'cpu', width).trace_all(seeds)
      for a, b in zip(one, got):
        self.assertEqual((b.history, b.iterations), (a.history, a.iterations))
        np.testing.assert_array_equal(distil.full_canvas(b, shape), a.seed)
    for width in (0, -1, 2.0, True, None):  # a width below 1 would loop forever
      with self.assertRaises(ValueError):
        distil.BatchTracer(BatchableNet(), shape, 'cpu', width)

  def test_mixed_formats(self):
    torch.manual_seed(0)
    dec = fz.Decoder((1, 2, 4, 1)).eval()
    torch.nn.init.normal_(dec.mask_b.weight, std=0.05)
    stack = fz.ConvStack(depth=2).eval()
    f, m = torch.randn(2, 32, 33, 33, 33), torch.randn(2, 1, 33, 33, 33)
    x = torch.randn(2, 2, 33, 33, 33)
    with torch.no_grad():
      ref_dec, ref_stack = dec(f, m), stack(x)
    self.assertTrue(torch.equal(distil.MixedNet(dec, 'fp32')(f, m), ref_dec))
    for fmt in ('fp16', 'fp16_head32'):
      for module, inputs, ref in ((dec, (f, m), ref_dec), (stack, (x,), ref_stack)):
        net = distil.MixedNet(module, fmt, half=torch.bfloat16)  # bfloat16: CPU convolutions
        out = net(*inputs)
        self.assertEqual(out.dtype, torch.float32)
        self.assertLess(float((out - ref).abs().max()), 0.1 * float(ref.abs().max()))
        head_dtype = torch.float32 if fmt == 'fp16_head32' else torch.bfloat16
        self.assertEqual(net.module.head.output.weight.dtype, head_dtype)
        self.assertEqual(net.module.blocks[0].conv_a.weight.dtype, torch.bfloat16)
        self.assertEqual(net.module.blocks[0].norm.scale.dtype, torch.float32)
        self.assertEqual(module.blocks[0].conv_a.weight.dtype, torch.float32)  # the original is untouched
    net = distil.MixedNet(dec, 'fp16', half=torch.bfloat16)
    net(f, m)
    self.assertEqual(int(net.nonfinite), 0)
    net(f, torch.full_like(m, float('inf')))
    self.assertGreater(int(net.nonfinite), 0)
    self.assertIsNone(distil.MixedNet(dec, 'fp32').nonfinite)
    with self.assertRaises(ValueError):
      distil.MixedNet(dec, 'int8')

  def test_scaled_stream_is_the_same_network(self):
    torch.manual_seed(0)
    dec = fz.Decoder((1, 2, 4, 1)).eval()
    torch.nn.init.normal_(dec.mask_b.weight, std=0.05)
    stack = fz.ConvStack(depth=2).eval()
    with torch.no_grad():  # a stream as large as the real ones (thousands), so that epsilon is negligible
      stack.pre_b.weight.mul_(1e4)
    f, m = 3000 * torch.randn(2, 32, 33, 33, 33), torch.randn(2, 1, 33, 33, 33)
    x = torch.randn(2, 2, 33, 33, 33)
    for module, inputs in ((dec, (f, m)), (stack, (x,))):
      # float64 as the reduced type isolates the rescaling from rounding
      plain = distil.MixedNet(module, 'fp16_head32', half=torch.float64)
      scaled = distil.MixedNet(module, 'fp16_head32', half=torch.float64, scale=1 / 16)
      a, b = plain(*inputs), scaled(*inputs)
      torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-4)
      blocks = scaled.module.blocks
      self.assertTrue(torch.equal(blocks[0].conv_b.weight, plain.module.blocks[0].conv_b.weight / 16))
      self.assertTrue(torch.equal(blocks[0].conv_a.weight, plain.module.blocks[0].conv_a.weight))
      self.assertTrue(torch.equal(module.blocks[0].conv_b.weight, plain.module.blocks[0].conv_b.weight.float()))
    for fmt, scale in (('fp32', 1 / 16), ('fp16', 0.1), ('fp16', -0.5), ('fp16', 0.0)):
      with self.assertRaises(ValueError):
        distil.MixedNet(dec, fmt, scale=scale)

  def test_norm_statistics_in_float32(self):
    norm = fz.ChannelLayerNorm()
    x = (25000 + 100 * torch.arange(32.0)).view(1, 32, 1, 1, 1).to(torch.float16)
    self.assertTrue(torch.isinf(x * x).any())  # float16 squares overflow
    y = distil._norm(norm, x)  # pylint: disable=protected-access
    self.assertEqual(y.dtype, torch.float16)
    self.assertTrue(torch.isfinite(y).all())
    self.assertTrue(torch.allclose(y.float(), norm(x.float()), atol=1e-2))

  def test_batched_nets_equal_single_calls(self):
    torch.manual_seed(0)
    dec = fz.Decoder((1, 2, 4, 1)).eval()
    feats = torch.randn(32, 60, 60, 60)
    s = distil.StudentNet(dec, feats)
    t = distil.TeacherNet(fz.ConvStack(depth=1).eval(), torch.randn(60, 60, 60), 'cpu')
    corners = [np.array([3, 10, 20]), np.array([20, 5, 0])]
    masks = torch.randn(2, 1, *reference.FOV_SIZE)
    for net in (s, t):
      b = net.batch(corners, masks)
      for j, c in enumerate(corners):
        torch.testing.assert_close(b[j], net(c, masks[j:j + 1]), rtol=1e-5, atol=1e-5)

  def test_encode_volume_out_device(self):
    torch.manual_seed(0)
    enc = fz.Encoder(depth=1).eval()
    image = np.random.default_rng(0).normal(size=(20, 20, 20)).astype(np.float32)
    with torch.no_grad():
      a = fz.encode_volume(enc, image, tile=8)
      b = fz.encode_volume(enc, image, tile=8, out_device='cpu')
    self.assertEqual(b.device.type, 'cpu')
    self.assertTrue(torch.equal(a, b))


class PrunedTest(unittest.TestCase):
  """Pruned FFN (docs/prune_design.md): copies of the teacher with fewer blocks, and their training."""

  def setUp(self):
    torch.manual_seed(0)
    self.teacher = fz.ConvStack().eval()
    for p in self.teacher.parameters():  # LayerNorm weights away from 1 and 0, so that copies are distinguishable
      torch.nn.init.normal_(p, std=0.05)

  def test_default_dilations_are_one(self):
    a, b = fz.ConvStack(depth=3), fz.ConvStack(dilations=(1, 1, 1))
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 2, 33, 33, 33)
    with torch.no_grad():
      self.assertTrue(torch.equal(a(x), b(x)))
    self.assertEqual([blk.conv_a.dilation for blk in fz.ConvStack(dilations=(1, 2, 4)).blocks],
                     [(1, 1, 1), (2, 2, 2), (4, 4, 4)])

  def test_pruned_copies_the_chosen_blocks(self):
    before = {k: v.clone() for k, v in self.teacher.state_dict().items()}
    p = fz.pruned_from_teacher(self.teacher, [16, 17, 18, 19], [1, 2, 4, 1])
    self.assertEqual(len(p.blocks), 4)
    for b, i in zip(p.blocks, (16, 17, 18, 19)):
      for (k, v), (_, w) in zip(b.state_dict().items(), self.teacher.blocks[i].state_dict().items()):
        self.assertTrue(torch.equal(v, w), (i, k))
    self.assertEqual(p.blocks[2].conv_b.dilation, (4, 4, 4))
    for name in ('pre_a', 'pre_norm', 'pre_b', 'head'):
      for (k, v), (_, w) in zip(getattr(p, name).state_dict().items(), getattr(self.teacher, name).state_dict().items()):
        self.assertTrue(torch.equal(v, w), (name, k))
    with torch.no_grad():
      p.blocks[0].conv_a.weight.add_(1.0)  # a copy, not shared
    for k, v in self.teacher.state_dict().items():
      self.assertTrue(torch.equal(v, before[k]), k)
    for keep, dil in (([0, 0], [1, 1]), ([20], [1]), ([3, 4], [1]), ([], [])):
      with self.assertRaises(ValueError):
        fz.pruned_from_teacher(self.teacher, keep, dil)

  def test_all_blocks_is_the_teacher(self):
    p = fz.pruned_from_teacher(self.teacher, range(20), [1] * 20)
    x = torch.randn(1, 2, 33, 33, 33)
    with torch.no_grad():
      self.assertTrue(torch.equal(p(x), self.teacher(x)))

  def test_box_gflop(self):
    self.assertAlmostEqual(fz.box_gflop(20), fz.flop_per_voxel()['teacher'] * 33 ** 3 / 1e9)
    self.assertAlmostEqual(fz.box_gflop(20), 81.6, places=1)
    self.assertAlmostEqual(fz.box_gflop(8), 33.9, places=1)
    self.assertAlmostEqual(fz.box_gflop(4), 18.0, places=1)

  def test_load_checkpoint_both_kinds(self):
    p = fz.pruned_from_teacher(self.teacher, [12, 13], [1, 2])
    with torch.no_grad():
      p.head.output.bias.add_(3.0)  # a trained network differs from its initialisation
    q = distil.load_checkpoint(self.teacher, {'net': p.state_dict(), 'keep': [12, 13], 'dilations': [1, 2], 'step': 5})
    self.assertIsInstance(q, fz.ConvStack)
    x = torch.randn(1, 2, 33, 33, 33)
    with torch.no_grad():
      self.assertTrue(torch.equal(q(x), p(x)))
    s = fz.student_from_teacher(self.teacher)
    with torch.no_grad():
      s.decoder.mask_b.bias.add_(1.0)
    t = distil.load_checkpoint(self.teacher, {'decoder': s.decoder.state_dict(), 'dilations': [1, 2, 4, 1], 'step': 5})
    self.assertIsInstance(t, fz.FactorisedFFN)
    self.assertTrue(torch.equal(t.decoder.mask_b.bias, s.decoder.mask_b.bias))

  def test_gather_inputs(self):
    images = {3: torch.randn(50, 50, 50), 7: torch.randn(50, 50, 50)}
    centers = np.array([[16, 20, 30], [25, 16, 33]], np.int16)
    masks = np.random.default_rng(0).normal(size=(2, 33, 33, 33)).astype(np.float16)
    x = distil.gather_inputs(images, np.array([3, 7]), centers, masks, 'cpu')
    self.assertEqual(tuple(x.shape), (2, 2, 33, 33, 33))
    for j, (k, c) in enumerate(zip((3, 7), centers)):
      lo = c - 16
      self.assertTrue(torch.equal(x[j, 0], images[k][lo[0]:lo[0] + 33, lo[1]:lo[1] + 33, lo[2]:lo[2] + 33]))
      self.assertTrue(torch.equal(x[j, 1], torch.from_numpy(masks[j]).float()))

  def test_scaled_stream_pruned(self):
    p = fz.pruned_from_teacher(self.teacher, [16, 17, 18, 19], [1, 2, 4, 1])
    with torch.no_grad():  # a stream as large as the real ones, so that epsilon is negligible
      p.pre_b.weight.mul_(1e4)
    x = torch.randn(2, 2, 33, 33, 33)
    plain = distil.MixedNet(p, 'fp16_head32', half=torch.float64)
    scaled = distil.MixedNet(p, 'fp16_head32', half=torch.float64, scale=1 / 16)
    torch.testing.assert_close(scaled(x), plain(x), rtol=1e-4, atol=1e-4)
    with torch.no_grad():
      torch.testing.assert_close(plain(x), p(x), rtol=1e-4, atol=1e-4)


class TrainPrunedTest(unittest.TestCase):
  """examples/train_pruned.py on tiny synthetic data (CPU): a round paused after every step and
  resumed equals the same round run in one go; round 1's step 0 cannot become best.pt."""

  def setUp(self):
    import sys  # pylint: disable=g-import-not-at-top
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'examples'))
    import train_pruned  # pylint: disable=g-import-not-at-top
    self.tp = train_pruned
    torch.manual_seed(0)
    teacher = fz.ConvStack().eval()
    rng = np.random.default_rng(0)

    def samples(n, crop):
      return {'centers': rng.integers(16, 24, (n, 3)).astype(np.int16), 'steps': np.array([0] + [5] * (n - 1), np.int32),
              'traces': np.zeros(n, np.int32), 'masks': rng.normal(size=(n, 33, 33, 33)).astype(np.float16),
              'targets': rng.normal(size=(n, 33, 33, 33)).astype(np.float16), 'crops': np.full(n, crop, np.int16)}

    split = {'train': samples(6, 0), 'valid': samples(3, distil.VALID_CROP)}
    raw = rng.integers(0, 255, (40, 40, 40)).astype(np.uint8)
    self.patches = [
        unittest.mock.patch.object(train_pruned.distil_teacher, 'load_export', lambda _: (teacher, {})),
        unittest.mock.patch.object(train_pruned.manyseed, 'load_raw_xyz', lambda _: raw),
        unittest.mock.patch.object(train_pruned.train_distil, 'load_split',
                                   lambda roots, crops: split['valid' if crops == (distil.VALID_CROP,) else 'train']),
        unittest.mock.patch.object(distil, 'BATCH', 2),
        unittest.mock.patch.object(distil, 'VALIDATION_SAMPLES', 2)]
    for p in self.patches:
      p.start()

  def tearDown(self):
    for p in self.patches:
      p.stop()

  def _args(self, out, *extra):
    return ['--export', 'x', '--teacher-out', 'x', '--keep', '18', '19', '--dilations', '1', '2', '--steps', '3',
            '--out', str(out), *extra]

  def test_resume_equals_one_run(self):
    with tempfile.TemporaryDirectory() as d, unittest.mock.patch.object(self.tp, 'LOG_EVERY', 2):
      one, split = Path(d) / 'one', Path(d) / 'split'
      self.tp.main(self._args(one))
      self.tp.main(self._args(split, '--max-hours', '0'))  # pauses after every step
      self.assertFalse((split / 'last.pt').exists())
      for _ in range(2):
        self.tp.main(self._args(split, '--max-hours', '0', '--resume'))
      a = torch.load(one / 'last.pt', weights_only=True)
      b = torch.load(split / 'last.pt', weights_only=True)
      for k, v in a['net'].items():
        self.assertTrue(torch.equal(v, b['net'][k]), k)
      self.assertEqual(a['keep'], [18, 19])
      log = json.loads((split / 'train_log.json').read_text())
      self.assertEqual(log['resumed_at'], [1, 2])
      self.assertEqual(len(log['job_seconds']), 3)
      self.assertEqual([v['step'] for v in log['valid']], [0, 3])
      one_log = json.loads((one / 'train_log.json').read_text())
      self.assertEqual([(e['step'], e['loss']) for e in log['train']], [(e['step'], e['loss']) for e in one_log['train']])

  def test_resume_rejects_changed_arguments(self):
    with tempfile.TemporaryDirectory() as d:
      out = Path(d) / 'run'
      self.tp.main(self._args(out, '--max-hours', '0'))
      changed = [['--export', 'x', '--teacher-out', 'x', '--keep', '18', '19', '--dilations', '1', '1', '--steps', '3',
                  '--out', str(out)], self._args(out, '--seed', '1'), self._args(out, '--extra', 'y')]
      for argv in changed:
        with self.assertRaises(ValueError):
          self.tp.main(argv + ['--max-hours', '0', '--resume'])
      self.tp.main(self._args(out, '--max-hours', '0', '--resume'))  # unchanged arguments continue
      self.assertEqual(json.loads((out / 'train_log.json').read_text())['resumed_at'], [1])

  def test_round1_step0_is_not_best(self):
    with tempfile.TemporaryDirectory() as d:
      r0, r1 = Path(d) / 'r0', Path(d) / 'r1'
      for out, extra in ((r0, ()), (r1, ('--init', str(Path(d) / 'r0' / 'last.pt')))):
        losses = iter([0.0, 1.0])  # validation loss rises after step 0, as it may on teacher states in round 1
        with unittest.mock.patch.object(self.tp, 'validate', lambda *_: {'loss': next(losses)}):
          self.tp.main(self._args(out, *extra))
      best0 = json.loads((r0 / 'train_log.json').read_text())['best']
      best1 = json.loads((r1 / 'train_log.json').read_text())['best']
      self.assertEqual(best0['step'], 0)  # round 0: the initialisation may be best
      self.assertEqual(best1['step'], 3)  # round 1: not the --init network
      self.assertEqual(torch.load(r1 / 'best.pt', weights_only=True)['step'], 3)


if __name__ == '__main__':
  unittest.main()
