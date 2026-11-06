"""Factorised FFN, part 2: tracing with PyTorch networks, and distillation (docs/distil_design.md).

TorchTracer repeats fasttrace.FastTracer with a PyTorch network: the same input assembly, canvas
update and six face maxima (here as torch operations), and FFN's own FaceMaxMovementPolicy and
MovementRestrictor on the host. The network is the ported teacher (image box + mask box) or the
student (encoder features of the box + mask box). Tensors are (batch, channels, z, y, x).
"""

import collections
import copy
import math
import time
import types

import numpy as np
from scipy.special import logit
import torch
import torch.nn.functional as F

from liconn_v2 import factorised as fz
from liconn_v2 import fasttrace
from liconn_v2 import reference

SIZE = reference.FOV_SIZE
DELTAS = reference.DELTAS
FACE = 17  # each face is the 17 x 17 square around the box centre at +-8 (deltas 8)


def _option(p) -> float:
  """FFN Canvas's threshold as a logit (fasttrace.canvas_options): the probability and its logit
  are each stored in a float32 InferenceOptions field."""
  return float(np.float32(logit(float(np.float32(p)))))


OPTIONS = types.SimpleNamespace(init_activation=_option(reference.INIT_ACTIVATION),
                                pad_value=_option(reference.PAD_VALUE),
                                move_threshold=_option(reference.MOVE_THRESHOLD),
                                segment_threshold=_option(reference.SEGMENT_THRESHOLD))

# Samples kept from a teacher trace.
EARLY_STEPS = 4  # steps 0-3 of every trace (the first-step reach risk)
KEEP_FRACTION = 0.25  # each later step with this probability
SHELL = 4  # the outer shell of the box where the teacher saw zero padding

# Optimisation (docs/distil_design.md).
BATCH = 32
PEAK_LR = 3e-4
WARMUP = 500
VALIDATION_SAMPLES = 4096
VALIDATION_EVERY = 1000
TRAIN_CROPS = tuple(range(11))
VALID_CROP = 11


def device_settings(tf32: bool) -> None:
  """TF32 off for the teacher, numerics checks and traces; allowed for training."""
  torch.backends.cuda.matmul.allow_tf32 = tf32
  torch.backends.cudnn.allow_tf32 = tf32


# ------------------------------------------------------------------ tracer

def box(corner):
  return tuple(slice(int(c), int(c) + s) for c, s in zip(corner, SIZE))


def faces(logits):
  """The six 17 x 17 faces of (..., 33, 33, 33) logits in FFN's order (z-, z+, y-, y+, x-, x+),
  flattened in C order: (..., 6, 289)."""
  h = np.array(SIZE) // 2
  sub = [slice(int(c) - 8, int(c) + 9) for c in h]
  out = []
  for axis in range(3):
    for offset in (-8, 8):
      sel = list(sub)
      sel[axis] = int(h[axis]) + offset
      out.append(logits[(Ellipsis, *sel)].reshape(*logits.shape[:-3], FACE * FACE))
  return torch.stack(out, dim=-2)


def first_max(f):
  """fasttrace's first_max: the first NaN if there is one, otherwise the first maximum."""
  nan = torch.isnan(f)
  return torch.where(nan.any(-1), torch.argmax(nan.to(torch.uint8), -1), torch.argmax(f, -1))


def assemble_mask(canvas, corner):
  """The mask input of one box: canvas logits with NaN -> FFN's pad value, (1, 1, 33, 33, 33)."""
  s = canvas[box(corner)]
  return torch.where(torch.isnan(s), torch.full_like(s, OPTIONS.pad_value), s)[None, None]


def update(canvas, logits, corner):
  """fasttrace's update in place on `canvas`: FFN's seed-consistency rule, the canvas write, and
  the six face maxima and their positions within each face (one host array (6, 3))."""
  sl = box(corner)
  old = canvas[sl]
  active = (logits >= np.float32(OPTIONS.move_threshold)).any()
  keep = active & (old < 0) & (logits > old)  # th_max = logit(0.5) = 0
  logits = torch.where(keep, old, logits)
  canvas[sl] = logits
  f = faces(logits)
  flat = first_max(f)
  scores = torch.gather(f, 1, flat[:, None])[:, 0]
  return torch.stack([scores, (flat // FACE).to(scores.dtype), (flat % FACE).to(scores.dtype)], 1)


FORMATS = ('fp32', 'fp16', 'fp16_head32')


def _norm(norm, x):
  """LayerNorm with its statistics in float32 (as flax's LayerNorm), returned in x's dtype."""
  return norm(x.float()).to(x.dtype)


class MixedNet:
  """A network (fz.ConvStack or fz.Decoder) in one of FORMATS (docs/distil_design.md, timing
  study). fp32 is the module itself. fp16: convolutions, ReLUs and residual additions in `half`
  (float16) with `half` weights; LayerNorm statistics in float32. fp16_head32: as fp16, with the
  head (LayerNorm, ReLU, 1x1x1 convolution) in float32. Inputs and outputs are float32. In the
  reduced formats `nonfinite` counts non-finite outputs on the device (read it once at the end;
  reading it per call would synchronise).

  `scale` (a power of two, reduced formats only) multiplies the residual stream: the features (or
  the stem's output), the mask stem's output and each block's last convolution are scaled, which
  keeps the stream inside float16's range (largest 65,504). Every LayerNorm ignores the scale of its
  input, so the network is unchanged in exact arithmetic, apart from LayerNorm's epsilon (1e-6)."""

  def __init__(self, module, fmt='fp32', half=torch.float16, scale=1.0):
    if fmt not in FORMATS:
      raise ValueError(f'unknown format {fmt}')
    if scale != 1.0 and (fmt == 'fp32' or scale <= 0 or math.frexp(scale)[0] != 0.5):
      raise ValueError('scale must be a power of two, in a reduced format')
    self.fmt, self.half, self.scale = fmt, half, scale
    self.module = module
    self.nonfinite = None
    if fmt != 'fp32':
      self.module = copy.deepcopy(module)
      if scale != 1.0:
        last = [b.conv_b for b in self.module.blocks]
        last.append(self.module.mask_b if isinstance(self.module, fz.Decoder) else self.module.pre_b)
        with torch.no_grad():
          for conv in last:
            conv.weight.mul_(scale)
            conv.bias.mul_(scale)
      for name, m in self.module.named_modules():
        if isinstance(m, torch.nn.Conv3d) and not (fmt == 'fp16_head32' and name == 'head.output'):
          m.to(half)

  def _blocks(self, x):
    for b in self.module.blocks:
      x = x + b.conv_b(F.relu(b.conv_a(F.relu(_norm(b.norm, x)))))
    h = self.module.head
    if self.fmt == 'fp16_head32':
      x = x.float()
    out = h.output(F.relu(_norm(h.norm, x))).float()
    bad = (~torch.isfinite(out)).sum()
    self.nonfinite = bad if self.nonfinite is None else self.nonfinite + bad
    return out

  @torch.no_grad()
  def __call__(self, *inputs):
    m = self.module
    if self.fmt == 'fp32':
      return m(*inputs)
    if isinstance(m, fz.Decoder):
      features, mask = (v.to(self.half) for v in inputs)
      if self.scale != 1.0:
        features = features * self.scale
      return self._blocks(features + m.mask_b(F.relu(m.mask_a(mask))))
    x = inputs[0].to(self.half)
    return self._blocks(m.pre_b(F.relu(_norm(m.pre_norm, m.pre_a(x)))))


class TeacherNet:
  """The authors' network on one box: [image, mask] -> logits (33, 33, 33)."""

  def __init__(self, teacher, image_zyx, device, fmt='fp32', scale=1.0):
    self.model = MixedNet(teacher.to(device).eval(), fmt, scale=scale)
    self.image = torch.as_tensor(np.asarray(image_zyx, np.float32)).to(device)

  @torch.no_grad()
  def __call__(self, corner, mask):
    return self.model(torch.cat([self.image[box(corner)][None, None], mask], 1))[0, 0]

  @torch.no_grad()
  def batch(self, corners, masks):
    """(B, 1, 33, 33, 33) masks of B boxes -> (B, 33, 33, 33) logits."""
    image = torch.stack([self.image[box(c)] for c in corners])[:, None]
    return self.model(torch.cat([image, masks], 1))[:, 0]


class StudentNet:
  """The student's decoder on one box: (encoder features of the box, mask) -> logits."""

  def __init__(self, decoder, features, dtype=torch.float32, fmt='fp32', scale=1.0):
    self.model = MixedNet(decoder.eval(), fmt, scale=scale)
    self.features = features  # (C, Z, Y, X) on the decoder's device
    self.dtype = dtype

  @torch.no_grad()
  def __call__(self, corner, mask):
    f = self.features[(slice(None), *box(corner))][None].to(self.dtype)
    return self.model(f, mask.to(self.dtype))[0, 0].float()

  @torch.no_grad()
  def batch(self, corners, masks):
    f = torch.stack([self.features[(slice(None), *box(c))] for c in corners]).to(self.dtype)
    return self.model(f, masks.to(self.dtype))[:, 0].float()


class _View:
  """What fasttrace._CanvasView asks of its tracer."""

  def __init__(self, tracer):
    self.value = tracer.value
    self.opts = OPTIONS
    self.margin = tracer.margin
    self.shape = tracer.shape


class TorchTracer:
  """Traces like fasttrace.FastTracer, with a PyTorch network `net(corner, mask) -> logits`."""

  def __init__(self, net, shape_zyx, device):
    self.net = net
    self.device = torch.device(device)
    self.shape = np.array(shape_zyx)
    self.margin = np.array(SIZE) // 2
    self._canvas = None

  def value(self, pos) -> np.float32:
    return np.float32(self._canvas[tuple(int(v) for v in pos)].item())

  def trace(self, seed_zyx, on_step=None, box_only=False) -> reference.Trace:
    """`on_step(step, corner, mask, logits)`, if given, sees each network input and raw output.
    With box_only, only the visited box of the canvas is copied back (Trace.seed is None;
    full_canvas rebuilds it)."""
    from ffn.inference import movement  # pylint: disable=g-import-not-at-top
    start = tuple(int(v) for v in seed_zyx)
    self._canvas = torch.full(tuple(self.shape), float('nan'), dtype=torch.float32, device=self.device)
    self._canvas[start] = OPTIONS.init_activation  # a float32 value (OPTIONS)
    view = fasttrace._CanvasView(_View(self))  # pylint: disable=protected-access
    policy = movement.FaceMaxMovementPolicy(view, deltas=DELTAS, score_threshold=OPTIONS.move_threshold)
    restrictor = movement.MovementRestrictor()
    policy.reset_state(start)
    if not policy:
      policy.append((policy.score_threshold * 2, start))
    history, iterations = [], 0
    if self.device.type == 'cuda':
      torch.cuda.synchronize(self.device)
    t0 = time.perf_counter()
    for pos in policy:  # Canvas.segment_at
      if self.value(start) < OPTIONS.move_threshold:
        break
      if not restrictor.is_valid_pos(pos):
        continue
      corner = np.array(pos) - self.margin
      mask = assemble_mask(self._canvas, corner)
      logits = self.net(corner, mask)
      if on_step is not None:
        on_step(iterations, corner, mask, logits)
      moves = update(self._canvas, logits, corner).cpu().numpy()  # one transfer
      iterations += 1
      fasttrace.queue_moves(policy, moves[:, 0], moves[:, 1:].astype(np.int64), pos, DELTAS)
      history.append(tuple(int(v) for v in pos))
    seed, counters = _copy_back(self._canvas, start, history, self.margin, box_only)
    trace_seconds = time.perf_counter() - t0
    kind = torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'cpu'
    return reference.Trace(seed, history, iterations, 0.0, trace_seconds, counters, self.device.type, kind)


def _copy_back(canvas, start, history, margin, box_only):
  """The canvas on the host: whole, or only the bounding box of the seed and every evaluated box
  (elsewhere the canvas is NaN)."""
  if not box_only:
    return canvas.cpu().numpy(), {}
  pts = np.array([start, *history])
  lo = np.maximum(pts.min(0) - margin, 0)
  hi = np.minimum(pts.max(0) + margin + 1, canvas.shape)
  return None, {'box_lo': lo, 'box': canvas[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))].cpu().numpy()}


def full_canvas(trace, shape) -> np.ndarray:
  """The whole canvas of a trace made with box_only (or the canvas itself)."""
  if trace.seed is not None:
    return trace.seed
  out = np.full(tuple(shape), np.nan, np.float32)
  lo, sub = trace.counters['box_lo'], trace.counters['box']
  out[tuple(slice(int(a), int(a) + s) for a, s in zip(lo, sub.shape))] = sub
  return out


class _Slot:
  """One open trace of BatchTracer: its canvas, FFN's policy and restrictor, and its history."""

  def __init__(self, index, seed_zyx, shape, device):
    from ffn.inference import movement  # pylint: disable=g-import-not-at-top
    self.index = index
    self.start = tuple(int(v) for v in seed_zyx)
    self.shape = shape
    self.margin = np.array(SIZE) // 2
    self.opts = OPTIONS
    self.canvas = torch.full(tuple(shape), float('nan'), dtype=torch.float32, device=device)
    self.canvas[self.start] = OPTIONS.init_activation
    self.view = fasttrace._CanvasView(_View(self))  # pylint: disable=protected-access; the policy keeps a weak reference
    self.policy = movement.FaceMaxMovementPolicy(self.view, deltas=DELTAS, score_threshold=OPTIONS.move_threshold)
    self.restrictor = movement.MovementRestrictor()
    self.policy.reset_state(self.start)
    if not self.policy:
      self.policy.append((self.policy.score_threshold * 2, self.start))
    self.positions = iter(self.policy)
    self.history, self.iterations = [], 0

  def value(self, pos) -> np.float32:
    return np.float32(self.canvas[tuple(int(v) for v in pos)].item())

  def next_position(self):
    """TorchTracer's loop up to its next network call: None when the trace ends."""
    for pos in self.positions:
      if self.value(self.start) < OPTIONS.move_threshold:
        return None
      if not self.restrictor.is_valid_pos(pos):
        continue
      return pos
    return None


class BatchTracer:
  """TorchTracer for up to `width` objects at once: one network call `net.batch(corners, masks)`
  per round for the boxes of all open traces, and one host transfer of their face maxima. Each
  trace has its own canvas and FFN's own policy, so its steps are those of TorchTracer whenever
  the network's output for a box does not depend on the batch."""

  def __init__(self, net, shape_zyx, device, width):
    self.net = net
    self.device = torch.device(device)
    self.shape = np.array(shape_zyx)
    self.margin = np.array(SIZE) // 2
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
      raise ValueError(f'width must be a positive integer, got {width!r}')
    self.width = width

  def trace_all(self, seeds, box_only=True) -> list:
    """Traces of `seeds` (z, y, x), in the same order."""
    pending = collections.deque(enumerate(seeds))
    slots, out = [], [None] * len(seeds)
    kind = torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'cpu'
    while pending or slots:
      while len(slots) < self.width and pending:
        slots.append(_Slot(*pending.popleft(), self.shape, self.device))
      ready, kept = [], []
      for s in slots:
        pos = s.next_position()
        while pos is None:  # the trace has ended: keep the batch full with the next object
          seed, counters = _copy_back(s.canvas, s.start, s.history, self.margin, box_only)
          out[s.index] = reference.Trace(seed, s.history, s.iterations, 0.0, 0.0, counters, self.device.type, kind)
          if not pending:
            break
          s = _Slot(*pending.popleft(), self.shape, self.device)
          pos = s.next_position()
        if pos is not None:
          ready.append((s, pos))
          kept.append(s)
      slots = kept
      if not ready:
        continue
      corners = [np.array(pos) - self.margin for _, pos in ready]
      masks = torch.cat([assemble_mask(s.canvas, c) for (s, _), c in zip(ready, corners)])
      logits = self.net.batch(corners, masks)
      moves = torch.stack([update(s.canvas, logits[j], c) for j, ((s, _), c) in enumerate(zip(ready, corners))])
      moves = moves.cpu().numpy()  # one transfer for the whole round
      for (s, pos), m in zip(ready, moves):
        s.iterations += 1
        fasttrace.queue_moves(s.policy, m[:, 0], m[:, 1:].astype(np.int64), pos, DELTAS)
        s.history.append(tuple(int(v) for v in pos))
    return out


# ------------------------------------------------------------------ teacher samples

class SampleRecorder:
  """Keeps steps 0-3 of a trace and each later step with probability 1/4 (a generator seeded by
  crop and trace, drawn once per step): input mask and teacher logits as float16."""

  def __init__(self, crop: int, trace: int):
    self.rng = np.random.default_rng([crop, trace])
    self.crop, self.trace = crop, trace
    self.centers, self.steps, self.masks, self.targets = [], [], [], []

  def keep(self, step: int) -> bool:
    u = self.rng.random()
    return step < EARLY_STEPS or u < KEEP_FRACTION

  def __call__(self, step, corner, mask, logits):
    if self.keep(step):
      self.store(step, corner, mask, logits)

  def store(self, step, corner, mask, target):
    self.centers.append(np.asarray(corner) + np.array(SIZE) // 2)
    self.steps.append(step)
    self.masks.append(mask[0, 0].to('cpu', torch.float16).numpy())
    self.targets.append(target.to('cpu', torch.float16).numpy())

  def arrays(self) -> dict:
    n = len(self.steps)
    return {'centers': np.array(self.centers, np.int16).reshape(n, 3),
            'steps': np.array(self.steps, np.int32),
            'traces': np.full(n, self.trace, np.int32),
            'masks': np.array(self.masks, np.float16).reshape(n, *SIZE),
            'targets': np.array(self.targets, np.float16).reshape(n, *SIZE)}


NEAR_MISS = 0.5  # logits below the move threshold that count as a near miss


class MoveCounts:
  """Face moves of student and teacher on the same inputs (network outputs, before FFN's canvas
  rule): a face moves if its maximum reaches the move threshold. For moves the student misses,
  `missed_gaps` holds how far its face maximum is below the threshold (logits)."""

  def __init__(self):
    self.steps = self.teacher = self.student = self.both = 0
    self.missed_gaps = []

  def count(self, student_logits, teacher_logits):
    t = np.float32(OPTIONS.move_threshold)
    sm, tm = faces(student_logits).amax(-1), faces(teacher_logits).amax(-1)  # (6,)
    sv, tv = sm >= t, tm >= t
    self.steps += 1
    self.teacher += int(tv.sum())
    self.student += int(sv.sum())
    self.both += int((sv & tv).sum())
    self.missed_gaps.extend((t - sm[tv & ~sv]).tolist())

  def add(self, other: 'MoveCounts'):
    self.steps += other.steps
    self.teacher += other.teacher
    self.student += other.student
    self.both += other.both
    self.missed_gaps.extend(other.missed_gaps)

  def summary(self) -> dict:
    g = np.array(self.missed_gaps)
    return {'steps': self.steps, 'teacher_moves': self.teacher, 'student_moves': self.student,
            'both_move': self.both, 'missed_moves': len(g), 'extra_moves': self.student - self.both,
            'missed_gap_median': float(np.median(g)) if len(g) else None,
            'missed_within_near_miss': int((g <= NEAR_MISS).sum())}


def written(logits, mask):
  """What `update` writes to the canvas for a batch (B, 33, 33, 33): FFN's rule keeps a voxel's
  old value where the box is active, the old value is < 0 and the new one is larger. The mask input
  equals the old canvas box (NaN -> 0, which the rule never keeps), so this is exact up to the
  float16 storage of the mask."""
  t = np.float32(OPTIONS.move_threshold)
  mask = mask.to(logits.dtype)
  active = (logits >= t).flatten(1).any(1).view(-1, 1, 1, 1)
  return torch.where(active & (mask < 0) & (logits > mask), mask, logits)


STEP_BINS = ((0, 1), (1, 4), (4, 50), (50, None))  # step 0, steps 1-3, 4-49, 50 and later


class MoveGaps:
  """Face moves of student and teacher after FFN's canvas rule, on the same inputs, by step bin.
  Per bin: moves of each network and of both, moves the student misses, and quantiles of
  (teacher face maximum - student face maximum) on faces where the teacher moves (a static offset
  shows up here on teacher states, where there is no drift) and of (threshold - student maximum)
  on missed moves.

  The same counts are kept for "frontier" faces: faces whose maximum in the mask input is below
  the move threshold (low or negative mask values there, which includes untraced faces). On
  frontier faces whose mean of teacher and student maxima is within NEAR_BAND of the threshold,
  the quantiles of (teacher - student) are reported. They are conditional descriptive quantiles,
  not an unbiased offset estimate: selecting on the mean still biases the difference when the two
  maxima vary by different amounts."""

  NEAR_BAND = 1.0

  def __init__(self):
    new = lambda: {'samples': 0, 'teacher_moves': 0, 'student_moves': 0, 'both_move': 0,
                   'teacher_minus_student': [], 'missed_below_threshold': [], 'near_threshold': []}
    self.bins = {self.name(b): {'all': new(), 'frontier': new()} for b in STEP_BINS}

  @staticmethod
  def name(b) -> str:
    lo, hi = b
    return f'{lo}+' if hi is None else (f'{lo}' if hi == lo + 1 else f'{lo}-{hi - 1}')

  def add(self, student_logits, teacher_logits, mask, steps):
    t = np.float32(OPTIONS.move_threshold)
    sm = faces(written(student_logits, mask)).amax(-1)  # (B, 6)
    tm = faces(written(teacher_logits, mask)).amax(-1)
    frontier = faces(mask.to(sm.dtype)).amax(-1) < t
    near = ((tm + sm) / 2 - t).abs() <= self.NEAR_BAND
    steps = np.asarray(steps)
    for b in STEP_BINS:
      sel = (steps >= b[0]) & (steps < (b[1] if b[1] is not None else np.inf))
      if not sel.any():
        continue
      i = torch.as_tensor(np.flatnonzero(sel), device=sm.device)
      for kind, keep in (('all', torch.ones_like(frontier[i])), ('frontier', frontier[i])):
        d = self.bins[self.name(b)][kind]
        s_, t_, n_ = sm[i][keep], tm[i][keep], near[i][keep]
        sv_, tv_ = s_ >= t, t_ >= t
        d['samples'] += len(i)
        d['teacher_moves'] += int(tv_.sum())
        d['student_moves'] += int(sv_.sum())
        d['both_move'] += int((sv_ & tv_).sum())
        d['teacher_minus_student'].extend((t_ - s_)[tv_].tolist())
        d['missed_below_threshold'].extend((t - s_)[tv_ & ~sv_].tolist())
        d['near_threshold'].extend((t_ - s_)[n_].tolist())

  def summary(self) -> dict:
    q = lambda v: {str(p): float(np.quantile(v, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(v) else None
    out = {}
    for k, kinds in self.bins.items():
      out[k] = {}
      for kind, d in kinds.items():
        out[k][kind] = {**{x: d[x] for x in ('samples', 'teacher_moves', 'student_moves', 'both_move')},
                        'missed_moves': len(d['missed_below_threshold']),
                        'teacher_minus_student_quantiles': q(d['teacher_minus_student']),
                        'missed_below_threshold_quantiles': q(d['missed_below_threshold']),
                        'missed_within_near_miss': int(np.sum(np.array(d['missed_below_threshold']) <= NEAR_MISS)),
                        'near_threshold_faces': len(d['near_threshold']),
                        'near_threshold_teacher_minus_student_quantiles': q(d['near_threshold'])}
    return out


class OnPolicyRecorder(SampleRecorder):
  """On-policy samples: the student traces. On each kept step (SampleRecorder's rule, with its own
  random stream) the teacher is run on the same box and mask input, and its output is stored as
  the target; `moves` compares the face moves of both networks on those steps."""

  STREAM = 1

  def __init__(self, crop: int, trace: int, teacher_net):
    super().__init__(crop, trace)
    self.rng = np.random.default_rng([crop, trace, self.STREAM])
    self.teacher_net = teacher_net
    self.moves = MoveCounts()

  def __call__(self, step, corner, mask, logits):
    if self.keep(step):
      target = self.teacher_net(corner, mask)
      self.moves.count(logits, target)
      self.store(step, corner, mask, target)


SAMPLE_KEYS = ('centers', 'steps', 'traces', 'masks', 'targets')


def concat_samples(parts: list) -> dict:
  return {k: np.concatenate([p[k] for p in parts]) for k in SAMPLE_KEYS}


def save_samples(directory, samples: dict) -> None:
  directory.mkdir(parents=True, exist_ok=True)
  for k in SAMPLE_KEYS:
    np.save(directory / f'{k}.npy', samples[k])


def load_samples(directory, mmap=None) -> dict:
  return {k: np.load(directory / f'{k}.npy', mmap_mode=mmap) for k in SAMPLE_KEYS}


# ------------------------------------------------------------------ teacher weights

def save_teacher(teacher, path) -> None:
  torch.save(teacher.state_dict(), path)


def load_teacher(path) -> fz.ConvStack:
  m = fz.ConvStack()
  m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
  return m.eval()


# ------------------------------------------------------------------ training

def gather_features(features, crops, centers):
  """Encoder features of a batch of boxes, (B, C, 33, 33, 33): features[crop] is (C, Z, Y, X)."""
  h = np.array(SIZE) // 2
  return torch.stack([features[int(k)][(slice(None), *box(np.asarray(c) - h))]
                      for k, c in zip(crops, centers)])


def gather_inputs(images, crops, centers, masks, device):
  """FFN's input for a batch of samples, (B, 2, 33, 33, 33): the image box at each sample's centre
  (images[crop] is the normalised (Z, Y, X) crop on `device`) and the sample's mask box."""
  h = np.array(SIZE) // 2
  image = torch.stack([images[int(k)][box(np.asarray(c) - h)] for k, c in zip(crops, centers)])
  mask = torch.from_numpy(np.asarray(masks)).to(device).float()
  return torch.stack([image, mask], 1)


def load_checkpoint(teacher, ckpt):
  """The network of a training checkpoint: a pruned FFN (train_pruned.py; the checkpoint has
  'keep'), returned as a fz.ConvStack, or the factorised student (train_distil.py; a decoder
  checkpoint), returned as a fz.FactorisedFFN. The teacher is unchanged."""
  if 'keep' in ckpt:
    net = fz.pruned_from_teacher(teacher, ckpt['keep'], ckpt['dilations'])
    net.load_state_dict(ckpt['net'])
    return net.eval()
  student = fz.student_from_teacher(teacher, dilations=tuple(ckpt['dilations']))
  student.decoder.load_state_dict(ckpt['decoder'])
  return student


def distil_loss(student_logits, teacher_logits):
  """Binary cross-entropy of the student against the teacher's probabilities, all voxels equal."""
  return F.binary_cross_entropy_with_logits(student_logits, torch.sigmoid(teacher_logits))


def learning_rate(step: int, total: int, peak=PEAK_LR, warmup=WARMUP) -> float:
  """Linear warm-up, then cosine decay to 0 at `total`."""
  if step < warmup:
    return peak * (step + 1) / warmup
  return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def shell_mask(shell=SHELL) -> torch.Tensor:
  """True on the outer `shell` voxels of the box."""
  inner = torch.zeros(SIZE, dtype=torch.bool)
  inner[shell:-shell, shell:-shell, shell:-shell] = True
  return ~inner


def face_moves(logits):
  """(B, 6): whether each face's maximum reaches the move threshold (FFN's face order)."""
  return faces(logits).amax(-1) >= np.float32(OPTIONS.move_threshold)


NEAR = 2  # face maxima at most this many voxels apart (in either face axis) count as the same move


def agreement(student_logits, teacher_logits, steps) -> dict:
  """Counts for validation: voxels on the same side of the 0.6 threshold (interior and shell), the
  loss on each part, and, for step-0 samples, faces where the teacher moves, where the student
  does, and where both move with face maxima within NEAR voxels of each other."""
  t = np.float32(OPTIONS.segment_threshold)
  same = (student_logits >= t) == (teacher_logits >= t)  # (B, 33, 33, 33)
  sh = shell_mask().to(same.device)
  bce = F.binary_cross_entropy_with_logits(student_logits, torch.sigmoid(teacher_logits), reduction='none')
  first = torch.as_tensor(np.asarray(steps) == 0, device=same.device)
  tf, sf = faces(teacher_logits[first]), faces(student_logits[first])
  ts, ss = face_moves(teacher_logits[first]), face_moves(student_logits[first])
  ti, si = first_max(tf), first_max(sf)
  near = ((ti // FACE - si // FACE).abs() <= NEAR) & ((ti % FACE - si % FACE).abs() <= NEAR)
  return {'interior_same': int(same[:, ~sh].sum()), 'interior_voxels': int((~sh).sum()) * len(same),
          'shell_same': int(same[:, sh].sum()), 'shell_voxels': int(sh.sum()) * len(same),
          'interior_loss_sum': float(bce[:, ~sh].sum()), 'shell_loss_sum': float(bce[:, sh].sum()),
          'first_steps': int(first.sum()), 'teacher_moves': int(ts.sum()),
          'both_move': int((ts & ss).sum()), 'both_move_near': int((ts & ss & near).sum()),
          'student_only_moves': int((ss & ~ts).sum())}


# ------------------------------------------------------------------ part 3: crop 12

SCORE_CROP = 12
SIGNAL_F1 = 0.86  # pilot signal fixed 2026-10-31 (docs/factorised_design.md)
SIGNAL_SPEEDUP = 2.0
SCORE_OBJECTS = 457  # the large objects of crop 12 (vitseg.tuning_objects)
SIGNAL_DECIMALS = 12  # both comparisons are inclusive; rounding removes float error (~1e-15) at the boundary


def _f1(r) -> float:
  p, q = r['precision'], r['recall']
  return 2 * p * q / (p + q) if p + q else 0.0


def crop_summary(ffn: list, student: list, ffn_setup_seconds: float, encode_seconds: float,
                 required_objects: int = SCORE_OBJECTS) -> dict:
  """Part 3 (docs/distil_design.md): accuracy of FFN and the student on the same objects in the
  same order, crop times (FFN: image to the GPU plus the traces' wall time; student: encoding plus
  the traces' wall time) and the pilot signal. The signal is decided only when all
  `required_objects` were scored; otherwise `eligible` is False and `passed` is None."""
  if [r['object_id'] for r in ffn] != [r['object_id'] for r in student]:
    raise ValueError('FFN and the student scored different objects')
  if not ffn:
    raise ValueError('no objects')
  if len({r['object_id'] for r in ffn}) != len(ffn):
    raise ValueError('an object is scored twice')
  out = {'objects': len(ffn)}
  for name, rows in (('ffn', ffn), ('student', student)):
    out[name] = {'mean_f1': float(np.mean([_f1(r) for r in rows])),
                 'mean_recall': float(np.mean([r['recall'] for r in rows])),
                 'mean_precision': float(np.mean([r['precision'] for r in rows])),
                 'mean_unlabeled_fraction': float(np.mean([r['unlabeled_fraction'] for r in rows])),
                 'merges_ge_1000': int(sum(len(r['merges']) for r in rows)),
                 'objects_with_merge': int(sum(bool(r['merges']) for r in rows)),
                 'merged_voxels': int(sum(r['merged_voxels'] for r in rows)),
                 'steps': int(sum(r['steps'] for r in rows)),
                 'stopped_at_first_step': int(sum(r['steps'] == 1 for r in rows)),
                 'trace_seconds': float(sum(r['trace_seconds'] for r in rows)),
                 'wall_seconds': float(sum(r['wall_seconds'] for r in rows))}
  diff = np.array([_f1(s) - _f1(f) for f, s in zip(ffn, student)])
  q = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
  out['paired'] = {'student_higher': int((diff > 0).sum()), 'ffn_higher': int((diff < 0).sum()),
                   'equal': int((diff == 0).sum()), 'mean_student_minus_ffn': float(diff.mean()),
                   'quantiles': {'q': list(q), 'value': [float(v) for v in np.quantile(diff, q)]}}
  ffn_crop = ffn_setup_seconds + out['ffn']['wall_seconds']
  student_crop = encode_seconds + out['student']['wall_seconds']
  out['crop_seconds'] = {'ffn': ffn_crop, 'ffn_setup': ffn_setup_seconds,
                         'student': student_crop, 'student_encode': encode_seconds}
  out['speedup'] = ffn_crop / student_crop
  s = {'f1_threshold': SIGNAL_F1, 'speedup_threshold': SIGNAL_SPEEDUP, 'decimals': SIGNAL_DECIMALS,
       'required_objects': required_objects, 'eligible': len(ffn) == required_objects,
       'f1_met': bool(round(out['student']['mean_f1'], SIGNAL_DECIMALS) >= SIGNAL_F1),
       'speedup_met': bool(round(out['speedup'], SIGNAL_DECIMALS) >= SIGNAL_SPEEDUP)}
  s['passed'] = (s['f1_met'] and s['speedup_met']) if s['eligible'] else None
  out['signal'] = s
  return out
