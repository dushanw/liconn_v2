"""Factorised FFN, part 2: train the student's mask decoder on teacher samples (docs/distil_design.md).

  train_distil.py --export DIR --teacher-out DIR --dilations 1 2 4 1 --steps N --out DIR [--probe]
      [--init CHECKPOINT] [--extra ONPOLICY_DIR]

The encoder is frozen (the teacher's stem and modules 0-15). Its features of crops 0-11 are
computed once on the GPU (float32, TF32 off) and kept on the GPU in float16. The decoder trains
on samples of crops 0-10 (TF32 allowed) and is validated every 1,000 steps on 4,096 fixed crop-11
samples (every step-0 sample, the rest drawn at random). OUT/best.pt holds the decoder with the
lowest validation loss; OUT/train_log.json the log. --probe runs 200 steps and reports the time
per step of steps 101-200. --init starts the decoder from a checkpoint; --extra adds the on-policy
samples of crops 0-10 (distil_onpolicy.py) to the training set, drawn uniformly from the union.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top
import torch.nn.functional as F  # pylint: disable=g-import-not-at-top

from liconn_v2 import distil  # pylint: disable=g-import-not-at-top
from liconn_v2 import factorised as fz  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_teacher  # pylint: disable=g-import-not-at-top,wrong-import-position

TILE = 128
VALID_BATCH = 64
LOG_EVERY = 100
PROBE_STEPS = 200
CHECK_BOXES = 8


def encode(encoder, k, dev):
  """Float16 features of crop k on the GPU. Features beyond the float16 range (65,504) would be
  stored as inf; every stored value is checked, and the job stops if any is not finite."""
  image = reference.normalize(manyseed.load_raw_xyz(k))
  features = fz.encode_volume(encoder, image, tile=TILE, dtype=torch.float16).to(dev)
  bad = sum(int((~torch.isfinite(c)).sum()) for c in features)
  peak = max(float(c.abs().max()) for c in features)
  print(json.dumps({'crop': k, 'max_abs_float16_feature': peak, 'non_finite': bad}), flush=True)
  if bad:
    sys.exit(f'crop {k}: {bad} encoder features are not finite in float16')
  return features, image


@torch.no_grad()
def float16_error(encoder, image, features, dev, n=CHECK_BOXES) -> dict:
  """Features of random boxes from the encoder on the box with its halo (float32) against the
  stored float16 dense features."""
  r = encoder.radius
  padded = F.pad(torch.as_tensor(image)[None, None], (r,) * 6)
  worst, scale = 0.0, 0.0
  for c in fz.check_centers(image.shape, n, seed=1):
    lo = np.asarray(c) - 16
    x = padded[..., lo[0]:lo[0] + 33 + 2 * r, lo[1]:lo[1] + 33 + 2 * r, lo[2]:lo[2] + 33 + 2 * r].to(dev)
    f32 = encoder(x)[0, :, r:r + 33, r:r + 33, r:r + 33]
    f16 = fz.box_features(features, c)[0].float()
    worst = max(worst, float((f32 - f16).abs().max()))
    scale = max(scale, float(f32.abs().max()))
  return {'boxes': n, 'max_abs_diff': worst, 'max_abs_feature': scale}


def load_split(roots, crops):
  parts, crop_ids = [], []
  for root in roots:
    for k in crops:
      s = distil.load_samples(root / f'crop-{k:02d}' / 'samples', mmap='r')  # read once, by the concatenation
      parts.append(s)
      crop_ids.append(np.full(len(s['steps']), k, np.int16))
  out = distil.concat_samples(parts)
  out['crops'] = np.concatenate(crop_ids)
  return out


def batch_tensors(s, idx, features, dev):
  f = distil.gather_features(features, s['crops'][idx], s['centers'][idx]).float()
  m = torch.from_numpy(s['masks'][idx]).to(dev).float()[:, None]
  t = torch.from_numpy(s['targets'][idx]).to(dev).float()
  return f, m, t


@torch.no_grad()
def validate(decoder, s, idx, features, dev) -> dict:
  decoder.eval()
  total, counts = 0.0, {}
  for i in range(0, len(idx), VALID_BATCH):
    j = idx[i:i + VALID_BATCH]
    f, m, t = batch_tensors(s, j, features, dev)
    out = decoder(f, m)[:, 0]
    total += float(distil.distil_loss(out, t)) * len(j)
    for key, v in distil.agreement(out, t, s['steps'][j]).items():
      counts[key] = counts.get(key, 0) + v
  decoder.train()
  return {'loss': total / len(idx),
          'interior_agreement': counts['interior_same'] / counts['interior_voxels'],
          'shell_agreement': counts['shell_same'] / counts['shell_voxels'],
          'first_step_teacher_moves': counts['teacher_moves'],
          'first_step_moves_kept': counts['both_move'] / max(1, counts['teacher_moves']),
          'first_step_moves_near': counts['both_move_near'] / max(1, counts['teacher_moves']),
          'first_step_extra_moves': counts['student_only_moves'],
          'interior_loss': counts['interior_loss_sum'] / counts['interior_voxels'],
          'shell_loss': counts['shell_loss_sum'] / counts['shell_voxels'], **counts}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--teacher-out', type=Path, required=True)
  ap.add_argument('--dilations', type=int, nargs=4, required=True)
  ap.add_argument('--steps', type=int, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--max-hours', type=float, default=5.3)
  ap.add_argument('--probe', action='store_true')
  ap.add_argument('--init', type=Path, default=None, help='decoder checkpoint to start from')
  ap.add_argument('--extra', type=Path, default=None, help='on-policy samples (OUT of distil_onpolicy.py)')
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  t_start = time.time()
  dev = torch.device('cuda')

  distil.device_settings(tf32=False)
  teacher, _ = distil_teacher.load_export(args.export)
  student = fz.student_from_teacher(teacher, dilations=tuple(args.dilations), seed=args.seed)
  if args.init:
    ckpt = torch.load(args.init, map_location='cpu', weights_only=True)
    if list(ckpt['dilations']) != list(args.dilations):
      raise ValueError(f'--init has dilations {ckpt["dilations"]}, not {args.dilations}')
    student.decoder.load_state_dict(ckpt['decoder'])
  encoder = student.encoder.to(dev).eval()
  features, f16 = {}, None
  for k in distil.TRAIN_CROPS + (distil.VALID_CROP,):
    features[k], image = encode(encoder, k, dev)
    if k == distil.VALID_CROP:
      f16 = float16_error(encoder, image, features[k], dev)
  del encoder
  encode_seconds = time.time() - t_start
  print(json.dumps({'encode_seconds': encode_seconds, 'float16_features': f16}), flush=True)

  train = load_split([args.teacher_out] + ([args.extra] if args.extra else []), distil.TRAIN_CROPS)
  valid = load_split([args.teacher_out], (distil.VALID_CROP,))
  rng = np.random.default_rng(args.seed)
  first = np.flatnonzero(valid['steps'] == 0)
  rest = rng.choice(np.flatnonzero(valid['steps'] != 0), distil.VALIDATION_SAMPLES - len(first), replace=False)
  valid_idx = np.sort(np.concatenate([first, rest]))

  distil.device_settings(tf32=True)
  decoder = student.decoder.to(dev).train()
  opt = torch.optim.AdamW(decoder.parameters(), lr=distil.PEAK_LR, weight_decay=0.0)
  steps = PROBE_STEPS if args.probe else args.steps
  log = {'args': {k: str(v) for k, v in vars(args).items()}, 'train_samples': len(train['steps']),
         'extra_samples': int(sum(len(np.load(args.extra / f'crop-{k:02d}' / 'samples' / 'steps.npy'))
                                  for k in distil.TRAIN_CROPS)) if args.extra else 0,
         'valid_samples': len(valid_idx), 'valid_first_steps': len(first), 'encode_seconds': encode_seconds,
         'float16_features': f16, 'device': torch.cuda.get_device_name(dev), 'torch': torch.__version__,
         'train': [], 'valid': [], 'best': None, 'stopped_early': False}

  def checkpoint(name, step):
    torch.save({'decoder': decoder.state_dict(), 'dilations': list(args.dilations), 'step': step},
               args.out / name)

  def run_validation(step):
    v = {'step': step, **validate(decoder, valid, valid_idx, features, dev)}
    log['valid'].append(v)
    if log['best'] is None or v['loss'] < log['best']['loss']:
      log['best'] = v
      checkpoint('best.pt', step)
    print(json.dumps(v), flush=True)

  if not args.probe:
    run_validation(0)  # the student as initialised
  t_loop, t_probe, running = time.time(), None, []
  for step in range(steps):
    if step == 100:
      torch.cuda.synchronize()
      t_probe = time.time()
    for g in opt.param_groups:
      g['lr'] = distil.learning_rate(step, steps)
    idx = np.sort(rng.integers(0, len(train['steps']), distil.BATCH))
    f, m, t = batch_tensors(train, idx, features, dev)
    loss = distil.distil_loss(decoder(f, m)[:, 0], t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    running.append(float(loss))
    if (step + 1) % LOG_EVERY == 0:
      log['train'].append({'step': step + 1, 'loss': float(np.mean(running)), 'lr': opt.param_groups[0]['lr'],
                           'seconds': time.time() - t_loop})
      running = []
    if not args.probe and (step + 1) % distil.VALIDATION_EVERY == 0:
      run_validation(step + 1)
      (args.out / 'train_log.json').write_text(json.dumps(log, indent=1))
    if time.time() - t_start > args.max_hours * 3600:
      log['stopped_early'] = step + 1
      run_validation(step + 1)
      break
  torch.cuda.synchronize()
  if args.probe:
    log['probe_seconds_per_step'] = (time.time() - t_probe) / (PROBE_STEPS - 100)
    log['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9
  elif not log['stopped_early'] and log['valid'][-1]['step'] != steps:
    run_validation(steps)
  checkpoint('last.pt', log['stopped_early'] or steps)
  log['total_seconds'] = time.time() - t_start
  (args.out / 'train_log.json').write_text(json.dumps(log, indent=1))
  print(json.dumps({k: log[k] for k in ('best', 'total_seconds', 'stopped_early')} |
                   ({'probe_seconds_per_step': log['probe_seconds_per_step'],
                     'peak_memory_gb': log['peak_memory_gb']} if args.probe else {})), flush=True)


if __name__ == '__main__':
  main()
