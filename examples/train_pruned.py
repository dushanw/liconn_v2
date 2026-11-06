"""Pruned FFN: train a pruned copy of the authors' network on teacher samples (docs/prune_design.md).

  train_pruned.py --export DIR --teacher-out DIR --keep 16 17 18 19 --dilations 1 2 4 1 --steps N --out DIR
      [--probe] [--init CHECKPOINT] [--extra ONPOLICY_DIR]

The network is factorised.pruned_from_teacher(teacher, keep, dilations): the teacher's stem, the
residual blocks `keep` and the head, all trained. Its input is FFN's: the image box at the sample's
centre and the mask box. The normalised images of crops 0-11 are kept on the GPU in float32.
Everything else is as in train_distil.py: samples of crops 0-10 (plus --extra on-policy samples),
validation every 1,000 steps on the same 4,096 fixed crop-11 samples, OUT/best.pt (lowest
validation loss), OUT/last.pt and OUT/train_log.json; --probe runs 200 steps and reports the time
per step of steps 101-200; --init starts from a pruned checkpoint with the same blocks and
dilations (round 1). In round 1, step 0 (the --init network itself) is validated but cannot become
best.pt.

A round that does not fit in one job runs over several: when --max-hours is reached the job saves
OUT/resume.pt (network, optimiser, step, log) and OUT/resume_rng.json (the batch generator's
state) and stops; the same command with --resume continues from there, so the learning-rate
schedule and the batches are those of one uninterrupted run.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

from liconn_v2 import distil  # pylint: disable=g-import-not-at-top
from liconn_v2 import factorised as fz  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_teacher  # pylint: disable=g-import-not-at-top,wrong-import-position
import train_distil  # pylint: disable=g-import-not-at-top,wrong-import-position

VALID_BATCH = train_distil.VALID_BATCH
LOG_EVERY = train_distil.LOG_EVERY
PROBE_STEPS = train_distil.PROBE_STEPS
RESUME_SAME = ('export', 'teacher_out', 'keep', 'dilations', 'steps', 'out', 'seed', 'init', 'extra')  # a resumed run's
# arguments must equal the paused run's (dilations are not in the weights; seed sets the validation subset)


def sync(dev):
  if dev.type == 'cuda':
    torch.cuda.synchronize(dev)


def batch_tensors(s, idx, images, dev):
  """Network input (B, 2, 33, 33, 33) = [image box, mask box] and teacher logits (B, 33, 33, 33)."""
  x = distil.gather_inputs(images, s['crops'][idx], s['centers'][idx], s['masks'][idx], dev)
  t = torch.from_numpy(s['targets'][idx]).to(dev).float()
  return x, t


@torch.no_grad()
def validate(net, s, idx, images, dev) -> dict:
  net.eval()
  total, counts = 0.0, {}
  for i in range(0, len(idx), VALID_BATCH):
    j = idx[i:i + VALID_BATCH]
    x, t = batch_tensors(s, j, images, dev)
    out = net(x)[:, 0]
    total += float(distil.distil_loss(out, t)) * len(j)
    for key, v in distil.agreement(out, t, s['steps'][j]).items():
      counts[key] = counts.get(key, 0) + v
  net.train()
  return {'loss': total / len(idx),
          'interior_agreement': counts['interior_same'] / counts['interior_voxels'],
          'shell_agreement': counts['shell_same'] / counts['shell_voxels'],
          'first_step_teacher_moves': counts['teacher_moves'],
          'first_step_moves_kept': counts['both_move'] / max(1, counts['teacher_moves']),
          'first_step_moves_near': counts['both_move_near'] / max(1, counts['teacher_moves']),
          'first_step_extra_moves': counts['student_only_moves'],
          'interior_loss': counts['interior_loss_sum'] / counts['interior_voxels'],
          'shell_loss': counts['shell_loss_sum'] / counts['shell_voxels'], **counts}


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--teacher-out', type=Path, required=True)
  ap.add_argument('--keep', type=int, nargs='+', required=True, help='0-based teacher blocks, in order')
  ap.add_argument('--dilations', type=int, nargs='+', required=True)
  ap.add_argument('--steps', type=int, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--max-hours', type=float, default=5.3)
  ap.add_argument('--probe', action='store_true')
  ap.add_argument('--init', type=Path, default=None, help='pruned checkpoint to start from')
  ap.add_argument('--extra', type=Path, default=None, help='on-policy samples (OUT of distil_onpolicy.py)')
  ap.add_argument('--resume', action='store_true', help='continue from OUT/resume.pt')
  args = ap.parse_args(argv)
  args.out.mkdir(parents=True, exist_ok=True)
  state = None
  if args.resume:  # checked before anything is loaded
    state = torch.load(args.out / 'resume.pt', map_location='cpu', weights_only=True)
    paused = json.loads(state['log'])['args']
    changed = [k for k in RESUME_SAME if paused[k] != str(getattr(args, k))]
    if changed:
      raise ValueError(f'--resume with different {changed} from the paused run')
  t_start = time.time()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CPU only in the unit tests

  distil.device_settings(tf32=False)
  teacher, _ = distil_teacher.load_export(args.export)
  torch.manual_seed(args.seed)
  net = fz.pruned_from_teacher(teacher, args.keep, args.dilations)
  if args.init:
    ckpt = torch.load(args.init, map_location='cpu', weights_only=True)
    if list(ckpt.get('keep', [])) != list(args.keep) or list(ckpt['dilations']) != list(args.dilations):
      raise ValueError(f'--init has blocks {ckpt.get("keep")} and dilations {ckpt["dilations"]}, '
                       f'not {args.keep} and {args.dilations}')
    net.load_state_dict(ckpt['net'])
  images = {k: torch.as_tensor(reference.normalize(manyseed.load_raw_xyz(k)), dtype=torch.float32).to(dev)
            for k in distil.TRAIN_CROPS + (distil.VALID_CROP,)}  # sha256 checked by manyseed
  load_seconds = time.time() - t_start

  train = train_distil.load_split([args.teacher_out] + ([args.extra] if args.extra else []), distil.TRAIN_CROPS)
  valid = train_distil.load_split([args.teacher_out], (distil.VALID_CROP,))
  rng = np.random.default_rng(args.seed)
  first = np.flatnonzero(valid['steps'] == 0)
  rest = rng.choice(np.flatnonzero(valid['steps'] != 0), distil.VALIDATION_SAMPLES - len(first), replace=False)
  valid_idx = np.sort(np.concatenate([first, rest]))

  distil.device_settings(tf32=True)
  net = net.to(dev).train()
  opt = torch.optim.AdamW(net.parameters(), lr=distil.PEAK_LR, weight_decay=0.0)
  steps = PROBE_STEPS if args.probe else args.steps
  log = {'args': {k: str(v) for k, v in vars(args).items()}, 'train_samples': len(train['steps']),
         'extra_samples': int(sum(len(np.load(args.extra / f'crop-{k:02d}' / 'samples' / 'steps.npy'))
                                  for k in distil.TRAIN_CROPS)) if args.extra else 0,
         'valid_samples': len(valid_idx), 'valid_first_steps': len(first), 'load_seconds': load_seconds,
         'gflop_per_box': fz.box_gflop(len(args.keep)), 'parameters': sum(p.numel() for p in net.parameters()),
         'device': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu', 'torch': torch.__version__,
         'train': [], 'valid': [], 'best': None}

  def checkpoint(name, step):
    torch.save({'net': net.state_dict(), 'keep': list(args.keep), 'dilations': list(args.dilations), 'step': step},
               args.out / name)

  def run_validation(step):
    v = {'step': step, **validate(net, valid, valid_idx, images, dev)}
    log['valid'].append(v)
    candidate = step > 0 or not args.init  # round 1: the round-0 network is not round 1's best
    if candidate and (log['best'] is None or v['loss'] < log['best']['loss']):
      log['best'] = v
      checkpoint('best.pt', step)
    print(json.dumps(v), flush=True)

  start = 0
  if state is not None:
    net.load_state_dict(state['net'])
    opt.load_state_dict(state['opt'])
    start, log = state['step'], json.loads(state['log'])
    rng.bit_generator.state = json.loads((args.out / 'resume_rng.json').read_text())
    log.setdefault('resumed_at', []).append(start)
  elif not args.probe:
    run_validation(0)  # the network as initialised (or as loaded by --init)
  t_loop, t_probe = time.time(), None
  running = json.loads(state['running']) if state is not None else []
  for step in range(start, steps):
    if step == 100:
      sync(dev)
      t_probe = time.time()
    for g in opt.param_groups:
      g['lr'] = distil.learning_rate(step, steps)
    idx = np.sort(rng.integers(0, len(train['steps']), distil.BATCH))
    x, t = batch_tensors(train, idx, images, dev)
    loss = distil.distil_loss(net(x)[:, 0], t)
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
    if not args.probe and step + 1 < steps and time.time() - t_start > args.max_hours * 3600:
      log.setdefault('job_seconds', []).append(time.time() - t_start)
      torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'step': step + 1, 'log': json.dumps(log),
                  'running': json.dumps(running)},  # losses since the last log entry
                 args.out / 'resume.pt')
      (args.out / 'resume_rng.json').write_text(json.dumps(rng.bit_generator.state))
      (args.out / 'train_log.json').write_text(json.dumps(log, indent=1))
      print(json.dumps({'paused_at': step + 1, 'resume': str(args.out / 'resume.pt')}), flush=True)
      return
  sync(dev)
  if args.probe:
    log['probe_seconds_per_step'] = (time.time() - t_probe) / (PROBE_STEPS - 100)
    log['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else None
  elif log['valid'][-1]['step'] != steps:
    run_validation(steps)
  checkpoint('last.pt', steps)
  log['total_seconds'] = time.time() - t_start  # this job; earlier jobs of the round are in job_seconds
  if args.resume:
    log['job_seconds'].append(log['total_seconds'])
  (args.out / 'train_log.json').write_text(json.dumps(log, indent=1))
  print(json.dumps({k: log[k] for k in ('best', 'total_seconds')} |
                   ({'probe_seconds_per_step': log['probe_seconds_per_step'],
                     'peak_memory_gb': log['peak_memory_gb']} if args.probe else {})), flush=True)


if __name__ == '__main__':
  main()
