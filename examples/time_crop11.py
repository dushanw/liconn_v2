"""Timing study on crop 11: copies, number formats and several objects at once (docs/distil_design.md).

  time_crop11.py --export DIR --checkpoint best.pt --out DIR [--runs S1 S2 ...] [--max-objects N]
      [--p8 CHECKPOINT] [--p4 CHECKPOINT]

One GPU. The student (round-1 checkpoint) and FFN (the torch port) trace the 512 crop-11 objects
in the runs of RUNS, in that order. Untimed: loading, seeds, one warm-up before each run, and
scoring. Timed: the encoder (by the earlier path through the host and straight to the GPU), FFN's
image copy to the GPU, and the wall time of each run's traces. Accuracy, the effect of each
number format, and the identity of paired runs are written to timing.json. Crop 11 only.
Runs P1/P2 and Q1/Q2 trace the pruned FFNs P8 and P4 (docs/prune_design.md) from --p8 and --p4;
they take FFN's input, so their crop time adds the image copy, not an encoder.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
from liconn_v2 import distil  # pylint: disable=g-import-not-at-top
from liconn_v2 import factorised as fz  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_teacher  # pylint: disable=g-import-not-at-top,wrong-import-position
import train_distil  # pylint: disable=g-import-not-at-top,wrong-import-position

# name: (network, format, objects at once, copy back only the visited box)
RUNS = {
    'S1': ('student', 'fp32', 1, False),
    'S2': ('student', 'fp32', 1, True),
    'S3': ('student', 'fp16', 1, True),
    'S4': ('student', 'fp16_head32', 1, True),
    'S5': ('student', 'fp32', 8, True),
    'S6': ('student', 'fp16_head32', 8, True),
    'S7': ('student', 'fp16_head32', 32, True),
    'F1': ('ffn', 'fp32', 1, True),
    'F2': ('ffn', 'fp16', 1, True),
    'F3': ('ffn', 'fp16_head32', 1, True),
    'F4': ('ffn', 'fp16_head32', 8, True),
    'S8': ('student', 'fp16_head32', 1, True),  # S8, S9: residual stream scaled by SCALE (added after the smoke run)
    'S9': ('student', 'fp16_head32', 8, True),
    'S10': ('student', 'fp16_head32', 32, True),
    'F5': ('ffn', 'fp16_head32', 32, True),  # follow-up (job C): FFN with the student's fastest settings
    'F6': ('ffn', 'fp16', 32, True),  # job D: plain float16 with the scaled stream, head included
    'F7': ('ffn', 'fp16', 1, True),
    'S11': ('student', 'fp16', 32, True),
    'P1': ('p8', 'fp32', 1, True),  # pruned FFNs (docs/prune_design.md)
    'P2': ('p8', 'fp16_head32', 32, True),
    'Q1': ('p4', 'fp32', 1, True),
    'Q2': ('p4', 'fp16_head32', 32, True),
}
SCALE = {'S8': 1 / 16, 'S9': 1 / 16, 'S10': 1 / 16, 'F5': 1 / 16, 'F6': 1 / 16, 'F7': 1 / 16, 'S11': 1 / 16,
         'P2': 1 / 16, 'Q2': 1 / 16}  # the smoke run's student stream reached 70,661, above float16's 65,504
IDENTITY = (('S1', 'S2'), ('S5', 'S2'), ('S6', 'S4'), ('S7', 'S4'), ('F4', 'F3'), ('S9', 'S8'), ('S10', 'S8'), ('F6', 'F7'))
FP32_RUN = {'student': 'S2', 'ffn': 'F1', 'p8': 'P1', 'p4': 'Q1'}
PRUNED = ('p8', 'p4')
DROP = 0.1  # a per-object F1 fall larger than this is listed
Q = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def clock(dev) -> float:
  torch.cuda.synchronize(dev)
  return time.perf_counter()


def f1(r) -> float:
  return 2 * r['precision'] * r['recall'] / (r['precision'] + r['recall']) if r['precision'] + r['recall'] else 0.0


def mask_of(trace, shape) -> np.ndarray:
  """benchmark.object_mask of the full canvas; for a box-only trace, from the box (the canvas is
  NaN elsewhere, and NaN is below the threshold)."""
  if trace.seed is not None:
    return benchmark.object_mask(trace.seed)
  mask = np.zeros(tuple(shape), bool)
  lo, sub = trace.counters['box_lo'], trace.counters['box']
  mask[tuple(slice(int(a), int(a) + s) for a, s in zip(lo, sub.shape))] = benchmark.object_mask(sub)
  return mask


def record(trace, seed, gt, wall) -> dict:
  mask = mask_of(trace, gt.shape)
  s = benchmark.score(mask, gt, seed['seed_zyx'])
  return {'object_id': seed['object_id'], 'steps': trace.iterations, 'wall_seconds': wall,
          'precision': s['precision'], 'recall': s['recall'], 'f1': f1(s), 'merges': len(s['merges']),
          'merged_voxels': s['merged_voxels'], 'unlabeled_fraction': s['unlabeled_fraction'],
          'mask_sha256': hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
          'history_sha256': hashlib.sha256(np.asarray(trace.history, np.int64).tobytes()).hexdigest()}


def run(name, net, shape, seeds, gt, dev) -> dict:
  _, fmt, width, box_only = RUNS[name]
  if width == 1:
    tracer = distil.TorchTracer(net, shape, dev)
    tracer.trace(seeds[0]['seed_zyx'], box_only=box_only)  # warm-up, not timed
    records, wall = [], 0.0
    for s in seeds:
      t0 = time.perf_counter()
      trace = tracer.trace(s['seed_zyx'], box_only=box_only)  # ends with a copy to the host
      w = time.perf_counter() - t0
      wall += w
      records.append(record(trace, s, gt, w))
  else:
    tracer = distil.BatchTracer(net, shape, dev, width)
    tracer.trace_all([s['seed_zyx'] for s in seeds[:width]], box_only=box_only)  # warm-up
    t0 = clock(dev)
    traces = tracer.trace_all([s['seed_zyx'] for s in seeds], box_only=box_only)
    wall = clock(dev) - t0
    records = [record(t, s, gt, None) for t, s in zip(traces, seeds)]
  steps = sum(r['steps'] for r in records)
  summary = {'network': RUNS[name][0], 'format': fmt, 'width': width, 'box_only': box_only,
             'objects': len(records), 'wall_seconds': wall, 'steps': steps, 'ms_per_step': 1000 * wall / steps,
             'mean_f1': float(np.mean([r['f1'] for r in records])),
             'mean_recall': float(np.mean([r['recall'] for r in records])),
             'mean_precision': float(np.mean([r['precision'] for r in records])),
             'merges_ge_1000': int(sum(r['merges'] for r in records)),
             'merged_voxels': int(sum(r['merged_voxels'] for r in records)),
             'stopped_at_first_step': int(sum(r['steps'] == 1 for r in records))}
  print(json.dumps({'run': name, **summary}), flush=True)
  return {'summary': summary, 'records': records}


def nonfinite(net):
  """Non-finite network outputs over the run, warm-up included (None in fp32: not counted)."""
  n = net.model.nonfinite
  return None if n is None else int(n)


def identity(a, b) -> dict:
  ra, rb = a['records'], b['records']
  same = [x['steps'] == y['steps'] and x['mask_sha256'] == y['mask_sha256'] and x['history_sha256'] == y['history_sha256']
          for x, y in zip(ra, rb)]
  return {'objects': len(same), 'identical': int(sum(same)),
          'same_steps': int(sum(x['steps'] == y['steps'] for x, y in zip(ra, rb)))}


def format_effect(run_, base) -> dict:
  d = np.array([x['f1'] - y['f1'] for x, y in zip(run_['records'], base['records'])])
  return {'mean': float(d.mean()), 'quantiles': {'q': list(Q), 'value': [float(v) for v in np.quantile(d, Q)]},
          'objects_f1_drop_gt_0.1': [r['object_id'] for r, v in zip(run_['records'], d) if v < -DROP],
          'objects_f1_rise_gt_0.1': int((d > DROP).sum())}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--checkpoint', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--runs', nargs='+', default=[r for r in RUNS if RUNS[r][0] not in PRUNED])  # pruned runs opt in
  ap.add_argument('--max-objects', type=int, default=None, help='smoke runs only')
  ap.add_argument('--p8', type=Path, default=None, help='pruned checkpoint for runs P1, P2')
  ap.add_argument('--p4', type=Path, default=None, help='pruned checkpoint for runs Q1, Q2')
  args = ap.parse_args()
  if any(r not in RUNS for r in args.runs):
    raise ValueError(f'runs must be among {list(RUNS)}')
  for n in PRUNED:
    if getattr(args, n) is None and any(RUNS[r][0] == n for r in args.runs):
      raise ValueError(f'runs of {n} need --{n}')
  args.out.mkdir(parents=True, exist_ok=True)
  k = distil.VALID_CROP
  distil.device_settings(tf32=False)
  dev = torch.device('cuda')

  teacher, manifest = distil_teacher.load_export(args.export)
  ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
  student = fz.student_from_teacher(teacher, dilations=tuple(ckpt['dilations']))
  student.decoder.load_state_dict(ckpt['decoder'])
  image = reference.normalize(manyseed.load_raw_xyz(k))  # sha256 checked by manyseed
  gt = manyseed.load_proofread_zyx(k)
  seeds = vitseg.tuning_objects(gt)[:args.max_objects]
  out = {'crop': k, 'checkpoint': str(args.checkpoint), 'checkpoint_step': ckpt['step'],
         'teacher_sha256': manifest['teacher.pt'], 'objects': len(seeds), 'max_objects': args.max_objects,
         'device': torch.cuda.get_device_name(dev), 'torch': torch.__version__, 'tf32': False, 'runs': {}}

  # Encoder: the earlier path (tiles to the host, then one copy) and straight to the GPU.
  encoder = student.encoder.to(dev).eval()
  decoder = student.decoder.to(dev).eval()
  with torch.no_grad():
    encoder(torch.zeros((1, 1, 64, 64, 64), device=dev))  # warm-up, not timed
    t0 = clock(dev)
    host_path = fz.encode_volume(encoder, image, tile=train_distil.TILE, dtype=torch.float16).to(dev)
    t_host = clock(dev) - t0
    t0 = clock(dev)
    features = fz.encode_volume(encoder, image, tile=train_distil.TILE, dtype=torch.float16, out_device=dev)
    t_direct = clock(dev) - t0
  out['encoder'] = {'host_path_seconds': t_host, 'direct_seconds': t_direct,
                    'bitwise_equal': bool(torch.equal(host_path, features)),
                    'non_finite': int(sum(int((~torch.isfinite(c)).sum()) for c in features))}
  del host_path
  torch.cuda.empty_cache()
  print(json.dumps({'encoder': out['encoder']}), flush=True)
  if out['encoder']['non_finite']:
    sys.exit('encoder features are not finite in float16')
  if not out['encoder']['bitwise_equal']:
    (args.out / 'timing.json').write_text(json.dumps(out))
    sys.exit('features written straight to the GPU differ from the earlier path')

  for name in [r for r in args.runs if RUNS[r][0] == 'student']:
    fmt = RUNS[name][1]  # features are float16 already; reduced formats skip a float32 round trip
    net = distil.StudentNet(decoder, features, dtype=torch.float32 if fmt == 'fp32' else torch.float16, fmt=fmt,
                            scale=SCALE.get(name, 1.0))
    out['runs'][name] = run(name, net, image.shape, seeds, gt, dev)
    finish(out, name, net, args.out)
    del net
  del features
  torch.cuda.empty_cache()

  teacher = teacher.to(dev).eval()
  t0 = clock(dev)
  copy = torch.as_tensor(np.asarray(image, np.float32)).to(dev)
  out['ffn_image_copy_seconds'] = clock(dev) - t0
  del copy
  nets = {'ffn': teacher}
  for n in PRUNED:
    if getattr(args, n) is not None:
      c = torch.load(getattr(args, n), map_location='cpu', weights_only=True)
      nets[n] = distil.load_checkpoint(teacher, c).to(dev)
      out.setdefault('pruned', {})[n] = {'checkpoint': str(getattr(args, n)), 'step': c['step'], 'keep': c['keep'],
                                         'dilations': c['dilations']}
  for name in [r for r in args.runs if RUNS[r][0] != 'student']:
    net = distil.TeacherNet(nets[RUNS[name][0]], image, dev, fmt=RUNS[name][1], scale=SCALE.get(name, 1.0))
    out['runs'][name] = run(name, net, image.shape, seeds, gt, dev)
    finish(out, name, net, args.out)
    del net
    torch.cuda.empty_cache()
  print(json.dumps({key: out[key] for key in ('encoder', 'identity', 'crop_seconds', 'speedup_vs_F1') if key in out}, indent=1))


def finish(out, name, net, directory: Path):
  """Records the run's non-finite count, recomputes the comparisons and rewrites timing.json, so
  that a job stopped by its time limit keeps every finished run."""
  out['runs'][name]['summary']['nonfinite_outputs'] = nonfinite(net)
  out['runs'][name]['summary']['stream_scale'] = SCALE.get(name, 1.0)
  print(json.dumps({'run': name, 'nonfinite_outputs': nonfinite(net)}), flush=True)
  runs = out['runs']
  out['identity'] = {f'{a}_vs_{b}': identity(runs[a], runs[b]) for a, b in IDENTITY if a in runs and b in runs}
  out['format_effect'] = {n: format_effect(runs[n], runs[FP32_RUN[RUNS[n][0]]]) for n in runs
                          if FP32_RUN[RUNS[n][0]] in runs and RUNS[n][1] != 'fp32'}
  enc = out['encoder']['direct_seconds']
  ffn_fp32 = runs['F1']['summary']['wall_seconds'] + out['ffn_image_copy_seconds'] if 'F1' in runs else None
  out['crop_seconds'] = {n: r['summary']['wall_seconds'] + (enc if RUNS[n][0] == 'student' else out['ffn_image_copy_seconds'])
                         for n, r in runs.items()}
  if ffn_fp32:
    out['speedup_vs_F1'] = {n: ffn_fp32 / s for n, s in out['crop_seconds'].items()}
  (directory / 'timing.json').write_text(json.dumps(out))


if __name__ == '__main__':
  main()
