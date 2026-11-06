"""Factorised FFN, part 2b: on-policy samples (docs/distil_design.md, 'On-policy round').

  distil_onpolicy.py --export DIR --teacher-out DIR --checkpoint best.pt --out DIR --crops 0 1 2

For each crop (only 0-10): the student traces the seeds the teacher traced (TEACHER_OUT/crop-KK/
traces.json) and decides every move. The checkpoint may be the factorised student's or a pruned
FFN's (train_pruned.py, docs/prune_design.md); a pruned FFN takes FFN's own input. On the kept steps the teacher is run on the same box and
mask input, and its output is stored as the target (distil.OnPolicyRecorder). Writes
OUT/crop-KK/: samples/*.npy, traces.json (per trace: student steps, benchmark.score, the
teacher's F1 on the same seed, move diagnostics) and done.json. A crop with done.json is skipped.
"""

import argparse
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


def f1(p, r) -> float:
  return 2 * p * r / (p + r) if p + r else 0.0


def collect_crop(k: int, teacher, student, teacher_out: Path, out: Path, dev, max_traces=None) -> dict:
  if isinstance(student, fz.ConvStack):  # pruned FFN: image and mask boxes, no encoder
    image = reference.normalize(manyseed.load_raw_xyz(k))  # sha256 checked by manyseed
    net = distil.TeacherNet(student, image, dev)
  else:
    features, image = train_distil.encode(student.encoder.to(dev).eval(), k, dev)
    net = distil.StudentNet(student.decoder.to(dev), features)
  gt = manyseed.load_proofread_zyx(k)
  teacher_records = json.loads((teacher_out / f'crop-{k:02d}' / 'traces.json').read_text())[:max_traces]
  tracer = distil.TorchTracer(net, image.shape, dev)
  teacher_net = distil.TeacherNet(teacher, image, dev)
  parts, records, moves = [], [], distil.MoveCounts()
  t_crop = time.perf_counter()
  for tr in teacher_records:
    rec = distil.OnPolicyRecorder(k, tr['trace'], teacher_net)
    trace = tracer.trace(tr['seed_zyx'], on_step=rec)
    mask = benchmark.object_mask(trace.seed)
    parts.append(rec.arrays())
    moves.add(rec.moves)
    records.append({'trace': tr['trace'], 'crop': k, 'object_id': tr['object_id'], 'seed_zyx': tr['seed_zyx'],
                    'steps': trace.iterations, 'samples': len(rec.steps), 'trace_seconds': trace.trace_seconds,
                    'teacher_steps': tr['steps'], 'teacher_f1': f1(tr['precision'], tr['recall']),
                    'moves': rec.moves.summary(), **benchmark.score(mask, gt, tr['seed_zyx'])})
    if tr['trace'] % 50 == 0:
      print(f'crop {k} trace {tr["trace"]}/{len(teacher_records)} steps {trace.iterations}', flush=True)
  d = out / f'crop-{k:02d}'
  distil.save_samples(d / 'samples', distil.concat_samples(parts))
  (d / 'traces.json').write_text(json.dumps(records))
  steps = sum(r['steps'] for r in records)
  summary = {'crop': k, 'traces': len(records), 'steps': steps, 'teacher_steps': sum(r['teacher_steps'] for r in records),
             'samples': sum(r['samples'] for r in records),
             'student_mean_f1': vitseg.mean_f1(records),
             'teacher_mean_f1': float(np.mean([r['teacher_f1'] for r in records])),
             'moves': moves.summary(), 'crop_seconds': time.perf_counter() - t_crop,
             'device': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}
  (d / 'done.json').write_text(json.dumps(summary, indent=1))
  return summary


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--teacher-out', type=Path, required=True)
  ap.add_argument('--checkpoint', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--crops', type=int, nargs='+', required=True)
  ap.add_argument('--max-traces', type=int, default=None, help='smoke runs only')
  args = ap.parse_args()
  if not set(args.crops) <= set(distil.TRAIN_CROPS):
    raise ValueError(f'crops must be among {distil.TRAIN_CROPS}: crop 11 validates, 12 scores, 13-40 are never read')
  distil.device_settings(tf32=False)
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  teacher, _ = distil_teacher.load_export(args.export)
  ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
  student = distil.load_checkpoint(teacher, ckpt)
  for k in args.crops:
    if (args.out / f'crop-{k:02d}' / 'done.json').exists():
      print(f'crop {k}: done, skipped', flush=True)
      continue
    print(json.dumps(collect_crop(k, teacher, student, args.teacher_out, args.out, dev, args.max_traces)), flush=True)


if __name__ == '__main__':
  main()
