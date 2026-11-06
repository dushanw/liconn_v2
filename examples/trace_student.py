"""Factorised FFN, part 2: student traces on the validation crop 11 (docs/distil_design.md).

  trace_student.py --export DIR --teacher-out DIR --checkpoint best.pt --out DIR

The student (frozen encoder features of crop 11, float16 on the GPU; decoder in float32, TF32 off),
or a pruned FFN (train_pruned.py; image and mask boxes, float32), traces the same crop-11 seeds as
the teacher with distil.TorchTracer and the authors' movement
policy. Per object: benchmark.score against the proofread labels, and precision, recall and F1
against the teacher's mask. Trace times are a diagnostic, not the part-3 timing.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_teacher  # pylint: disable=g-import-not-at-top,wrong-import-position
import train_distil  # pylint: disable=g-import-not-at-top,wrong-import-position


def f1(p, r) -> float:
  return 2 * p * r / (p + r) if p + r else 0.0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--teacher-out', type=Path, required=True)
  ap.add_argument('--checkpoint', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  k = distil.VALID_CROP
  distil.device_settings(tf32=False)
  dev = torch.device('cuda')

  teacher, _ = distil_teacher.load_export(args.export)
  ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
  student = distil.load_checkpoint(teacher, ckpt)
  t0 = time.time()
  if isinstance(student, fz.ConvStack):  # pruned FFN: no encoder
    image = reference.normalize(manyseed.load_raw_xyz(k))  # sha256 checked by manyseed
    net = distil.TeacherNet(student, image, dev)
    encode_seconds = 0.0
  else:
    features, image = train_distil.encode(student.encoder.to(dev).eval(), k, dev)
    net = distil.StudentNet(student.decoder.to(dev), features)
    encode_seconds = time.time() - t0
  gt = manyseed.load_proofread_zyx(k)
  tdir = args.teacher_out / f'crop-{k:02d}'
  teacher_records = json.loads((tdir / 'traces.json').read_text())
  teacher_masks = np.load(tdir / 'masks.npz')

  tracer = distil.TorchTracer(net, image.shape, dev)
  tracer.trace(teacher_records[0]['seed_zyx'])  # warm-up, not recorded
  records = []
  for tr in teacher_records:
    t1 = time.perf_counter()
    trace = tracer.trace(tr['seed_zyx'])
    wall = time.perf_counter() - t1
    mask = benchmark.object_mask(trace.seed)
    flat = np.flatnonzero(mask)
    tm = teacher_masks[str(tr['trace'])]
    common = len(np.intersect1d(flat, tm, assume_unique=True))
    p_t = common / len(flat) if len(flat) else 0.0
    r_t = common / len(tm) if len(tm) else 0.0
    records.append({'trace': tr['trace'], 'object_id': tr['object_id'], 'seed_zyx': tr['seed_zyx'],
                    'steps': trace.iterations, 'trace_seconds': trace.trace_seconds, 'wall_seconds': wall,
                    'teacher_steps': tr['steps'], 'teacher_f1': f1(tr['precision'], tr['recall']),
                    'vs_teacher': {'precision': p_t, 'recall': r_t, 'f1': f1(p_t, r_t)},
                    **benchmark.score(mask, gt, tr['seed_zyx'])})
  s_f1 = np.array([f1(r['precision'], r['recall']) for r in records])
  t_f1 = np.array([r['teacher_f1'] for r in records])
  summary = {
      'checkpoint': str(args.checkpoint), 'checkpoint_step': ckpt['step'], 'dilations': ckpt['dilations'],
      'keep': ckpt.get('keep'),
      'objects': len(records),
      'student_mean_f1': float(s_f1.mean()), 'teacher_mean_f1': float(t_f1.mean()),
      'student_mean_recall': float(np.mean([r['recall'] for r in records])),
      'student_mean_precision': float(np.mean([r['precision'] for r in records])),
      'teacher_mean_recall': float(np.mean([r['recall'] for r in teacher_records])),
      'teacher_mean_precision': float(np.mean([r['precision'] for r in teacher_records])),
      'paired_f1_difference_quantiles': {q: float(np.quantile(s_f1 - t_f1, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
      'objects_student_higher': int((s_f1 > t_f1).sum()), 'objects_teacher_higher': int((s_f1 < t_f1).sum()),
      'mean_f1_vs_teacher_mask': float(np.mean([r['vs_teacher']['f1'] for r in records])),
      'stopped_at_first_step': int(sum(r['steps'] == 1 for r in records)),
      'teacher_stopped_at_first_step': int(sum(r['steps'] == 1 for r in teacher_records)),
      'mean_unlabeled_fraction': {'student': float(np.mean([r['unlabeled_fraction'] for r in records])),
                                  'teacher': float(np.mean([r['unlabeled_fraction'] for r in teacher_records]))},
      'merged_voxels': {'student': int(sum(r['merged_voxels'] for r in records)),
                        'teacher': int(sum(r['merged_voxels'] for r in teacher_records))},
      'merges_ge_1000': {'student': int(sum(len(r['merges']) for r in records)),
                         'teacher': int(sum(len(r['merges']) for r in teacher_records))},
      'steps': {'student': int(sum(r['steps'] for r in records)),
                'teacher': int(sum(r['steps'] for r in teacher_records))},
      'trace_seconds': {'student': float(sum(r['trace_seconds'] for r in records)),
                        'teacher_with_recording': float(sum(r['trace_seconds'] for r in teacher_records))},
      'encode_seconds': encode_seconds, 'device': torch.cuda.get_device_name(dev),
      'note': 'crop 11 is the validation crop; these numbers choose the decoder for part 3 and are not test results',
  }
  summary['mean_f1_gap_to_teacher'] = summary['teacher_mean_f1'] - summary['student_mean_f1']
  (args.out / 'student_crop11.json').write_text(json.dumps({'summary': summary, 'records': records}))
  print(json.dumps(summary, indent=1))


if __name__ == '__main__':
  main()
