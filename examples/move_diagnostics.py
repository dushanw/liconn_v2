"""Factorised FFN, part 2b: face moves of student and teacher on stored samples (docs/distil_design.md).

  move_diagnostics.py --export DIR --checkpoint best.pt --samples DIR --crops 11 --out moves.json

For each sample (mask input and teacher output, stored by distil_teacher.py or distil_onpolicy.py),
the student's output is recomputed from the checkpoint, FFN's canvas rule is applied to both
outputs (distil.written), and distil.MoveGaps compares the face moves by step bin. On teacher
samples (no drift) a static offset of the student's face maxima shows in
teacher_minus_student; on on-policy samples the same numbers include drift. Crops 0-11 only.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_teacher  # pylint: disable=g-import-not-at-top,wrong-import-position
import train_distil  # pylint: disable=g-import-not-at-top,wrong-import-position

BATCH = 64


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--checkpoint', type=Path, required=True)
  ap.add_argument('--samples', type=Path, required=True, help='a directory with crop-KK/samples')
  ap.add_argument('--crops', type=int, nargs='+', required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--max-samples', type=int, default=None, help='per crop, evenly spaced')
  args = ap.parse_args()
  if not set(args.crops) <= set(distil.TRAIN_CROPS + (distil.VALID_CROP,)):
    raise ValueError('crops must be among 0-11: crop 12 scores, 13-40 are never read')
  distil.device_settings(tf32=False)
  dev = torch.device('cuda')
  teacher, _ = distil_teacher.load_export(args.export)
  ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
  student = fz.student_from_teacher(teacher, dilations=tuple(ckpt['dilations']))
  student.decoder.load_state_dict(ckpt['decoder'])
  decoder = student.decoder.to(dev).eval()
  gaps, per_crop = distil.MoveGaps(), {}
  t0 = time.time()
  for k in args.crops:
    features, _ = train_distil.encode(student.encoder.to(dev).eval(), k, dev)
    s = distil.load_samples(args.samples / f'crop-{k:02d}' / 'samples', mmap='r')
    n = len(s['steps'])
    idx = np.arange(n) if not args.max_samples or n <= args.max_samples else \
        np.unique(np.linspace(0, n - 1, args.max_samples).astype(int))
    crop_gaps = distil.MoveGaps()
    with torch.no_grad():
      for i in range(0, len(idx), BATCH):
        j = idx[i:i + BATCH]
        f = distil.gather_features({k: features}, np.full(len(j), k), s['centers'][j]).float()
        m = torch.from_numpy(np.asarray(s['masks'][j])).to(dev).float()
        tl = torch.from_numpy(np.asarray(s['targets'][j])).to(dev).float()
        sl = decoder(f, m[:, None])[:, 0]
        for g in (gaps, crop_gaps):
          g.add(sl, tl, m, s['steps'][j])
    per_crop[k] = {'samples': len(idx), **crop_gaps.summary()}
    print(json.dumps({'crop': k, 'samples': len(idx), 'seconds': time.time() - t0}), flush=True)
    del features
  result = {'checkpoint': str(args.checkpoint), 'checkpoint_step': ckpt['step'], 'samples_dir': str(args.samples),
            'crops': args.crops, 'max_samples': args.max_samples, 'all': gaps.summary(), 'per_crop': per_crop,
            'device': torch.cuda.get_device_name(dev)}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=1))
  print(json.dumps(result['all'], indent=1))


if __name__ == '__main__':
  main()
