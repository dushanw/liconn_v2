"""Factorised FFN, part 3: FFN and the student on the 457 crop-12 objects (docs/distil_design.md).

  score_crop12.py --export DIR --checkpoint best.pt --out DIR [--max-objects N]

One GPU, FFN first. Both networks trace with distil.TorchTracer (the authors' movement policy and
thresholds) in float32 with TF32 off; the student's encoder features are float16. Untimed: loading
the models and the crop, one warm-up trace per network and one encoder pass on a small tile.
Timed: FFN's image copy to the GPU and the student's encoding of the crop, then each trace's wall
time. Scores, crop times and the pilot signal come from distil.crop_summary. --max-objects is for smoke runs only; the signal is then
not meaningful.
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


def clock(dev) -> float:
  torch.cuda.synchronize(dev)
  return time.perf_counter()


def trace_all(name, tracer, seeds, gt, out: Path) -> list:
  tracer.trace(seeds[0]['seed_zyx'])  # warm-up, not timed
  records, masks = [], {}
  for i, s in enumerate(seeds):
    t0 = time.perf_counter()
    trace = tracer.trace(s['seed_zyx'])
    wall = time.perf_counter() - t0  # the tracer synchronises: its canvas is copied to the host
    mask = benchmark.object_mask(trace.seed)
    masks[str(s['object_id'])] = np.flatnonzero(mask).astype(np.int32)
    records.append({'object_id': s['object_id'], 'seed_zyx': s['seed_zyx'], 'steps': trace.iterations,
                    'trace_seconds': trace.trace_seconds, 'wall_seconds': wall,
                    **benchmark.score(mask, gt, s['seed_zyx'])})
    if i % 50 == 0:
      print(f'{name} object {i}/{len(seeds)} steps {trace.iterations} {wall:.2f} s', flush=True)
  np.savez_compressed(out / f'{name}_masks.npz', shape=np.array(gt.shape), **masks)
  return records


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--checkpoint', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--max-objects', type=int, default=None, help='smoke runs only')
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  k = distil.SCORE_CROP
  distil.device_settings(tf32=False)
  dev = torch.device('cuda')

  teacher, manifest = distil_teacher.load_export(args.export)
  ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
  student = fz.student_from_teacher(teacher, dilations=tuple(ckpt['dilations']))
  student.decoder.load_state_dict(ckpt['decoder'])
  image = reference.normalize(manyseed.load_raw_xyz(k))  # sha256 checked by manyseed
  gt = manyseed.load_proofread_zyx(k)
  seeds = vitseg.tuning_objects(gt)[:args.max_objects]

  # FFN: the image goes to the GPU once, then every object is traced.
  teacher = teacher.to(dev).eval()
  t0 = clock(dev)
  ffn_net = distil.TeacherNet(teacher, image, dev)
  ffn_setup = clock(dev) - t0
  ffn = trace_all('ffn', distil.TorchTracer(ffn_net, image.shape, dev), seeds, gt, args.out)
  del ffn_net
  torch.cuda.empty_cache()

  # Student: the encoder runs once over the crop, then every object is traced.
  encoder = student.encoder.to(dev).eval()
  decoder = student.decoder.to(dev).eval()
  with torch.no_grad():  # warm-up of the encoder's kernels on one small tile, not timed
    encoder(torch.zeros((1, 1, 64, 64, 64), device=dev))
  t0 = clock(dev)
  with torch.no_grad():
    features = fz.encode_volume(encoder, image, tile=train_distil.TILE, dtype=torch.float16).to(dev)
  encode_seconds = clock(dev) - t0
  bad = sum(int((~torch.isfinite(c)).sum()) for c in features)
  if bad:
    sys.exit(f'crop {k}: {bad} encoder features are not finite in float16')
  st = trace_all('student', distil.TorchTracer(distil.StudentNet(decoder, features), image.shape, dev),
                 seeds, gt, args.out)

  summary = distil.crop_summary(ffn, st, ffn_setup_seconds=ffn_setup, encode_seconds=encode_seconds)
  summary.update({'crop': k, 'checkpoint': str(args.checkpoint), 'checkpoint_step': ckpt['step'],
                  'dilations': ckpt['dilations'], 'teacher_sha256': manifest['teacher.pt'],
                  'max_objects': args.max_objects, 'device': torch.cuda.get_device_name(dev),
                  'torch': torch.__version__, 'tf32': False, 'encoder_features': 'float16',
                  'note': 'pilot signal on crop 12 (a tuning crop); not a test result. Crops 13-40 not read.'})
  (args.out / 'crop12.json').write_text(json.dumps({'summary': summary, 'ffn': ffn, 'student': st}))
  print(json.dumps(summary, indent=1))


if __name__ == '__main__':
  main()
