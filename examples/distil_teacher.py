"""Factorised FFN, part 2: teacher traces and distillation samples (docs/distil_design.md).

  distil_teacher.py --export DIR --out DIR --crops 0 1 2

For each crop (only 0-11): the large objects (vitseg.large_objects), each seeded at
benchmark.seed_point and traced by the torch teacher (float32, TF32 off) with distil.TorchTracer.
Writes OUT/crop-KK/: samples/*.npy (distil.SampleRecorder), traces.json (per trace: seed, steps,
time, benchmark.score against the proofread labels) and masks.npz (flat voxel indices of each
teacher mask). A crop with OUT/crop-KK/done.json is skipped.
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
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

ALLOWED_CROPS = distil.TRAIN_CROPS + (distil.VALID_CROP,)


def load_export(directory: Path):
  manifest = json.loads((directory / 'export.json').read_text())
  path = directory / 'teacher.pt'
  actual = hashlib.sha256(path.read_bytes()).hexdigest()
  if actual != manifest['teacher.pt']:
    raise ValueError(f'{path}: sha256 {actual}, expected {manifest["teacher.pt"]}')
  return distil.load_teacher(path), manifest


def trace_crop(k: int, teacher, dev, out: Path, max_traces=None) -> dict:
  image = reference.normalize(manyseed.load_raw_xyz(k))  # sha256 checked by manyseed
  gt = manyseed.load_proofread_zyx(k)
  seeds = vitseg.tuning_objects(gt)[:max_traces]
  tracer = distil.TorchTracer(distil.TeacherNet(teacher, image, dev), image.shape, dev)
  parts, records, masks = [], [], {}
  t_crop = time.perf_counter()
  for i, s in enumerate(seeds):
    rec = distil.SampleRecorder(k, i)
    t0 = time.perf_counter()
    trace = tracer.trace(s['seed_zyx'], on_step=rec)
    wall = time.perf_counter() - t0
    mask = benchmark.object_mask(trace.seed)
    parts.append(rec.arrays())
    masks[str(i)] = np.flatnonzero(mask).astype(np.int32)
    records.append({'trace': i, 'crop': k, 'object_id': s['object_id'], 'seed_zyx': s['seed_zyx'],
                    'steps': trace.iterations, 'samples': len(rec.steps), 'trace_seconds': trace.trace_seconds,
                    'wall_seconds': wall, **benchmark.score(mask, gt, s['seed_zyx'])})
    if i % 50 == 0:
      print(f'crop {k} trace {i}/{len(seeds)} steps {trace.iterations} {wall:.1f} s', flush=True)
  d = out / f'crop-{k:02d}'
  distil.save_samples(d / 'samples', distil.concat_samples(parts))
  np.savez(d / 'masks.npz', shape=np.array(gt.shape), **masks)
  (d / 'traces.json').write_text(json.dumps(records))
  steps = sum(r['steps'] for r in records)
  summary = {'crop': k, 'traces': len(records), 'steps': steps,
             'samples': sum(r['samples'] for r in records),
             'trace_seconds': sum(r['trace_seconds'] for r in records),
             'crop_seconds': time.perf_counter() - t_crop,
             'ms_per_step': 1000 * sum(r['trace_seconds'] for r in records) / max(1, steps),
             'mean_f1': vitseg.mean_f1(records),
             'device': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}
  (d / 'done.json').write_text(json.dumps(summary, indent=1))
  return summary


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--crops', type=int, nargs='+', required=True)
  ap.add_argument('--max-traces', type=int, default=None, help='smoke runs only')
  args = ap.parse_args()
  if not set(args.crops) <= set(ALLOWED_CROPS):
    raise ValueError(f'crops must be among {ALLOWED_CROPS}; crop 12 is for scoring and 13-40 are never read')
  distil.device_settings(tf32=False)
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  teacher, _ = load_export(args.export)
  for k in args.crops:
    if (args.out / f'crop-{k:02d}' / 'done.json').exists():
      print(f'crop {k}: done, skipped', flush=True)
      continue
    print(json.dumps(trace_crop(k, teacher, dev, args.out, args.max_traces)), flush=True)


if __name__ == '__main__':
  main()
