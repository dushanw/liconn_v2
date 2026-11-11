"""Crop 11: FFN in float32 and in fast FFN's format, 1, 8, 32 and 64 objects at once (docs/fp32_batch_design.md).

  time_widths.py --export DIR --out DIR [--runs F32x1 ...] [--max-objects N]

One B200. FFN (the torch port) traces the 512 crop-11 objects in each run of liconn_v2.fp32_batch.RUNS,
with the loading, warm-up, timing and scoring of examples/time_crop11.py. Writes OUT/widths.json after
every run, so a job stopped by its time limit keeps the runs it finished. Crop 11 only.
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
from liconn_v2 import fp32_batch  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_teacher  # pylint: disable=g-import-not-at-top,wrong-import-position
import time_crop11 as tc  # pylint: disable=g-import-not-at-top,wrong-import-position


def run(name, net, shape, seeds, gt, dev, copy_seconds) -> dict:
  """time_crop11.run with the settings of fp32_batch.RUNS[name]."""
  fmt, width, scale = fp32_batch.check_run(name)
  if width == 1:
    tracer = distil.TorchTracer(net, shape, dev)
    tracer.trace(seeds[0]['seed_zyx'], box_only=True)  # warm-up, not timed
    records, wall = [], 0.0
    for s in seeds:
      t0 = time.perf_counter()
      trace = tracer.trace(s['seed_zyx'], box_only=True)  # ends with a copy to the host
      w = time.perf_counter() - t0
      wall += w
      records.append(tc.record(trace, s, gt, w))
  else:
    tracer = distil.BatchTracer(net, shape, dev, width)
    tracer.trace_all([s['seed_zyx'] for s in seeds[:width]], box_only=True)  # warm-up
    t0 = tc.clock(dev)
    traces = tracer.trace_all([s['seed_zyx'] for s in seeds], box_only=True)
    wall = tc.clock(dev) - t0
    records = [tc.record(t, s, gt, None) for t, s in zip(traces, seeds)]
  steps = sum(r['steps'] for r in records)
  summary = {'network': 'ffn', 'format': fmt, 'width': width, 'stream_scale': scale, 'box_only': True,
             'objects': len(records), 'wall_seconds': wall, 'crop_seconds': wall + copy_seconds, 'steps': steps,
             'ms_per_step': 1000 * wall / steps, 'mean_f1': float(np.mean([r['f1'] for r in records])),
             'mean_recall': float(np.mean([r['recall'] for r in records])),
             'mean_precision': float(np.mean([r['precision'] for r in records])),
             'merges_ge_1000': int(sum(r['merges'] for r in records)),
             'merged_voxels': int(sum(r['merged_voxels'] for r in records)),
             'stopped_at_first_step': int(sum(r['steps'] == 1 for r in records)),
             'nonfinite_outputs': tc.nonfinite(net)}
  print(json.dumps({'run': name, **summary}), flush=True)
  return {'summary': summary, 'records': records}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--runs', nargs='+', default=list(fp32_batch.RUNS))
  ap.add_argument('--max-objects', type=int, default=None, help='smoke runs only')
  args = ap.parse_args()
  for r in args.runs:
    fp32_batch.check_run(r)
  args.out.mkdir(parents=True, exist_ok=True)
  k = distil.VALID_CROP
  distil.device_settings(tf32=False)
  dev = torch.device('cuda')

  teacher, manifest = distil_teacher.load_export(args.export)
  if manifest['teacher.pt'] != unseen.TEACHER_SHA256:
    raise ValueError(f"teacher export {manifest['teacher.pt']} is not the test's")
  image = reference.normalize(manyseed.load_raw_xyz(k))  # sha256 checked by manyseed
  gt = manyseed.load_proofread_zyx(k)
  seeds = vitseg.tuning_objects(gt)[:args.max_objects]
  out = {'design': 'docs/fp32_batch_design.md', 'design_sha256': fp32_batch.design_sha256(), 'crop': k,
         'objects': len(seeds), 'max_objects': args.max_objects, 'teacher_sha256': manifest['teacher.pt'],
         'device': torch.cuda.get_device_name(dev), 'torch': torch.__version__, 'tf32': False, 'runs': {}}
  teacher = teacher.to(dev).eval()
  torch.zeros(1, device=dev).add_(1)  # the timed copy below is not the job's first GPU operation
  t0 = tc.clock(dev)
  copy = torch.as_tensor(np.asarray(image, np.float32)).to(dev)
  out['ffn_image_copy_seconds'] = tc.clock(dev) - t0
  del copy
  for name in args.runs:
    fmt, _, scale = fp32_batch.RUNS[name]
    net = distil.TeacherNet(teacher, image, dev, fmt=fmt, scale=scale)
    out['runs'][name] = run(name, net, image.shape, seeds, gt, dev, out['ffn_image_copy_seconds'])
    del net
    torch.cuda.empty_cache()
    out['report'] = fp32_batch.widths_report(out['runs']) if args.max_objects is None else None
    tmp = args.out / '.widths.json.tmp'
    tmp.write_text(json.dumps(out))
    tmp.replace(args.out / 'widths.json')
  if out['report'] is not None:
    print(json.dumps({key: v for key, v in out['report'].items() if key != 'fingerprints'}, indent=1))


if __name__ == '__main__':
  main()
