"""Float32 FFN, 32 objects at once, on one test crop, one B200 (docs/fp32_batch_design.md, part 2).

  run_unseen_fp32x32.py --run-dir DIR --crop K --out DIR [--smoke N]

DIR is the MIT run folder distil-20261031 (export/). K is one of crops 41-50; --smoke N runs the same
code on the first N objects of crop 11 instead and is refused with a crop of 41-50. Loading, checks
and crop time are those of the test's B200 runs (examples/run_unseen.py): teacher and crop sha256s,
TF32 off, crop time = traces plus the image copy. Writes OUT/crop-K/F32x32.json when the run ends.
"""

import argparse
import json
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

from liconn_v2 import distil  # pylint: disable=g-import-not-at-top
from liconn_v2 import fp32_batch  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_teacher  # pylint: disable=g-import-not-at-top,wrong-import-position
import run_unseen as ru  # pylint: disable=g-import-not-at-top,wrong-import-position
import time_crop11 as tc  # pylint: disable=g-import-not-at-top,wrong-import-position
import time_widths as tw  # pylint: disable=g-import-not-at-top,wrong-import-position

NAME = fp32_batch.TEST_RUN


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run-dir', type=Path, required=True)
  ap.add_argument('--crop', type=int, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--smoke', type=int, default=None, help='first N objects of crop 11 only')
  args = ap.parse_args()
  raw, gt = ru.load_crop(args.crop, args.smoke)  # refuses any other crop before the GPU is touched
  folder = args.out / f'crop-{args.crop:02d}'
  folder.mkdir(parents=True, exist_ok=True)
  distil.device_settings(tf32=False)
  dev = torch.device('cuda')

  teacher, manifest = distil_teacher.load_export(args.run_dir / 'export')
  if manifest['teacher.pt'] != unseen.TEACHER_SHA256:
    raise ValueError(f"teacher export {manifest['teacher.pt']} is not the one fixed in the design")
  image = reference.normalize(raw)
  seeds = vitseg.tuning_objects(gt)[:args.smoke]
  out = {'design': 'docs/fp32_batch_design.md', 'design_sha256': fp32_batch.design_sha256(), 'crop': args.crop,
         'smoke_objects': args.smoke, 'object_count': len(seeds), 'object_ids': [s['object_id'] for s in seeds],
         'crop_sha256': ru.crop_sha256(args.crop, args.smoke), 'teacher_sha256': manifest['teacher.pt'],
         'device': torch.cuda.get_device_name(dev), 'host': socket.gethostname(), 'torch': torch.__version__,
         'tf32': False, 'run_name': NAME}
  teacher = teacher.to(dev).eval()
  torch.zeros(1, device=dev).add_(1)  # as in time_widths.py
  t0 = tc.clock(dev)
  copy = torch.as_tensor(np.asarray(image, np.float32)).to(dev)
  out['ffn_image_copy_seconds'] = tc.clock(dev) - t0
  del copy
  fmt, _, scale = fp32_batch.RUNS[NAME]
  net = distil.TeacherNet(teacher, image, dev, fmt=fmt, scale=scale)
  out['run'] = tw.run(NAME, net, image.shape, seeds, gt, dev, out['ffn_image_copy_seconds'])
  tmp = folder / f'.{NAME}.json.tmp'
  tmp.write_text(json.dumps(out))
  tmp.replace(folder / f'{NAME}.json')  # a task stopped while writing leaves no partial result file


if __name__ == '__main__':
  main()
