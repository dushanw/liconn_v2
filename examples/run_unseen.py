"""One B200 run (FFN, the student, P8 or P4) on one unseen crop, one B200 (docs/unseen_design.md).

  run_unseen.py --run-dir DIR --crop K --run NAME --out DIR [--smoke N]

DIR is the MIT run folder distil-20261031 (export/, train-*/). K is one of crops 41-50; --smoke N
runs the same code on the first N objects of the validation crop 11 instead and is refused with a
crop of 41-50. NAME is one of S10, F1, F5, P2 and Q2, run with the settings and timing of
examples/time_crop11.py. Each run is its own Slurm task (cluster/mit/unseen.sbatch), so a crash or
time-out of one method cannot stop another. The checkpoint the run uses and the teacher export are
checked against the sha256s fixed in liconn_v2/unseen.py, and the design's sha256 is recorded.
Writes OUT/crop-K/NAME.json when the run ends (failed, for non-finite encoder features, or finished).
"""

import argparse
import json
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import distil  # pylint: disable=g-import-not-at-top
from liconn_v2 import factorised as fz  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_teacher  # pylint: disable=g-import-not-at-top,wrong-import-position
import time_crop11 as tc  # pylint: disable=g-import-not-at-top,wrong-import-position
import train_distil  # pylint: disable=g-import-not-at-top,wrong-import-position


def checked(run_dir: Path, name: str, record: dict) -> Path:
  """The checkpoint's path after its sha256 is checked against the design and recorded."""
  rel, expected = unseen.CHECKPOINTS[name]
  path = run_dir / rel
  actual = data.sha256_file(path)
  if actual != expected:
    raise ValueError(f'{path}: sha256 {actual}, expected {expected}')
  record[name] = {'path': rel, 'sha256': actual}
  return path


def crop_sha256(k: int, smoke: int | None) -> dict:
  """The pinned sha256s of the crop's files, which the loaders checked."""
  if smoke is None:
    return unseen._entry(k)['sha256']  # pylint: disable=protected-access
  return manyseed.load_manifest()['crops'][k]['sha256']


def load_crop(k: int, smoke: int | None):
  if smoke is None:
    unseen.check_crop(k)
    return unseen.load_raw_xyz(k), unseen.load_proofread_zyx(k)
  if k != distil.VALID_CROP:
    raise ValueError(f'smoke runs use crop {distil.VALID_CROP} only')
  return manyseed.load_raw_xyz(k), manyseed.load_proofread_zyx(k)


def check_run(name: str) -> None:
  """Refuses a run outside the design, or one whose settings in time_crop11 are not the design's."""
  if name not in unseen.B200_RUNS:
    raise ValueError(f'run {name} is not one of {unseen.B200_RUNS}')
  if (*tc.RUNS[name], tc.SCALE.get(name, 1.0)) != unseen.RUN_SETTINGS[name]:
    raise ValueError(f'time_crop11 settings of {name} are not the design\'s')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run-dir', type=Path, required=True)
  ap.add_argument('--crop', type=int, required=True)
  ap.add_argument('--run', required=True, choices=unseen.B200_RUNS)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--smoke', type=int, default=None, help='first N objects of crop 11 only')
  args = ap.parse_args()
  name = args.run
  check_run(name)
  raw, gt = load_crop(args.crop, args.smoke)  # refuses any other crop before the GPU is touched
  folder = args.out / f'crop-{args.crop:02d}'
  folder.mkdir(parents=True, exist_ok=True)
  distil.device_settings(tf32=False)
  dev = torch.device('cuda')

  teacher, manifest = distil_teacher.load_export(args.run_dir / 'export')
  if manifest['teacher.pt'] != unseen.TEACHER_SHA256:
    raise ValueError(f"teacher export {manifest['teacher.pt']} is not the one fixed in the design")
  checkpoints = {}
  image = reference.normalize(raw)
  seeds = vitseg.tuning_objects(gt)[:args.smoke]
  out = {'design': 'docs/unseen_design.md', 'design_sha256': unseen.design_sha256(), 'crop': args.crop,
         'smoke_objects': args.smoke, 'object_count': len(seeds), 'object_ids': [s['object_id'] for s in seeds],
         'crop_sha256': crop_sha256(args.crop, args.smoke), 'teacher_sha256': manifest['teacher.pt'],
         'checkpoints': checkpoints, 'device': torch.cuda.get_device_name(dev), 'host': socket.gethostname(),
         'torch': torch.__version__, 'tf32': False, 'run_name': name}

  if name == 'S10':
    ckpt = torch.load(checked(args.run_dir, 'student', checkpoints), map_location='cpu', weights_only=True)
    student = fz.student_from_teacher(teacher, dilations=tuple(ckpt['dilations']))
    student.decoder.load_state_dict(ckpt['decoder'])
    encoder = student.encoder.to(dev).eval()
    decoder = student.decoder.to(dev).eval()
    with torch.no_grad():
      encoder(torch.zeros((1, 1, 64, 64, 64), device=dev))  # warm-up, not timed
      t0 = tc.clock(dev)
      features = fz.encode_volume(encoder, image, tile=train_distil.TILE, dtype=torch.float16, out_device=dev)
      out['encoder_seconds'] = tc.clock(dev) - t0
    bad = int(sum(int((~torch.isfinite(c)).sum()) for c in features))
    out['encoder_non_finite'] = bad
    if bad:  # a failure of the student on this crop (the design)
      out['run'] = {'failed': f'{bad} non-finite encoder features in float16'}
    else:
      net = distil.StudentNet(decoder, features, dtype=torch.float16, fmt='fp16_head32', scale=tc.SCALE[name])
      out['run'] = finish(name, tc.run(name, net, image.shape, seeds, gt, dev), net, out['encoder_seconds'])
  else:
    teacher = teacher.to(dev).eval()
    t0 = tc.clock(dev)
    copy = torch.as_tensor(np.asarray(image, np.float32)).to(dev)
    out['ffn_image_copy_seconds'] = tc.clock(dev) - t0
    del copy
    network = tc.RUNS[name][0]
    if network in tc.PRUNED:
      c = torch.load(checked(args.run_dir, network, checkpoints), map_location='cpu', weights_only=True)
      net_module = distil.load_checkpoint(teacher, c).to(dev)
      out['pruned'] = {network: {'step': c['step'], 'keep': c['keep'], 'dilations': c['dilations']}}
    else:
      net_module = teacher
    net = distil.TeacherNet(net_module, image, dev, fmt=tc.RUNS[name][1], scale=tc.SCALE.get(name, 1.0))
    out['run'] = finish(name, tc.run(name, net, image.shape, seeds, gt, dev), net, out['ffn_image_copy_seconds'])
  tmp = folder / f'.{name}.json.tmp'
  tmp.write_text(json.dumps(out))
  tmp.replace(folder / f'{name}.json')  # a task stopped while writing leaves no partial result file
  print(json.dumps(out['run'].get('summary', out['run']), indent=1), flush=True)


def finish(name: str, result: dict, net, setup_seconds: float) -> dict:
  """The run with its non-finite count, stream scale and crop time (traces plus the encoder for the
  student, plus the image copy for FFN and the pruned FFNs), as in time_crop11.py."""
  s = result['summary']
  s['nonfinite_outputs'] = tc.nonfinite(net)
  s['stream_scale'] = tc.SCALE.get(name, 1.0)
  s['crop_seconds'] = s['wall_seconds'] + setup_seconds
  return result


if __name__ == '__main__':
  main()
