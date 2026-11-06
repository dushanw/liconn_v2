"""Factorised FFN, part 2: the torch teacher's numerics on the GPU (docs/distil_design.md).

  check_gpu_port.py --export DIR --raw raw_xyz_uint8.npy --out DIR

Float32 with TF32 off. (1) The teacher on the 15 recorded reference-trace inputs against the
authors' JAX logits recorded on the CPU (pre-registered: max abs difference < 1e-3). (2) The
torch tracer with the teacher traces the notebook's seed on the notebook crop; the step count,
history, visited region (predicted_mask_sha256 hashes the non-NaN canvas) and the thresholded
masks (seed component and notebook mask) are compared with the recorded reference (reported).
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

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import distil  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top

TOL = 1e-3


def checked(path: Path, expected: str) -> Path:
  actual = hashlib.sha256(path.read_bytes()).hexdigest()
  if actual != expected:
    raise ValueError(f'{path}: sha256 {actual}, expected {expected}')
  return path


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--export', type=Path, required=True)
  ap.add_argument('--raw', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  distil.device_settings(tf32=False)
  dev = torch.device('cuda')
  manifest = json.loads((args.export / 'export.json').read_text())
  teacher = distil.load_teacher(checked(args.export / 'teacher.pt', manifest['teacher.pt'])).to(dev)
  rec = np.load(checked(args.export / 'gpu_check.npz', manifest['gpu_check.npz']))
  with torch.no_grad():
    y = torch.cat([teacher(torch.from_numpy(x[None]).to(dev)) for x in rec['inputs']])[:, 0].cpu().numpy()
  d = np.abs(y - rec['authors_logits'])
  t = np.float32(distil.OPTIONS.segment_threshold)

  raw = np.load(checked(args.raw, data.PINNED_SHA256[data.RAW_FILE]))
  image = reference.normalize(raw)
  tracer = distil.TorchTracer(distil.TeacherNet(teacher, image, dev), image.shape, dev)
  tracer.trace(reference.SEED_XYZ[::-1])  # warm-up (cuDNN algorithm choice)
  t0 = time.perf_counter()
  trace = tracer.trace(reference.SEED_XYZ[::-1])
  wall = time.perf_counter() - t0
  got, expected = reference.fingerprint(trace), reference.load_expected()
  keys = ('iterations', 'history_sha256', 'predicted_voxels', 'predicted_mask_sha256', 'seed_component_voxels',
          'seed_component_sha256', 'notebook_mask_voxels', 'notebook_mask_sha256')
  result = {
      'device': torch.cuda.get_device_name(dev), 'torch': torch.__version__,
      'cudnn': torch.backends.cudnn.version(), 'tf32': False,
      'recorded_inputs': {'max_abs_logit_diff': float(d.max()), 'mean_abs_logit_diff': float(d.mean()),
                          'threshold_flips': int(((y >= t) != (rec['authors_logits'] >= t)).sum()),
                          'voxels': int(d.size), 'pass': bool(d.max() < TOL)},
      'reference_trace': {k: {'gpu': got[k], 'recorded': expected[k], 'same': got[k] == expected[k]} for k in keys},
      'trace_seconds': trace.trace_seconds, 'trace_wall_seconds': wall,
      'ms_per_step': 1000 * trace.trace_seconds / trace.iterations,
  }
  (args.out / 'gpu_port.json').write_text(json.dumps(result, indent=1))
  print(json.dumps(result, indent=1))
  if not result['recorded_inputs']['pass']:
    sys.exit('the teacher on the GPU misses the pre-registered tolerance; later jobs must not run')


if __name__ == '__main__':
  main()
