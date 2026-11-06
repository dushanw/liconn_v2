"""Factorised FFN, part 1: check the PyTorch port and the student initialisation (CPU).

  run_factorised_port.py --out DIR [--boxes 16]   # DIR/factorised_port.json

Inputs: network inputs recorded every --every steps of the notebook's reference trace (real
mid-trace inputs; the trace is checked against the recorded reference fingerprint), and boxes
of the notebook crop with masks built from the proofread labels (factorised.realistic_box).
See docs/factorised_design.md for the pre-registered checks.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import factorised as fz  # pylint: disable=g-import-not-at-top
from liconn_v2 import pruned  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top

TOL = 1e-3
THRESHOLD_LOGIT = float(np.log(reference.SEGMENT_THRESHOLD / (1 - reference.SEGMENT_THRESHOLD)))


def stats(a, b) -> dict:
  d = np.abs(a - b)
  return {'max_abs_logit_diff': float(d.max()), 'mean_abs_logit_diff': float(d.mean()),
          'threshold_flips': int(((a >= THRESHOLD_LOGIT) != (b >= THRESHOLD_LOGIT)).sum()),
          'voxels': int(d.size), 'pass': bool(d.max() < TOL)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--boxes', type=int, default=16)
  ap.add_argument('--every', type=int, default=24)
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  from clu import checkpoint  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top
  from connectomics.jax.models import convstack  # pylint: disable=g-import-not-at-top
  params = checkpoint.Checkpoint('').load_state(state=None, checkpoint=data.checkpoint_path())['params']
  model = convstack.ResConvStack(convstack.ConvstackConfig(depth=reference.DEPTH, padding='same',
                                                           use_layernorm=True))
  authors = jax.jit(lambda d: model.apply({'params': params}, d))
  teacher = fz.teacher_from_params(params)
  student = fz.student_from_teacher(teacher, dilations=(1,) * len(fz.DECODER_DILATIONS)).eval()

  image = reference.normalize(data.load_raw_xyz())
  with reference.Tracer(data.checkpoint_path()) as tracer:
    rec = pruned.InputRecorder(every=args.every)
    trace = tracer.trace(image, reference.SEED_XYZ[::-1], instrument=rec.attach)
  trace_matches_reference = reference.fingerprint(trace) == reference.load_expected()
  recorded = np.transpose(np.stack(rec.inputs), (0, 4, 1, 2, 3))  # (b, 2, z, y, x)
  labels = np.transpose(data.load_proofread_xyz(), [2, 1, 0])
  centers = fz.check_centers(image.shape, args.boxes)
  x = np.stack([fz.realistic_box(image, labels, c) for c in centers])
  unknown = x.copy()
  unknown[:, 1] = 0.0

  t0 = time.time()
  y_authors = np.concatenate([np.asarray(authors(np.transpose(x[i:i + 1], (0, 2, 3, 4, 1))))[..., 0]
                              for i in range(len(x))])
  t_jax = time.time() - t0
  with torch.no_grad():
    t0 = time.time()
    y_port = teacher(torch.from_numpy(x)).numpy()[:, 0]
    t_port = time.time() - t0
    y_port_unknown = teacher(torch.from_numpy(unknown)).numpy()[:, 0]
    y_student = student(torch.from_numpy(x)).numpy()[:, 0]

  y_rec_authors = np.concatenate([np.asarray(authors(np.transpose(recorded[i:i + 1], (0, 2, 3, 4, 1))))[..., 0]
                                  for i in range(len(recorded))])
  with torch.no_grad():
    y_rec_port = teacher(torch.from_numpy(recorded)).numpy()[:, 0]

  v = fz.flop_per_voxel()
  result = {
      'boxes': len(x),
      'centers_zyx': [list(c) for c in centers],
      'tolerance_max_abs_logit_diff': TOL,
      'reference_trace': {'steps': trace.iterations, 'recorded_steps': rec.steps,
                          'matches_recorded_reference': bool(trace_matches_reference)},
      'port_vs_authors_recorded_inputs': stats(y_rec_port, y_rec_authors),
      'port_vs_authors': stats(y_port, y_authors),
      'student_at_init_vs_teacher_with_unknown_mask': stats(y_student, y_port_unknown),
      # Not a check: how much the mask changes the teacher's output on these boxes.
      'teacher_mask_effect': {k: v for k, v in stats(y_port, y_port_unknown).items() if k != 'pass'},
      'cpu_seconds': {'authors_jax': t_jax, 'port_torch': t_port,
                      'note': 'CPU, one process; not indicative of GPU speed'},
      'flop_per_voxel': v,
      'flop_per_box_gflop': {k: val * int(np.prod(reference.FOV_SIZE)) / 1e9 for k, val in v.items()},
      'crop12_pflop': {f'tile_{t}': {k: (val / 1e15 if k != 'ratio' else val)
                                      for k, val in fz.crop_flop(137723, (500, 500, 500), tile=t).items()}
                       for t in (None, 250, 125)},
      'parameters': {'teacher': sum(p.numel() for p in teacher.parameters()),
                     'encoder': sum(p.numel() for p in student.encoder.parameters()),
                     'decoder': sum(p.numel() for p in student.decoder.parameters())},
      'encoder_radius_voxels': student.encoder.radius,
  }
  (args.out / 'factorised_port.json').write_text(json.dumps(result, indent=1))
  print(json.dumps({k: result[k] for k in ('reference_trace', 'port_vs_authors_recorded_inputs', 'port_vs_authors',
                                           'student_at_init_vs_teacher_with_unknown_mask',
                                           'crop12_pflop')}, indent=1))


if __name__ == '__main__':
  main()
