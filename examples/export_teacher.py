"""Factorised FFN, part 2: files the MIT jobs need from the local CPU setup (docs/distil_design.md).

  export_teacher.py --out DIR

DIR/teacher.pt       the ported teacher's state (distil.load_teacher), equal to the checkpoint port;
DIR/gpu_check.npz    the 15 network inputs recorded every 24th step of the notebook's reference
                     trace and the authors' JAX logits for them (CPU, float32);
DIR/export.json      sha256 of both files and the checks made here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import distil  # pylint: disable=g-import-not-at-top
from liconn_v2 import factorised as fz  # pylint: disable=g-import-not-at-top
from liconn_v2 import pruned  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top


def sha256(path) -> str:
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', type=Path, required=True)
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
  distil.save_teacher(teacher, args.out / 'teacher.pt')
  loaded = distil.load_teacher(args.out / 'teacher.pt')
  same_state = all(torch.equal(a, b) for a, b in zip(teacher.state_dict().values(), loaded.state_dict().values()))

  image = reference.normalize(data.load_raw_xyz())
  with reference.Tracer(data.checkpoint_path()) as tracer:
    rec = pruned.InputRecorder(every=24)
    trace = tracer.trace(image, reference.SEED_XYZ[::-1], instrument=rec.attach)
  inputs = np.transpose(np.stack(rec.inputs), (0, 4, 1, 2, 3))  # (b, 2, z, y, x)
  logits = np.concatenate([np.asarray(authors(np.transpose(inputs[i:i + 1], (0, 2, 3, 4, 1))))[..., 0]
                           for i in range(len(inputs))])
  np.savez(args.out / 'gpu_check.npz', inputs=inputs, authors_logits=logits, steps=np.array(rec.steps))
  with torch.no_grad():
    cpu_diff = float(np.abs(loaded(torch.from_numpy(inputs)).numpy()[:, 0] - logits).max())

  result = {
      'teacher.pt': sha256(args.out / 'teacher.pt'),
      'gpu_check.npz': sha256(args.out / 'gpu_check.npz'),
      'checkpoint': data.CHECKPOINT_PREFIX,
      'saved_state_equals_port': same_state,
      'reference_trace_matches_recorded': reference.fingerprint(trace) == reference.load_expected(),
      'recorded_inputs': len(inputs),
      'cpu_port_vs_authors_max_abs_logit_diff': cpu_diff,
      'notebook_raw_sha256': data.PINNED_SHA256[data.RAW_FILE],
  }
  (args.out / 'export.json').write_text(json.dumps(result, indent=1))
  print(json.dumps(result, indent=1))


if __name__ == '__main__':
  main()
