"""Runs the authors' LICONN FFN example on CPU and writes the reference outputs.

Outputs (in ./outputs):
  reference_trace.json  fingerprint, comparison with the committed reference,
                        timing profile and package versions
  reference_mask.npz    the notebook's mask, label(cv.seed > 0)[0] == 1, cropped
                        to its bounding box (offset_zyx gives its position)
"""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import reference  # pylint: disable=g-import-not-at-top


def main():
  out = Path('outputs')
  out.mkdir(exist_ok=True)
  trace, fp = reference.run_reference()
  expected = reference.load_expected()
  mismatches = sorted(k for k in expected if fp.get(k) != expected[k])

  mask = reference.notebook_mask(trace.seed)
  reference.save_mask_npz(out / 'reference_mask.npz', mask)
  if not np.array_equal(reference.load_mask_npz(out / 'reference_mask.npz'), mask):
    raise SystemExit('reference_mask.npz does not reconstruct the notebook mask')

  versions = {}
  for name in ('ffn', 'connectomics', 'jax', 'jaxlib', 'flax', 'tensorflow', 'numpy', 'scipy', 'clu'):
    versions[name] = importlib.metadata.version(name)
  result = {
      'fingerprint': fp,
      'matches_committed_reference': not mismatches,
      'mismatched_fields': mismatches,
      'timing': {
          'compile_seconds': round(trace.compile_seconds, 3),
          'trace_seconds': round(trace.trace_seconds, 3),
          'seconds_per_step': round(trace.trace_seconds / trace.iterations, 4),
          'counters': trace.counters,
      },
      'platform': {'jax_backend': trace.backend,
                   'machine': platform.machine(), 'processor': platform.processor(),
                   'cpu_count': os.cpu_count(), 'python': sys.version.split()[0]},
      'versions': versions,
  }
  (out / 'reference_trace.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))
  if mismatches:
    raise SystemExit('Trace differs from the committed reference: ' + ', '.join(mismatches))


if __name__ == '__main__':
  main()
