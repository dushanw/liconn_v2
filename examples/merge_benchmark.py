"""Combines per-seed benchmark runs (e.g. one per cluster job) into one run.

  merge_benchmark.py OUT_DIR PART_DIR [PART_DIR ...]

Writes OUT_DIR/benchmark.json and OUT_DIR/benchmark_masks.npz in the same format
as a single run. All 12 benchmark seeds must be present exactly once.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
from liconn_v2 import compare  # pylint: disable=g-import-not-at-top
from liconn_v2 import data  # pylint: disable=g-import-not-at-top


def main():
  if len(sys.argv) < 3:
    raise SystemExit(__doc__)
  out_dir, parts = sys.argv[1], sys.argv[2:]
  gt = np.ascontiguousarray(np.transpose(data.load_proofread_xyz(), [2, 1, 0]))
  names = [s['name'] for s in benchmark.select_seeds(gt)]
  result = compare.merge_runs(parts, out_dir, names)
  print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
  main()
