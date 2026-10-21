"""Combines per-seed held-out runs (examples/run_heldout.py) into one run.

  merge_heldout.py OUT_DIR PART_DIR [PART_DIR ...]
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import compare  # pylint: disable=g-import-not-at-top
from liconn_v2 import heldout  # pylint: disable=g-import-not-at-top


def main():
  if len(sys.argv) < 3:
    raise SystemExit(__doc__)
  gt = np.ascontiguousarray(np.transpose(heldout.load_proofread_xyz(), [2, 1, 0]))
  names = [s['name'] for s in heldout.select_seeds(gt)]
  result = compare.merge_runs(sys.argv[2:], sys.argv[1], names)
  print(json.dumps(result['summary']['all'], indent=2))


if __name__ == '__main__':
  main()
