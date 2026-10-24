"""Fast tracer bitwise check on CPU (docs/fasttrace_design.md).

  check_fasttrace.py --cpu-notebook [--out DIR]

Traces the authors' notebook seed with liconn_v2.fasttrace.FastTracer on CPU and compares
every field of the fingerprint with the pinned CPU reference (liconn_v2/expected_reference.json).
Exits with an error on any difference.

Output: DIR/fasttrace_check.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import fasttrace  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--cpu-notebook', action='store_true', required=True)
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  raw = reference.normalize(data.load_raw_xyz())
  with reference.Tracer(data.checkpoint_path()) as tracer:
    t = fasttrace.FastTracer(tracer).trace(raw, reference.SEED_XYZ[::-1])
  got, expected = reference.fingerprint(t), reference.load_expected()
  differences = {k: {'fast': got[k], 'expected': expected.get(k)} for k in got if expected.get(k) != got[k]}
  result = {'design': 'docs/fasttrace_design.md', 'backend': t.backend, 'identical': not differences,
            'differences': differences, 'fingerprint': got,
            'ms_per_step': 1000 * t.trace_seconds / t.iterations if t.iterations else None}
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / 'fasttrace_check.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps({k: result[k] for k in ('identical', 'differences', 'ms_per_step')}, indent=2))
  if differences:
    raise SystemExit('fast tracer differs from the pinned reference')


if __name__ == '__main__':
  main()
