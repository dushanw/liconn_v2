"""Compares a candidate benchmark run with a base run, seed by seed.

  compare_runs.py BASE_DIR CANDIDATE_DIR [--out FILE]

Prints a per-seed table and writes the full comparison as JSON (default:
CANDIDATE_DIR/compare_to_base.json). Tolerance checks use the owner-confirmed
tolerance (all rules at 1%). Runs with different inputs or seeds are rejected;
seeds missing from either run are listed, and the comparison then covers only the
shared seeds.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liconn_v2 import compare  # pylint: disable=g-import-not-at-top


def main():
  p = argparse.ArgumentParser()
  p.add_argument('base')
  p.add_argument('candidate')
  p.add_argument('--out')
  a = p.parse_args()
  result = compare.compare_runs(a.base, a.candidate)
  out = Path(a.out) if a.out else Path(a.candidate) / 'compare_to_base.json'
  out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
  if not result['complete']:
    print('INCOMPLETE: missing in candidate', result['missing_in_candidate'],
          '| missing in base', result['missing_in_base'],
          '| tolerance below covers shared seeds only')
  if not result['inputs_sha256_checked']:
    print('note: input hashes not recorded in both runs; seed coordinates and objects were checked')
  print(f"{'seed':16s} {'IoU':>7s} {'steps':>11s} {'d_recall':>9s} {'d_prec':>8s} {'merges b/c':>12s} flag")
  for r in result['per_seed']:
    print(f"{r['name']:16s} {r['mask_iou']:7.4f} {r['base_steps']:5d}/{r['cand_steps']:<5d} "
          f"{r['delta_recall']:+9.4f} {r['delta_precision']:+8.4f} "
          f"{len(r['base_merges']):5d}/{len(r['cand_merges']):<6d} {'REVIEW' if r['flag_for_review'] else ''}"
          f"{'' if r['seed_tolerance']['within_seed_rules'] else ' OUTSIDE-SEED-RULES'}")
  for g, v in result['overall'].items():
    tol = v['tolerance']
    print(g, 'within tolerance:', tol['within_tolerance'] if tol else 'no seeds')
  print('PASSES TOLERANCE' if result['passes_tolerance'] else 'DOES NOT PASS (see above)')
  print('written', out)


if __name__ == '__main__':
  main()
