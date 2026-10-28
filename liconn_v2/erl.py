"""Expected run length (ERL), adapted to single-seed FFN traces (docs/fresh_design.md).

Each seed's proofread object (within its crop) is skeletonised with kimimaro (TEASAR).
The runs of a trace are the connected pieces of that skeleton whose edges have both ends
inside the trace mask. Seed ERL = sum of squared run lengths / skeleton length, 0 if the
trace merges. Overall ERL is length-weighted over seeds.
"""

import numpy as np

from liconn_v2 import benchmark

ANISOTROPY_XYZ = (18, 18, 24)  # nm per voxel
TEASAR = {'scale': 1.5, 'const': 300, 'pdrf_scale': 100000, 'pdrf_exponent': 4}
DUST_THRESHOLD = 1000
GATE_RATIO = 3.0  # skeleton length at most this many bounding-box diagonals (pre-registered; owner kept 3x) ...
GATE_FRACTION = 0.95  # ... for at least this share of objects


def skeleton(obj_xyz: np.ndarray, origin_xyz=(0, 0, 0)) -> dict:
  """kimimaro skeleton of one binary object (x, y, z order), pooled over pieces.

  Returns voxel coordinates (x, y, z, in the frame of `origin_xyz`), edges, edge lengths
  in micrometres, the total length and the object's bounding-box diagonal in micrometres.
  """
  import kimimaro  # pylint: disable=g-import-not-at-top
  sks = kimimaro.skeletonize(obj_xyz.astype(np.uint8), teasar_params=TEASAR, anisotropy=ANISOTROPY_XYZ,
                             dust_threshold=DUST_THRESHOLD, fix_branching=True, fix_borders=True,
                             progress=False, parallel=1)
  an = np.array(ANISOTROPY_XYZ, np.float64)
  verts, edges, offset = [], [], 0
  for k in sorted(sks):
    sk = sks[k]
    verts.append(np.asarray(sk.vertices, np.float64))
    edges.append(np.asarray(sk.edges, np.int64) + offset)
    offset += len(sk.vertices)
  v = np.concatenate(verts) if verts else np.zeros((0, 3))
  e = np.concatenate(edges) if edges else np.zeros((0, 2), np.int64)
  lengths = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1) / 1000.0 if len(e) else np.zeros(0)
  vox = np.rint(v / an).astype(np.int64) + np.asarray(origin_xyz, np.int64)
  nz = np.argwhere(obj_xyz)
  diag = float(np.linalg.norm((nz.max(0) - nz.min(0) + 1) * an) / 1000.0) if len(nz) else 0.0
  return {'voxels_xyz': vox, 'edges': e, 'lengths_um': lengths, 'length_um': float(lengths.sum()),
          'bbox_diagonal_um': diag}


def run_lengths(covered: np.ndarray, edges: np.ndarray, lengths: np.ndarray) -> list:
  """Lengths of the connected pieces formed by edges whose two ends are covered."""
  if not len(edges):
    return []
  keep = covered[edges[:, 0]] & covered[edges[:, 1]]
  parent = {}

  def find(a):
    parent.setdefault(a, a)
    while parent[a] != a:
      parent[a] = parent[parent[a]]
      a = parent[a]
    return a

  for a, b in edges[keep]:
    ra, rb = find(int(a)), find(int(b))
    if ra != rb:
      parent[ra] = rb
  total = {}
  for (a, _), l in zip(edges[keep], lengths[keep]):
    r = find(int(a))
    total[r] = total.get(r, 0.0) + float(l)
  return sorted(total.values(), reverse=True)


def seed_terms(skel: dict, mask_zyx: np.ndarray, merged: bool) -> dict:
  """One seed's contribution: its skeleton length and squared runs (0 if merged)."""
  v = skel['voxels_xyz']
  inside = np.all((v >= 0) & (v < np.array(mask_zyx.shape[::-1])), axis=1) if len(v) else np.zeros(0, bool)
  covered = np.zeros(len(v), bool)
  covered[inside] = mask_zyx[v[inside, 2], v[inside, 1], v[inside, 0]]
  runs = run_lengths(covered, skel['edges'], skel['lengths_um'])
  sq = float(sum(r * r for r in runs))
  return {'skeleton_um': skel['length_um'], 'sum_sq_um2': 0.0 if merged else sq, 'sum_sq_no_merge_um2': sq,
          'longest_run_um': runs[0] if runs else 0.0, 'runs': len(runs), 'merged': bool(merged)}


def merged(record: dict) -> bool:
  """The project's merge definition: another object covered by at least MERGE_MIN_VOXELS."""
  return any(v >= benchmark.MERGE_MIN_VOXELS for _, v in record['merges'])


def summarize(terms: list) -> dict:
  """Length-weighted ERL over seeds (and by class), with and without the merge penalty."""
  def erl(ts, key):
    total = sum(t['skeleton_um'] for t in ts)
    return sum(t[key] for t in ts) / total if total else None

  out = {'seeds': len(terms), 'erl_um': erl(terms, 'sum_sq_um2'),
         'erl_no_merge_penalty_um': erl(terms, 'sum_sq_no_merge_um2'),
         'seed_mean_erl_um': float(np.mean([t['sum_sq_um2'] / t['skeleton_um'] for t in terms if t['skeleton_um']]))
         if terms else None,
         'merged_seeds': sum(t['merged'] for t in terms)}
  for cls in ('axon', 'dendrite'):
    ts = [t for t in terms if t.get('class') == cls]
    out[f'erl_um_{cls}'] = erl(ts, 'sum_sq_um2')
    out[f'erl_no_merge_penalty_um_{cls}'] = erl(ts, 'sum_sq_no_merge_um2')
  out['empty_skeletons'] = sum(1 for t in terms if not t['skeleton_um'])
  return out


def gate(skeletons: list) -> dict:
  """Sanity gate: enough skeletons no longer than GATE_RATIO bounding-box diagonals."""
  ok = [0 < s['length_um'] <= GATE_RATIO * s['bbox_diagonal_um'] for s in skeletons]  # empty fails
  frac = sum(ok) / len(ok) if ok else 0.0
  ratios = [s['length_um'] / s['bbox_diagonal_um'] for s in skeletons if s['bbox_diagonal_um']]
  return {'objects': len(ok), 'within_ratio': sum(ok), 'fraction': frac, 'passes': frac >= GATE_FRACTION,
          'empty': sum(1 for s in skeletons if not s['length_um']),
          'median_ratio': float(np.median(ratios)) if ratios else None,
          'max_ratio': float(max(ratios)) if ratios else None}
