"""SegNeuron zero-shot on the tuning crop (docs/segneuron_design.md): input resampling, the
mapping of x-y upsampled outputs back to the crop grid, our 7-channel layout, and scoring.

SegNeuron (github.com/yanchaoz/SegNeuron) itself runs unmodified in its own environments;
this module needs only numpy and scipy and reads its outputs: affinities (3, z, y, x) for
pairs (v, v - o), o = (1,0,0), (0,1,0), (0,0,1) as in vitseg.OFFSETS[:3], and a foreground
probability (z, y, x) trained against label != 0.
"""

import numpy as np

from liconn_v2 import vitseg

RESOLUTIONS = ('native', 'xy2')
BETAS = (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9)  # FRMC multicut bias; 0.25 is the authors' default
FOREGROUND = vitseg.NEURON  # foreground >= 0.5: a neuron voxel, as for the ViT's neuron channel
LONG_RANGE_FILL = 0.5  # our channels 3-5, which SegNeuron does not predict and segmentation never reads
MIN_PIECE = 0.01  # pieces holding at least 1% of an object
DIAGNOSTIC_CUTS = (0.3, 0.5, 0.7)  # foreground cuts reported (0.5 is the one used)
HIST_BINS = 1000
QUANTILES = (0.1, 0.5, 0.9)


def upsample_xy2(raw_zyx: np.ndarray) -> np.ndarray:
  """y and x upsampled 2x by linear interpolation, pixels treated as areas (fine voxels 2i
  and 2i + 1 lie within coarse voxel i), rounded back to uint8 for SegNeuron's input."""
  from scipy import ndimage  # pylint: disable=g-import-not-at-top
  if raw_zyx.dtype != np.uint8 or raw_zyx.ndim != 3:
    raise ValueError('expected a uint8 (z, y, x) volume')
  fine = ndimage.zoom(raw_zyx.astype(np.float32), (1, 2, 2), order=1, grid_mode=True, mode='nearest')
  return np.clip(np.rint(fine), 0, 255).astype(np.uint8)


def to_crop_grid(aff_fine: np.ndarray, fg_fine: np.ndarray):
  """(affinities (3, z, y, x), foreground (z, y, x)) on the crop grid from outputs on the
  x-y upsampled grid (3, z, 2y, 2x) and (z, 2y, 2x). Foreground: mean of each 1x2x2 block.
  z affinity of v: mean of the 4 fine z affinities in v's block. y affinity of v (pair v, v - y),
  along each of the 2 fine columns of the block: the minimum of the three fine y edges between
  the centres of v - y and v (inside v - y's block, crossing into v's block, inside v's block),
  so a boundary predicted inside either block is kept; then the mean over the 2 columns. x
  likewise. At i = 0 (or j = 0) the pair leaves the crop; nothing reads that value as an edge."""
  c, z, y2, x2 = aff_fine.shape
  if c != 3 or y2 % 2 or x2 % 2 or fg_fine.shape != (z, y2, x2):
    raise ValueError('expected (3, z, 2y, 2x) affinities and a (z, 2y, 2x) foreground')
  y, x = y2 // 2, x2 // 2
  blocks = lambda a: a.reshape(z, y, 2, x, 2)
  aff = np.empty((3, z, y, x), np.float32)
  aff[0] = blocks(aff_fine[0]).mean(axis=(2, 4))
  for ch, axis in ((1, 1), (2, 2)):
    f = np.moveaxis(aff_fine[ch], axis, 0)  # (2n, ...): the fine edges along this axis
    inside = f[1::2]  # fine edge (2i + 1, 2i), inside coarse block i
    crossing = f[0::2]  # fine edge (2i, 2i - 1), from block i - 1 into block i
    low = np.minimum(crossing, inside)
    low[1:] = np.minimum(low[1:], inside[:-1])  # fine edge (2i - 1, 2i - 2), inside block i - 1
    low = np.moveaxis(low, 0, axis)  # (z, y, 2x) or (z, 2y, x)
    aff[ch] = _pair_mean(low, ch, z, y, x)
  return aff, blocks(fg_fine).mean(axis=(2, 4)).astype(np.float32)


def _pair_mean(low: np.ndarray, ch: int, z: int, y: int, x: int) -> np.ndarray:
  """Mean over the 2 parallel fine lines of a coarse voxel: x pairs for the y channel
  (low is (z, y, 2x)), y pairs for the x channel (low is (z, 2y, x))."""
  if ch == 1:
    return low.reshape(z, y, x, 2).mean(axis=-1)
  return low.reshape(z, y, 2, x).mean(axis=2)


def vitseg_layout(aff: np.ndarray, fg: np.ndarray) -> np.ndarray:
  """Our (z, y, x, 7) layout: SegNeuron's z, y, x affinities, LONG_RANGE_FILL for the three
  long-range channels, and the foreground as the neuron channel."""
  out = np.full(fg.shape + (vitseg.CHANNELS,), LONG_RANGE_FILL, np.float32)
  out[..., :vitseg.NEAREST] = np.moveaxis(aff, 0, -1)
  out[..., vitseg.NEURON_CHANNEL] = fg
  return out


def mask_background(seg: np.ndarray, fg: np.ndarray) -> np.ndarray:
  """The segmentation with voxels of foreground < FOREGROUND set to 0 (background)."""
  return np.where(fg >= FOREGROUND, seg, 0).astype(seg.dtype)


def ours(aff: np.ndarray, fg: np.ndarray, gt: np.ndarray, objects: list) -> dict:
  """Post-processing A: vitseg fragments and agglomeration, threshold chosen on these objects."""
  import time  # pylint: disable=g-import-not-at-top
  a = vitseg_layout(aff, fg)
  t = time.perf_counter()
  frag = vitseg.fragments(a)
  ws = time.perf_counter() - t
  t = time.perf_counter()
  merges = vitseg.agglomerate(*vitseg.fragment_edges(frag, a))
  ag = time.perf_counter() - t
  threshold, table = vitseg.choose_threshold(frag, merges, gt, objects)
  seg = vitseg.segmentation_at(frag, merges, threshold)
  return {'threshold': threshold, 'table': table, 'fragments': int(frag.max()),
          'watershed_seconds': ws, 'agglomeration_seconds': ag, 'segmentation': seg}


def object_accounting(seg: np.ndarray, gt: np.ndarray, objects: list) -> dict:
  """Per-object means over the objects: recall, precision, F1 (vitseg.seed_records), the
  fraction of each object called background (seg 0) or in other segments, and the number of
  segments holding at least MIN_PIECE of it; plus vitseg.background_scores."""
  recs = vitseg.seed_records(seg, gt, objects)
  sid, gid, cnt = vitseg.overlap_table(seg, gt)
  per_obj = {}
  for s, g, c in zip(sid.tolist(), gid.tolist(), cnt.tolist()):
    per_obj.setdefault(g, {})[s] = c
  rows = []
  for r in recs:
    parts = per_obj[r['object_id']]
    size = r['object_voxels']
    background = parts.get(0, 0) / size
    pieces = sum(1 for s, c in parts.items() if s and c >= MIN_PIECE * size)
    f1 = 2 * r['precision'] * r['recall'] / (r['precision'] + r['recall']) if r['precision'] + r['recall'] else 0.0
    rows.append({'object_id': r['object_id'], 'recall': r['recall'], 'precision': r['precision'], 'f1': f1,
                 'lost_to_background': background, 'lost_to_other_segments': 1.0 - r['recall'] - background,
                 'pieces_over_1pct': pieces, 'seed_in_background': r['mask_voxels'] == 0})
  mean = {k: float(np.mean([row[k] for row in rows])) for k in
          ('recall', 'precision', 'f1', 'lost_to_background', 'lost_to_other_segments', 'pieces_over_1pct')}
  mean['seeds_in_background'] = int(sum(row['seed_in_background'] for row in rows))
  return {'per_object_mean': mean, 'median_pieces': float(np.median([r['pieces_over_1pct'] for r in rows])),
          'background': vitseg.background_scores(seg, gt), 'objects': rows}


def _hist(values: np.ndarray) -> np.ndarray:
  return np.histogram(values, bins=HIST_BINS, range=(0.0, 1.0))[0].astype(np.float64)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
  """P(a positive scores above a negative), ties (within one of HIST_BINS bins) counted half:
  threshold-free separation of two classes of probabilities."""
  hp, hn = _hist(pos), _hist(neg)
  if not hp.sum() or not hn.sum():
    return float('nan')
  below = np.concatenate([[0.0], np.cumsum(hn)[:-1]])
  return float((hp * (below + 0.5 * hn)).sum() / (hp.sum() * hn.sum()))


def quantiles(values: np.ndarray) -> list:
  return [float(q) for q in np.quantile(values, QUANTILES)] if values.size else []


def output_scores(aff: np.ndarray, fg: np.ndarray, gt: np.ndarray) -> dict:
  """Network outputs against the labels. Foreground channel against label > 0: precision and
  recall at FOREGROUND (and at the other DIAGNOSTIC_CUTS), AUC and quantiles per class.
  Each nearest-neighbour channel, over pairs inside the crop: same-neuron pairs (one nonzero
  label on both voxels), different-neuron pairs (two different nonzero labels), and
  neuron-background pairs (exactly one voxel labelled; background-background pairs are not
  counted): the fractions on the wrong side of 0.5, the AUC of same-neuron against
  different-neuron pairs and against neuron-background pairs, and quantiles. The thresholds are
  fixed by design; AUC and quantiles show whether the outputs separate the classes even when
  0.5 is not the right cut for this modality."""
  true = gt > 0
  cuts = {}
  for cut in DIAGNOSTIC_CUTS:
    pred = fg >= cut
    both = int((pred & true).sum())
    cuts[str(cut)] = {'precision': both / max(int(pred.sum()), 1), 'recall': both / max(int(true.sum()), 1),
                      'predicted_fraction': float(pred.mean())}
  pos, neg = fg[true], fg[~true]
  out = {'foreground': {**cuts[str(FOREGROUND)], 'labelled_fraction': float(true.mean()), 'by_cut': cuts,
                        'auc': auc(pos, neg), 'quantiles': {'q': list(QUANTILES), 'label_nonzero': quantiles(pos),
                                                            'label_zero': quantiles(neg)}},
         'affinities': []}
  for c, (name, o) in enumerate(zip(('z', 'y', 'x'), vitseg.OFFSETS[:vitseg.NEAREST])):
    here = tuple(slice(d, None) for d in o)
    there = tuple(slice(0, n - d) for d, n in zip(o, gt.shape))
    a, g1, g0 = aff[c][here], gt[here], gt[there]
    same = (g1 == g0) & (g1 > 0)
    diff = (g1 != g0) & (g1 > 0) & (g0 > 0)
    edge = (g1 > 0) != (g0 > 0)
    high = a >= 0.5
    out['affinities'].append({
        'channel': name, 'same_pairs': int(same.sum()), 'diff_pairs': int(diff.sum()),
        'neuron_background_pairs': int(edge.sum()),
        'same_below_0.5': float((~high & same).sum() / max(int(same.sum()), 1)),
        'diff_at_or_above_0.5': float((high & diff).sum() / max(int(diff.sum()), 1)),
        'neuron_background_at_or_above_0.5': float((high & edge).sum() / max(int(edge.sum()), 1)),
        'same_mean': float(a[same].mean()) if same.any() else 0.0,
        'auc_same_vs_diff': auc(a[same], a[diff]), 'auc_same_vs_neuron_background': auc(a[same], a[edge]),
        'quantiles': {'q': list(QUANTILES), 'same': quantiles(a[same]), 'diff': quantiles(a[diff]),
                      'neuron_background': quantiles(a[edge])}})
  return out


def summary(acc: dict) -> dict:
  """Mean recall, precision and F1 of an object_accounting result."""
  m = acc['per_object_mean']
  return {'mean_recall': m['recall'], 'mean_precision': m['precision'], 'mean_f1': m['f1']}
