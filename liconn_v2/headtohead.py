"""FFN against fine-tuned SegNeuron on the tuning crop 12 (docs/headtohead_design.md).

Numpy-only pairing and summaries; the runs are examples/run_headtohead.py.
"""

import numpy as np

CROP = 12
FFN_VARIANT = 'fp16_head32'
THRESHOLD = 0.6  # our pipeline's threshold, chosen on crop 12 in docs/segneuron_finetune_results.md
CHECK_OBJECT = 346093  # FFN's recorded crop-12 trace (manyseed-fp16_head32-20261022)


def f1(precision: float, recall: float) -> float:
  return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def means(rows: list) -> dict:
  return {k: float(np.mean([r[k] for r in rows])) for k in ('recall', 'precision', 'f1')}


def paired(ffn: list, sn: list) -> dict:
  """Per-object F1 of the two methods over the same objects in the same order."""
  if [r['object_id'] for r in ffn] != [r['object_id'] for r in sn]:
    raise ValueError('the two methods scored different objects')
  diff = np.array([a['f1'] - b['f1'] for a, b in zip(ffn, sn)])
  q = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
  found = [i for i, b in enumerate(sn) if not b['seed_in_background']]
  out = {'objects': len(diff), 'ffn': means(ffn), 'segneuron': means(sn),
         'ffn_higher': int((diff > 0).sum()), 'segneuron_higher': int((diff < 0).sum()),
         'equal': int((diff == 0).sum()), 'mean_f1_difference': float(diff.mean()),
         'f1_difference_quantiles': {'q': list(q), 'value': [float(v) for v in np.quantile(diff, q)]},
         'segneuron_seeds_in_background': len(diff) - len(found),
         'ffn_mean_f1_where_segneuron_seed_found': float(np.mean([ffn[i]['f1'] for i in found])),
         'segneuron_mean_f1_where_seed_found': float(np.mean([sn[i]['f1'] for i in found]))}
  for name, rows in (('ffn', ffn), ('segneuron', sn)):
    out[f'{name}_merges'] = int(sum(len(r['merges']) for r in rows))
    out[f'{name}_objects_with_merge'] = int(sum(bool(r['merges']) for r in rows))
    out[f'{name}_merged_voxels'] = int(sum(r['merged_voxels'] for r in rows))
  return out


def speed(ffn: list, ffn_setup_seconds: float, inference_wall_seconds: float, inference_seconds: float,
          post_seconds: float) -> dict:
  """Seconds for the objects traced (FFN) and for the whole crop (SegNeuron), per object, and the
  break-even number of objects at which FFN's mean time per object adds up to SegNeuron's crop.
  GPU only: FFN's trace clock against inference.py's own inference_seconds. End to end: FFN's
  per-seed wall time against inference.py's process wall time plus the CPU post-processing."""
  n = len(ffn)
  trace = float(sum(r['trace_seconds'] for r in ffn))
  wall = float(sum(r['wall_seconds'] for r in ffn))
  steps = int(sum(r['steps'] for r in ffn))
  sn_total = inference_wall_seconds + post_seconds
  return {'objects': n,
          'ffn': {'trace_seconds': trace, 'wall_seconds': wall, 'setup_seconds': ffn_setup_seconds,
                  'steps': steps, 'ms_per_step': 1000 * trace / steps,
                  'wall_seconds_per_object': wall / n,
                  'median_wall_seconds_per_object': float(np.median([r['wall_seconds'] for r in ffn]))},
          'segneuron': {'inference_wall_seconds': inference_wall_seconds, 'inference_seconds': inference_seconds,
                        'post_seconds': post_seconds, 'total_seconds': sn_total, 'seconds_per_object': sn_total / n},
          'ffn_over_segneuron': wall / sn_total,
          'break_even_objects': {'end_to_end': sn_total / (wall / n), 'gpu_only': inference_seconds / (trace / n)}}
