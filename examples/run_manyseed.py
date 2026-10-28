"""Many-seed runs on one new crop (docs/manyseed_design.md).

  run_manyseed.py --variant fp32|fp16|fp16_head32|... --crop K [--seed NAME ...] [--out DIR]

Traces crop K's recorded seeds (docs/manyseed_seeds.json, re-derived from the crop to
check they follow the seed rule) with the authors' network in the given number format
(precision.VARIANTS, or a candidate layout of liconn_v2/earlystop.py) and writes benchmark.json and benchmark_masks.npz in the benchmark's format. The fp32
run is the reference. Set LICONN_BACKEND=gpu on the cluster.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import earlystop  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import precision  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position


VARIANTS = sorted(precision.VARIANTS) + ['fp16_head32']


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--variant', required=True, choices=VARIANTS)
  p.add_argument('--crop', type=int, required=True)
  p.add_argument('--seed', action='append', default=[])
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)

  manifest = manyseed.load_manifest()
  crop = manifest['crops'][a.crop]
  raw = reference.normalize(manyseed.load_raw_xyz(a.crop))
  gt = manyseed.load_proofread_zyx(a.crop)
  all_seeds = manyseed.seeds_for_crop(a.crop, gt)
  seeds = benchmark.pick([{**s, 'seed_zyx': tuple(s['seed_zyx'])} for s in all_seeds], a.seed)
  started = time.time()
  run = precision.run_variant if a.variant in precision.VARIANTS else earlystop.run_layout
  records, masks = run(
      seeds, raw, gt, data.checkpoint_path(), a.variant,
      on_record=lambda r: print(f"{r['name']}: steps={r['steps']} recall={r['recall']:.3f} "
                                f"precision={r['precision']:.3f} {r['trace_seconds']:.0f}s", flush=True))
  arrays = {'shape_zyx': np.array(gt.shape)}
  for name, mask in masks.items():
    if not mask.any():
      continue
    zz, yy, xx = np.nonzero(mask)
    lo = np.array([zz.min(), yy.min(), xx.min()])
    hi = np.array([zz.max(), yy.max(), xx.max()]) + 1
    arrays['mask_' + name] = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    arrays['offset_zyx_' + name] = lo
  np.savez_compressed(out / 'benchmark_masks.npz', **arrays)
  result = {
      'design': 'docs/manyseed_design.md',
      'crop': a.crop,
      'crop_xyz': crop['xyz'],
      'subset': 'all seeds' if len(seeds) == len(all_seeds) else [s['name'] for s in seeds],
      'number_format': a.variant,
      'seeds': [{**s, 'seed_zyx': list(s['seed_zyx'])} for s in seeds],
      'records': records,
      'summary': benchmark.summarize(records),
      'inputs_sha256': {**crop['sha256'], **{k: v for k, v in data.PINNED_SHA256.items() if k.startswith('ckpt/')}},
      'wall_seconds_total': time.time() - started,
      'platform': provenance(),
  }
  (out / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result['summary']['all'], indent=2))


if __name__ == '__main__':
  main()
