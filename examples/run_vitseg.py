"""Segments the tuning and test crops with a trained ViT and judges it (docs/vitseg_design.md).

  run_vitseg.py --params PARAMS --reference RUN_DIR [--reference-fp32 RUN_DIR]
                [--fast-run RUN_JSON] [--tune-only] [--out DIR]

1. Crop 12 (tuning): segment, then choose the agglomeration threshold by mean F1 over
   its large proofread objects.
2. Fresh crops 13-40 (test): segment at that threshold and score each registered seed's
   segment with the proofread labels.
3. Pair each seed with the recorded fp16_head32 FFN trace (--reference: a run directory
   with crop-XX/benchmark.json) and apply the tiered verdict; the same against float32
   FFN (--reference-fp32) and on the leakage subset are reported, not gated.
Background (segment 0) is compared with label 0 voxel by voxel on every crop, and seeds
that fall in predicted background are counted; both are reported, not gated.
Speed per crop: network (GPU), watershed and agglomeration (CPU); with --fast-run, the
fast tracer's time for the same crop's seeds. Set LICONN_BACKEND=gpu on the cluster.
--tune-only stops after step 1 (no test crop is read), so crop 12 can be inspected first.

Output: DIR/vitseg_tuning.json (step 1), DIR/vitseg_run.json (all steps)
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import fresh  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position


def reference_records(run_dir: Path) -> dict:
  out = {}
  for c in fresh.load_manifest()['crops']:
    path = run_dir / f"crop-{c['index']:02d}" / 'benchmark.json'
    for r in json.loads(path.read_text())['records']:
      out[r['name']] = r
  return out


def training_objects() -> set:
  """Proofread objects present in the training crops or the tuning crop."""
  ids = set()
  for k in vitseg.TRAIN_CROPS + (vitseg.TUNE_CROP,):
    ids |= set(np.unique(manyseed.load_proofread_zyx(k)).tolist())
  return ids - {0}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--params', required=True)
  p.add_argument('--reference', required=True)
  p.add_argument('--reference-fp32')
  p.add_argument('--fast-run')
  p.add_argument('--tune-only', action='store_true')
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  import jax  # pylint: disable=g-import-not-at-top
  from flax import serialization  # pylint: disable=g-import-not-at-top
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  started = time.time()
  model = vitseg.make_model()
  template = model.init(jax.random.PRNGKey(0),
                        np.zeros((1, vitseg.BLOCK, vitseg.BLOCK, vitseg.BLOCK, 1), np.float32))['params']
  params = serialization.from_bytes(template, Path(a.params).read_bytes())
  net = jax.jit(lambda x: model.apply({'params': params}, x))
  apply_fn = lambda x: jax.device_get(net(x))

  # 1. Tuning crop.
  frag, merges, timing = vitseg.segment_crop(apply_fn, manyseed.load_raw_xyz(vitseg.TUNE_CROP))
  gt = manyseed.load_proofread_zyx(vitseg.TUNE_CROP)
  objects = vitseg.tuning_objects(gt)
  threshold, table = vitseg.choose_threshold(frag, merges, gt, objects)
  tuning = {'crop': vitseg.TUNE_CROP, 'objects': len(objects), 'threshold': threshold, 'table': table,
            'timing': timing,
            'background': vitseg.background_scores(vitseg.segmentation_at(frag, merges, threshold), gt)}
  print(json.dumps({k: v for k, v in tuning.items() if k != 'table'}), flush=True)
  (out / 'vitseg_tuning.json').write_text(json.dumps({**tuning, 'params': a.params, 'platform': provenance()},
                                                     indent=1) + '\n')
  if a.tune_only:
    return

  # 2. Test crops.
  seeds = fresh.recorded_seeds()
  fast = {}
  if a.fast_run:
    fast = {r['name']: r['fast_seconds'] for r in json.loads(Path(a.fast_run).read_text())['records']}
  records, crops = [], []
  for c in fresh.load_manifest()['crops']:
    k = c['index']
    frag, merges, timing = vitseg.segment_crop(apply_fn, fresh.load_raw_xyz(k))
    t = time.perf_counter()
    seg = vitseg.segmentation_at(frag, merges, threshold)
    timing['relabel_seconds'] = time.perf_counter() - t
    timing['total_seconds'] = sum(timing[s] for s in ('network_seconds', 'watershed_seconds',
                                                      'agglomeration_seconds', 'relabel_seconds'))
    gt = fresh.load_proofread_zyx(k)
    mine = [s for s in seeds if s['crop'] == k]
    recs = vitseg.seed_records(seg, gt, mine)
    for r, s in zip(recs, mine):
      if r['object_id'] != s['object_id']:
        raise ValueError(f"seed {s['name']}: object {r['object_id']} != registered {s['object_id']}")
    records += recs
    entry = {'crop': k, 'seeds': len(mine), 'segments': int(len(np.unique(seg[seg > 0]))), **timing,
             'large_objects': len(vitseg.large_objects(gt)), 'background': vitseg.background_scores(seg, gt)}
    if fast:
      entry['ffn_fast_seconds_seeds'] = sum(fast[s['name']] for s in mine)
    crops.append(entry)
    print(json.dumps(entry), flush=True)

  # 3. Verdicts.
  leak = training_objects()
  for r in records:
    r['object_in_training_crops'] = r['object_id'] in leak  # training or tuning crops
  ref = reference_records(Path(a.reference))
  pairs = [(ref[r['name']], r) for r in records]
  for f, r in pairs:
    r['ffn_unlabeled_fraction'] = f['unlabeled_fraction']
  result = {'design': 'docs/vitseg_design.md', 'params': a.params, 'reference': a.reference,
            'threshold': threshold, 'tuning': tuning, 'records': records, 'crops': crops,
            'verdict': vitseg.verdict(pairs),
            'leakage_subset': {
                'in_training_crops': vitseg.verdict([pr for pr in pairs if pr[1]['object_in_training_crops']]),
                'not_in_training_crops': vitseg.verdict([pr for pr in pairs if not pr[1]['object_in_training_crops']])},
            'by_class': {cls: vitseg.verdict([pr for pr in pairs if pr[1]['class'] == cls])
                         for cls in ('axon', 'dendrite')},
            'seeds_in_predicted_background': sum(r['mask_voxels'] == 0 for r in records),
            'background_mean_over_crops': {k: float(np.mean([c['background'][k] for c in crops]))
                                           for k in ('precision', 'recall', 'label0_fraction', 'predicted_fraction')},
            'mean_unlabeled_fraction': {  # over seeds whose ViT segment is not empty
                'vit': float(np.mean([c['unlabeled_fraction'] for _, c in pairs if c['mask_voxels']] or [0.0])),
                'ffn': float(np.mean([r['unlabeled_fraction'] for r, c in pairs if c['mask_voxels']] or [0.0]))}}
  if a.reference_fp32:
    ref32 = reference_records(Path(a.reference_fp32))
    result['against_fp32'] = vitseg.verdict([(ref32[r['name']], r) for r in records])
  if fast:
    vit_s = sum(c['total_seconds'] for c in crops)
    ffn_s = sum(c['ffn_fast_seconds_seeds'] for c in crops)
    mean_seed = ffn_s / len(records)
    est = sum(c['large_objects'] for c in crops) * mean_seed
    result['speed'] = {'vit_total_seconds': vit_s, 'ffn_fast_seconds_for_the_seeds': ffn_s,
                       'ffn_mean_seconds_per_seed': mean_seed,
                       'estimate_ffn_seconds_all_large_objects': est,
                       'note': 'estimate = objects with >= 20,000 interior voxels x mean FFN seconds per seed'}
  result['wall_seconds_total'] = time.time() - started
  result['platform'] = provenance()
  (out / 'vitseg_run.json').write_text(json.dumps(result, indent=1) + '\n')
  print(json.dumps({k: result[k] for k in ('threshold', 'verdict')}, indent=1), flush=True)


if __name__ == '__main__':
  main()
