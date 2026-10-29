"""Fine-tuned SegNeuron on the tuning crop 12 (docs/segneuron_finetune_design.md).

  run_segneuron_ft.py prep  --out DIR   # DIR/data/crop-XX_{raw,labels}.npy (crops 0-12), DIR/native/raw.npy
  run_segneuron_ft.py score --out DIR   # DIR/segneuron_ft_crop12.json

Between the steps (cluster/wahab/segneuron_ft.sbatch): examples/train_segneuron.py writes
DIR/train/{train_log.json,best.ckpt}; the authors' inference.py with best.ckpt writes
DIR/native/infer/{affinities.npy,boundaries.tif,inference.json}; their FRMC_post.py writes
DIR/native/frmc/beta_<beta>.npy. score then does exactly what the zero-shot score does for the
native resolution (examples/run_segneuron.py), with the same functions of liconn_v2/segneuron.py.
No test crop is read.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import segneuron  # pylint: disable=g-import-not-at-top
from liconn_v2 import segneuron_ft as ft  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position
from run_segneuron import load_outputs  # pylint: disable=g-import-not-at-top,wrong-import-position


def raw_zyx(k):
  return np.ascontiguousarray(np.transpose(manyseed.load_raw_xyz(k), [2, 1, 0]))


def prep(out: Path):
  if ft.VALID_CROP != vitseg.TUNE_CROP or ft.TRAIN_CROPS != vitseg.TRAIN_CROPS:
    raise ValueError('crop split differs from the ViT design')
  corners = {c['index']: c['xyz'] for c in json.loads((ROOT / 'docs' / 'manyseed_crops.json').read_text())['crops']}
  gaps = {k: ft.crop_gap(corners[ft.VALID_CROP], corners[k]) for k in ft.TRAIN_CROPS}
  if min(gaps.values()) <= 0:
    raise ValueError(f'crop {ft.VALID_CROP} overlaps or touches a training crop: {gaps}')
  manifest = ft.write_crops(out / 'data', raw_zyx, manyseed.load_proofread_zyx)
  manifest['valid_crop_gap_voxels'] = gaps
  (out / 'data' / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
  (out / 'native').mkdir(parents=True, exist_ok=False)
  np.save(out / 'native' / 'raw.npy', raw_zyx(vitseg.TUNE_CROP), allow_pickle=False)  # as zero-shot prep
  print(json.dumps({'crops': manifest['crops'], 'min_gap_voxels': min(gaps.values())}), flush=True)


def score(out: Path):
  gt = manyseed.load_proofread_zyx(vitseg.TUNE_CROP)
  objects = vitseg.tuning_objects(gt)
  aff, fg = load_outputs(out / 'native' / 'infer')
  if fg.shape != gt.shape:
    raise ValueError(f'outputs {fg.shape} are not on the crop grid {gt.shape}')
  log = json.loads((out / 'train' / 'train_log.json').read_text())
  r = {'design': 'docs/segneuron_finetune_design.md', 'crop': vitseg.TUNE_CROP, 'objects': len(objects),
       'resolution': 'native', 'training': {k: log[k] for k in ('best', 'steps_done', 'seconds', 'stopped', 'args')},
       'inference': json.loads((out / 'native' / 'infer' / 'inference.json').read_text()),
       'outputs': segneuron.output_scores(aff, fg, gt)}
  a = segneuron.ours(aff, fg, gt, objects)
  acc = segneuron.object_accounting(a.pop('segmentation'), gt, objects)
  r['ours'] = {**a, **segneuron.summary(acc), **acc}
  frmc = []
  for beta in segneuron.BETAS:
    seg = np.load(out / 'native' / 'frmc' / f'beta_{beta}.npy', allow_pickle=False)
    if seg.shape != gt.shape:
      raise ValueError(f'beta {beta}: segmentation {seg.shape}')
    released = segneuron.object_accounting(seg, gt, objects)
    masked = segneuron.object_accounting(segneuron.mask_background(seg, fg), gt, objects)
    frmc.append({'beta': beta, 'segments': int(len(np.unique(seg))),
                 'as_released': {**segneuron.summary(released), 'background': released['background']},
                 'masked': {**segneuron.summary(masked), 'background': masked['background']},
                 '_accounting': (released, masked)})
  best = max(frmc, key=lambda e: (e['masked']['mean_f1'], e['beta']))  # as zero-shot
  released, masked = best['_accounting']
  for e in frmc:
    del e['_accounting']
  r['frmc'] = {'beta': best['beta'], 'table': frmc,
               'as_released_at_beta': {**segneuron.summary(released), **released},
               'masked_at_beta': {**segneuron.summary(masked), **masked}}
  r['platform'] = provenance()
  (out / 'segneuron_ft_crop12.json').write_text(json.dumps(r, indent=1) + '\n')
  print(json.dumps({'ours': segneuron.summary(acc) | {'threshold': a['threshold']}, 'frmc_beta': best['beta'],
                    'frmc_masked': best['masked'], 'frmc_as_released': best['as_released']}, default=str), flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('step', choices=('prep', 'score'))
  p.add_argument('--out', required=True)
  a = p.parse_args()
  started = time.time()
  {'prep': prep, 'score': score}[a.step](Path(a.out))
  print(json.dumps({'step': a.step, 'seconds': time.time() - started}), flush=True)


if __name__ == '__main__':
  main()
