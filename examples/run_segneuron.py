"""SegNeuron zero-shot on the tuning crop 12 (docs/segneuron_design.md).

  run_segneuron.py prep  --out DIR   # DIR/native/raw.npy and DIR/xy2/raw.npy, uint8 (z, y, x)
  run_segneuron.py grid  --out DIR   # DIR/xy2/grid/{affinities.npy,boundaries.tif} on the crop grid
  run_segneuron.py score --out DIR   # DIR/segneuron_crop12.json

Between the steps, the authors' scripts run unmodified in their own environments
(cluster/wahab/segneuron.sbatch): Train_and_Inference/inference.py writes
DIR/<resolution>/infer/{affinities.npy,boundaries.tif,inference.json}, and
Postprocess/FRMC_post.py writes DIR/<resolution>/frmc/beta_<beta>.npy from the crop-grid
outputs (native: infer/, xy2: grid/).

score, for each resolution: the outputs against the labels; post-processing A (ours: vitseg
fragments and agglomeration, threshold chosen on crop 12); post-processing B (FRMC) at each beta,
as released and with foreground < 0.5 set to background, beta chosen on crop 12 by the masked
mean F1. All on the 457 tuning objects of crop 12; no test crop is read.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import segneuron  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position


def outputs_dir(out: Path, res: str) -> Path:
  """Where the crop-grid outputs of a resolution are."""
  return out / res / ('infer' if res == 'native' else 'grid')


def load_outputs(d: Path):
  import tifffile  # pylint: disable=g-import-not-at-top
  aff = np.load(d / 'affinities.npy', allow_pickle=False)
  fg = tifffile.imread(d / 'boundaries.tif')
  if aff.shape != (3,) + fg.shape or aff.dtype != np.float32 or fg.dtype != np.float32:
    raise ValueError(f'unexpected outputs in {d}')
  return aff, fg


def prep(out: Path):
  raw = np.ascontiguousarray(np.transpose(manyseed.load_raw_xyz(vitseg.TUNE_CROP), [2, 1, 0]))
  for res, vol in (('native', raw), ('xy2', segneuron.upsample_xy2(raw))):
    (out / res).mkdir(parents=True, exist_ok=False)
    np.save(out / res / 'raw.npy', vol, allow_pickle=False)
    print(json.dumps({'resolution': res, 'shape_zyx': list(vol.shape)}), flush=True)


def grid(out: Path):
  import tifffile  # pylint: disable=g-import-not-at-top
  aff, fg = segneuron.to_crop_grid(*load_outputs(out / 'xy2' / 'infer'))
  d = out / 'xy2' / 'grid'
  d.mkdir(exist_ok=False)
  np.save(d / 'affinities.npy', aff, allow_pickle=False)
  tifffile.imwrite(d / 'boundaries.tif', fg, photometric='minisblack', metadata={'axes': 'ZYX'})
  print(json.dumps({'grid': str(d), 'shape_zyx': list(fg.shape)}), flush=True)


def score(out: Path):
  gt = manyseed.load_proofread_zyx(vitseg.TUNE_CROP)
  objects = vitseg.tuning_objects(gt)
  result = {'design': 'docs/segneuron_design.md', 'crop': vitseg.TUNE_CROP, 'objects': len(objects),
            'resolutions': {}}
  for res in segneuron.RESOLUTIONS:
    aff, fg = load_outputs(outputs_dir(out, res))
    if fg.shape != gt.shape:
      raise ValueError(f'{res}: outputs {fg.shape} are not on the crop grid {gt.shape}')
    r = {'inference': json.loads((out / res / 'infer' / 'inference.json').read_text()),
         'outputs': segneuron.output_scores(aff, fg, gt)}
    a = segneuron.ours(aff, fg, gt, objects)
    acc = segneuron.object_accounting(a.pop('segmentation'), gt, objects)
    r['ours'] = {**a, **segneuron.summary(acc), **acc}
    frmc = []
    for beta in segneuron.BETAS:
      seg = np.load(out / res / 'frmc' / f'beta_{beta}.npy', allow_pickle=False)
      if seg.shape != gt.shape:
        raise ValueError(f'{res} beta {beta}: segmentation {seg.shape}')
      released = segneuron.object_accounting(seg, gt, objects)
      masked = segneuron.object_accounting(segneuron.mask_background(seg, fg), gt, objects)
      frmc.append({'beta': beta, 'segments': int(len(np.unique(seg))),
                   'as_released': {**segneuron.summary(released), 'background': released['background']},
                   'masked': {**segneuron.summary(masked), 'background': masked['background']},
                   '_accounting': (released, masked)})
    best = max(frmc, key=lambda e: (e['masked']['mean_f1'], e['beta']))
    released, masked = best['_accounting']
    for e in frmc:
      del e['_accounting']
    r['frmc'] = {'beta': best['beta'], 'table': frmc,
                 'as_released_at_beta': {**segneuron.summary(released), **released},
                 'masked_at_beta': {**segneuron.summary(masked), **masked}}
    result['resolutions'][res] = r
    print(json.dumps({'resolution': res, 'ours': segneuron.summary(acc) | {'threshold': a['threshold']},
                      'frmc_beta': best['beta'], 'frmc_masked': best['masked'],
                      'frmc_as_released': best['as_released']}, default=str), flush=True)
  result['platform'] = provenance()
  (out / 'segneuron_crop12.json').write_text(json.dumps(result, indent=1) + '\n')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('step', choices=('prep', 'grid', 'score'))
  p.add_argument('--out', required=True)
  a = p.parse_args()
  out = Path(a.out)
  started = time.time()
  {'prep': prep, 'grid': grid, 'score': score}[a.step](out)
  print(json.dumps({'step': a.step, 'seconds': time.time() - started}), flush=True)


if __name__ == '__main__':
  main()
