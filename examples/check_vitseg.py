"""CPU smoke check of the ViT segmentation path (docs/vitseg_design.md); not an accuracy test.

  check_vitseg.py --cpu-smoke [--out DIR]

On a synthetic 64^3 volume (random Voronoi cells, unlabelled 1-voxel walls, an image
that is bright inside cells and dark on walls):
1. tiling: an identity "network" reassembles the whole volume exactly;
2. segmentation: a perfect output (the training targets: affinities and the neuron
   channel) gives every cell's seed recall and precision 1, and the walls (label 0) stay
   background (segment 0);
3. training and inference: a tiny model (32^3 blocks, 4^3 patches) trains for a few steps
   and the full path (tiling, watershed, agglomeration, scoring, verdict) runs.
Exits non-zero if check 1 or 2 fails, or the path raises.

Output: DIR/vitseg_check.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

SIDE = 64
TINY = {'block': 32, 'patch': 4, 'width': 32, 'depth': 1, 'heads': 2, 'encoder': (8, 16)}
TINY_STRIDE = 24
TRAIN_STEPS = 30


def synthetic(side=SIDE, cells=12, seed=0):
  """(labels uint32, image float32): Voronoi cells 1..cells with 0-valued walls."""
  rng = np.random.default_rng(seed)
  pts = rng.uniform(0, side, size=(cells, 3))
  grid = np.stack(np.meshgrid(*(np.arange(side),) * 3, indexing='ij'), -1).reshape(-1, 3)
  d = ((grid[:, None, :] - pts[None]) ** 2).sum(-1)
  lab = (np.argmin(d, 1) + 1).reshape((side,) * 3).astype(np.uint32)
  wall = np.zeros(lab.shape, bool)
  for axis in range(3):
    diff = np.diff(lab, axis=axis) != 0
    sl = [slice(None)] * 3
    sl[axis] = slice(1, None)
    wall[tuple(sl)] |= diff
  lab[wall] = 0
  image = np.where(lab > 0, 1.0, -1.0).astype(np.float32) + rng.normal(0, 0.1, lab.shape).astype(np.float32)
  return lab, image


def seeds_for(lab):
  out = []
  for i in np.unique(lab).tolist():
    if i == 0:
      continue
    zz, yy, xx = np.nonzero(lab == i)
    k = len(zz) // 2
    out.append({'name': f'c{i}', 'class': 'axon', 'object_id': i, 'seed_zyx': [int(zz[k]), int(yy[k]), int(xx[k])]})
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--cpu-smoke', action='store_true', required=True)
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  import jax  # pylint: disable=g-import-not-at-top
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  lab, image = synthetic()
  seeds = seeds_for(lab)
  checks = {}

  # 1. Tiling reassembles exactly (identity network, logit channels = the image).
  ident = lambda x: np.repeat(x, 2, axis=-1)
  got = vitseg.predict_affinities(ident, image, TINY['block'], TINY_STRIDE)
  from scipy.special import expit  # pylint: disable=g-import-not-at-top
  checks['tiling_exact'] = bool(np.array_equal(got[..., 0], expit(image)))

  # 2. A perfect output recovers every cell and keeps the walls as background.
  target, mask = vitseg.training_targets(lab)
  perfect = np.where(mask > 0, target, 0.0).astype(np.float32)
  frag = vitseg.fragments(perfect)
  merges = vitseg.agglomerate(*vitseg.fragment_edges(frag, perfect))
  seg = vitseg.segmentation_at(frag, merges, 0.5)
  recs = vitseg.seed_records(seg, lab, seeds)
  checks['perfect_affinities_recover_cells'] = all(r['recall'] == 1.0 and r['precision'] == 1.0 for r in recs)
  checks['background_stays_zero'] = bool(np.array_equal(seg == 0, lab == 0))

  # 3. Tiny model: train, infer, segment, score, judge (path only).
  model = vitseg.make_model(**TINY)
  params = model.init(jax.random.PRNGKey(0), np.zeros((1,) + (TINY['block'],) * 3 + (1,), np.float32))['params']
  opt = vitseg.make_optimizer(TRAIN_STEPS)
  opt_state = opt.init(params)
  step = vitseg.make_train_step(model, opt)
  losses = []
  for i in range(TRAIN_STEPS):
    x, t, m = vitseg.sample_batch([image], [lab], i, block=TINY['block'])
    params, opt_state, loss, _ = step(params, opt_state, x, t, m)
    losses.append(float(loss))
  net = jax.jit(lambda x: model.apply({'params': params}, x))
  aff = vitseg.predict_affinities(lambda x: np.asarray(net(x)), image, TINY['block'], TINY_STRIDE)
  frag = vitseg.fragments(aff)
  merges = vitseg.agglomerate(*vitseg.fragment_edges(frag, aff))
  threshold, table = vitseg.choose_threshold(frag, merges, lab, seeds)
  model_recs = vitseg.seed_records(vitseg.segmentation_at(frag, merges, threshold), lab, seeds)
  verdict = vitseg.verdict(list(zip(recs, model_recs)))
  background = vitseg.background_scores(vitseg.segmentation_at(frag, merges, threshold), lab)
  checks['path_ran'] = True

  result = {'checks': checks, 'synthetic': {'side': SIDE, 'cells': len(seeds)},
            'tiny_model': {k: list(v) if isinstance(v, tuple) else v for k, v in TINY.items()},
            'train_losses_first_last': [losses[0], losses[-1]], 'threshold': threshold, 'tuning_table': table,
            'verdict_vs_perfect': verdict, 'tiny_model_background': background,
            'note': 'Path check on synthetic data; says nothing about accuracy on LICONN data.'}
  (out / 'vitseg_check.json').write_text(json.dumps(result, indent=1) + '\n')
  print(json.dumps(checks))
  if not (checks['tiling_exact'] and checks['perfect_affinities_recover_cells'] and checks['background_stays_zero']):
    sys.exit(1)


if __name__ == '__main__':
  main()
