"""Fine-tunes the pretrained SegNeuron on LICONN crops 0-11 (docs/segneuron_finetune_design.md).

  train_segneuron.py --sn-src DIR --ckpt FILE --data DIR --out DIR --iters N [--batch B]
                     [--hours H] [--workers W]

Runs in the torch training environment, not the liconn one. The network, targets, loss,
optimiser settings and augmentations are the authors': they are imported from the pinned
checkout (--sn-src, its Train_and_Inference folder), and the augmentations are their
provider's own perturbations_init and apply_perturbations methods. Only the data source
(LICONN crops from --data, written by examples/run_segneuron_ft.py prep), the learning rate,
the run length and a held-out validation on crop 12 are ours.

Writes DIR/train_log.json (loss every 100 steps, validation every 1000, the step chosen),
DIR/best.ckpt (lowest validation loss, the authors' {'model_weights': ...} format, readable by
their inference.py) and DIR/last.ckpt. --iters 200 --hours 0.1 is the timing probe.
"""

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import segneuron_ft as ft  # pylint: disable=g-import-not-at-top


def authors(src: Path):
  """The authors' modules, imported from Train_and_Inference of the pinned checkout."""
  sys.path.insert(0, str(src))
  from model.Mnet import MNet  # pylint: disable=g-import-not-at-top,import-error
  from loss.loss import WeightedBCE  # pylint: disable=g-import-not-at-top,import-error
  from utils.seg_util import mknhood3d  # pylint: disable=g-import-not-at-top,import-error
  from utils.aff_util import seg_to_affgraph  # pylint: disable=g-import-not-at-top,import-error
  from utils.augmentation import SimpleAugment  # pylint: disable=g-import-not-at-top,import-error
  import supervised_provider  # pylint: disable=g-import-not-at-top,import-error
  return MNet, WeightedBCE, mknhood3d, seg_to_affgraph, SimpleAugment, supervised_provider


class Augmenter:
  """The authors' supervised augmentations: their provider's perturbations_init and
  apply_perturbations, with the values of their SegNeuron.yaml (DATA section)."""

  def __init__(self, provider_module, simple_augment):
    t = provider_module.Train.__new__(provider_module.Train)  # no dataset loading
    t.model_type = 'superhuman'
    t.min_noise_std, t.max_noise_std = 0.01, 0.2
    t.min_kernel_size, t.max_kernel_size = 3, 9
    t.min_sigma, t.max_sigma = 0, 2
    t.perturbations_init()
    self.t, self.simple = t, simple_augment()

  def __call__(self, imgs, lb):
    imgs, lb = self.simple([imgs, lb])
    return self.t.apply_perturbations(imgs, lb)


def make_dataset(torch, data: Path, mods, seed: int):
  MNet, WeightedBCE, mknhood3d, seg_to_affgraph, SimpleAugment, provider = mods  # pylint: disable=unused-variable,invalid-name
  raws = {k: np.load(data / ft.RAW.format(k), mmap_mode='r') for k in ft.TRAIN_CROPS}
  labs = {k: np.load(data / ft.LABELS.format(k), mmap_mode='r') for k in ft.TRAIN_CROPS}
  shapes = {k: raws[k].shape for k in ft.TRAIN_CROPS}
  nhood = mknhood3d(1)

  class Patches(torch.utils.data.IterableDataset):
    def __iter__(self):
      info = torch.utils.data.get_worker_info()
      wid = info.id if info else 0
      rng = np.random.default_rng([seed, wid])
      np.random.seed((seed * 1000 + wid) % 2**32)  # the authors' augmentations draw from these
      import random  # pylint: disable=g-import-not-at-top
      random.seed(seed * 1000 + wid)
      aug = Augmenter(provider, SimpleAugment)
      while True:
        k, (z, y, x) = ft.train_draw(rng, shapes)
        sl = (slice(z, z + ft.PATCH[0]), slice(y, y + ft.PATCH[1]), slice(x, x + ft.PATCH[2]))
        imgs = np.asarray(raws[k][sl], np.float32) / 255.0
        lb = ft.compact_patch(np.asarray(labs[k][sl]))  # the elastic step stores labels as uint16
        imgs, lb = aug(imgs, lb)
        yield targets(imgs, lb, nhood, seg_to_affgraph)

  return Patches()


def targets(imgs, lb, nhood, seg_to_affgraph):
  """As the authors' provider: affinities with pad='replicate', foreground label != 0, mask 1."""
  affs = seg_to_affgraph(lb, nhood, pad='replicate').astype(np.float32)
  bounds = np.float32(lb != 0)
  return (np.ascontiguousarray(imgs[np.newaxis], np.float32), np.ascontiguousarray(affs),
          np.ascontiguousarray(bounds[np.newaxis]), np.ones((1,) + lb.shape, np.float32))


def load_pretrained(torch, model, ckpt: Path):
  """All weights of the released model (the authors' continue_train path), strictly."""
  state = torch.load(ckpt, map_location='cpu', weights_only=True)['model_weights']
  model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--sn-src', required=True)
  p.add_argument('--ckpt', required=True)
  p.add_argument('--data', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--iters', type=int, required=True)
  p.add_argument('--batch', type=int, default=8)
  p.add_argument('--hours', type=float, default=20.0)
  p.add_argument('--workers', type=int, default=6)
  p.add_argument('--seed', type=int, default=666)  # the authors' random_seed
  a = p.parse_args()
  import torch  # pylint: disable=g-import-not-at-top
  out, data = Path(a.out), Path(a.data)
  out.mkdir(parents=True, exist_ok=False)
  mods = authors(Path(a.sn_src))
  MNet, WeightedBCE, mknhood3d, seg_to_affgraph = mods[:4]  # pylint: disable=invalid-name
  torch.manual_seed(a.seed)
  dev = torch.device('cuda:0')
  model = MNet(1, kn=(32, 64, 96, 128, 256), FMU='sub')
  load_pretrained(torch, model, Path(a.ckpt))
  model = model.to(dev)
  opt = torch.optim.Adam(model.parameters(), lr=ft.BASE_LR, betas=(0.9, 0.999), eps=0.01,
                         weight_decay=1e-6, amsgrad=True)  # supervised_train.py
  crit = WeightedBCE()

  # validation: fixed crop-12 patches, no augmentation
  vraw = np.load(data / ft.RAW.format(ft.VALID_CROP), mmap_mode='r')
  vlab = np.load(data / ft.LABELS.format(ft.VALID_CROP), mmap_mode='r')
  nhood = mknhood3d(1)
  vbatch, vlabels = [], []
  for z, y, x in ft.valid_corners(vraw.shape):
    sl = (slice(z, z + ft.PATCH[0]), slice(y, y + ft.PATCH[1]), slice(x, x + ft.PATCH[2]))
    vlabels.append(ft.compact_patch(np.asarray(vlab[sl])))
    vbatch.append(targets(np.asarray(vraw[sl], np.float32) / 255.0, vlabels[-1], nhood, seg_to_affgraph))

  def validate():
    """Mean loss on the fixed crop-12 patches (chooses the checkpoint), and the boundary pair
    counts (a diagnostic only)."""
    model.eval()
    total, counts = 0.0, {}
    with torch.no_grad():
      for i in range(0, len(vbatch), a.batch):
        parts = [torch.from_numpy(np.stack(c)).to(dev) for c in zip(*vbatch[i:i + a.batch])]
        pred, pred_b = model(parts[0])
        n = parts[0].shape[0]
        total += n * float(crit(pred, parts[1], weight=parts[3]) + crit(pred_b, parts[2], weight=parts[3]))
        for j, p in enumerate(pred.float().cpu().numpy()):
          for key, v in ft.boundary_errors(p, vlabels[i + j]).items():
            counts[key] = counts.get(key, 0) + v
    model.train()
    return {'valid_loss': total / len(vbatch), 'same_below_0.5': counts['same_below_0.5'] / counts['same'],
            'diff_at_or_above_0.5': counts['diff_at_or_above_0.5'] / counts['diff'], 'diff_pairs': counts['diff']}

  loader = iter(torch.utils.data.DataLoader(make_dataset(torch, data, mods, a.seed), batch_size=a.batch,
                                            num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0))
  log = {'design': 'docs/segneuron_finetune_design.md', 'args': vars(a), 'base_lr': ft.BASE_LR, 'end_lr': ft.END_LR,
         'power': ft.POWER, 'patch_zyx': ft.PATCH, 'train_crops': list(ft.TRAIN_CROPS), 'valid_crop': ft.VALID_CROP,
         'valid_patches': ft.VALID_PATCHES, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
         'node': platform.node(), 'train': [], 'validation': [], 'stopped': None}
  save = lambda name, step: torch.save({'current_iter': step, 'model_weights': model.state_dict()}, out / name)
  write_log = lambda: (out / 'train_log.json').write_text(json.dumps(log, indent=1) + '\n')

  log['validation'].append({'step': 0, **validate()})  # the pretrained model
  save('best.ckpt', 0)
  model.train()
  started, window, wtime = time.time(), [], time.time()
  for step in range(1, a.iters + 1):
    lr = ft.learning_rate(step, a.iters)
    for g in opt.param_groups:
      g['lr'] = lr
    inputs, target, bound, mask = (t.to(dev, non_blocking=True) for t in next(loader))
    opt.zero_grad()
    pred, pred_b = model(inputs)
    loss = crit(pred, target, weight=mask) + crit(pred_b, bound, weight=mask)
    if not math.isfinite(float(loss)):
      log['stopped'] = {'step': step, 'reason': 'non-finite training loss'}
      break
    loss.backward()
    opt.step()
    window.append(float(loss))
    if step % 100 == 0:
      log['train'].append({'step': step, 'loss': float(np.mean(window)), 'lr': lr,
                           'seconds_per_step': (time.time() - wtime) / len(window),
                           'max_memory_gb': torch.cuda.max_memory_allocated() / 2**30})
      window, wtime = [], time.time()
    if step % ft.VALID_EVERY == 0 or step == a.iters:
      log['validation'].append({'step': step, **validate()})
      if log['validation'][-1]['valid_loss'] < ft.best_step(log['validation'][:-1])['valid_loss']:
        save('best.ckpt', step)
      save('last.ckpt', step)
      write_log()
    if time.time() - started > a.hours * 3600:
      log['stopped'] = {'step': step, 'reason': f'time budget {a.hours} h'}
      break
  if log['validation'][-1]['step'] != step:
    log['validation'].append({'step': step, **validate()})  # ended between validations: judge the final weights too
    if log['validation'][-1]['valid_loss'] < ft.best_step(log['validation'][:-1])['valid_loss']:
      save('best.ckpt', step)
    save('last.ckpt', step)
  log['steps_done'] = step
  log['seconds'] = time.time() - started
  log['best'] = ft.best_step(log['validation'])
  write_log()
  print(json.dumps({'steps': step, 'best': log['best'], 'stopped': log['stopped']}), flush=True)


if __name__ == '__main__':
  main()
