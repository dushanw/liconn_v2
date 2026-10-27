"""Trains the single-pass ViT segmentation pilot (docs/vitseg_design.md).

  train_vitseg.py --steps N --out DIR [--timing] [--checkpoint-every K]

Random 128^3 blocks from many-seed crops 0-11 (proofread labels), batch 2, AdamW with
warm-up and cosine decay, float16 compute with a static loss scale. Batches depend only
on (seed, step), so a run resumed from DIR/state.msgpack draws the same batches.
Gradients are clipped at global norm 1.0. The run stops itself (log 'stopped') after
vitseg.MAX_CONSECUTIVE_SKIPS non-finite steps in a row or a non-finite validation loss,
keeping the best checkpoint so far.
--timing runs N steps without saving parameters and reports the time per step.
Set LICONN_BACKEND=gpu on the cluster.

At every checkpoint the loss on fixed crop-12 validation blocks is recorded; the
checkpoint with the lowest validation loss is the selected model.

Output: DIR/train_log.json, DIR/params.msgpack (the selected checkpoint),
DIR/final_params.msgpack (the last step), DIR/state.msgpack (latest resumable state).
"""

import argparse
import hashlib
import json
from pathlib import Path
import queue
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position


def prefetch(images, labels, first, last, depth=4):
  """Yields (step, batch) for steps first..last-1, prepared by a background thread."""
  q = queue.Queue(depth)

  def work():
    for step in range(first, last):
      q.put((step, vitseg.sample_batch(images, labels, step)))
    q.put(None)

  threading.Thread(target=work, daemon=True).start()
  while (item := q.get()) is not None:
    yield item


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--steps', type=int, required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--timing', action='store_true')
  p.add_argument('--checkpoint-every', type=int, default=2000)
  a = p.parse_args()
  import jax  # pylint: disable=g-import-not-at-top
  from flax import serialization  # pylint: disable=g-import-not-at-top
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  started = time.time()
  images = [vitseg.normalize_zyx(manyseed.load_raw_xyz(k)) for k in vitseg.TRAIN_CROPS]
  labels = [manyseed.load_proofread_zyx(k) for k in vitseg.TRAIN_CROPS]
  model = vitseg.make_model()
  params = model.init(jax.random.PRNGKey(vitseg.RNG_SEED),
                      np.zeros((1, vitseg.BLOCK, vitseg.BLOCK, vitseg.BLOCK, 1), np.float32))['params']
  optimizer = vitseg.make_optimizer(a.steps)
  opt_state = optimizer.init(params)
  step_fn = vitseg.make_train_step(model, optimizer)
  val = vitseg.validation_batch(vitseg.normalize_zyx(manyseed.load_raw_xyz(vitseg.TUNE_CROP)),
                                manyseed.load_proofread_zyx(vitseg.TUNE_CROP))
  eval_loss = vitseg.make_eval_loss(model)

  def validation_loss(p):
    return float(np.mean([float(eval_loss(p, *(v[i:i + vitseg.BATCH] for v in val)))
                          for i in range(0, vitseg.VALIDATION_BLOCKS, vitseg.BATCH)]))

  log = {'design': 'docs/vitseg_design.md', 'steps': a.steps, 'timing_only': a.timing,
         'train_crops': list(vitseg.TRAIN_CROPS), 'losses': [], 'skipped_steps': [], 'resumed_at': [],
         'validation': [], 'best_step': None, 'best_validation_loss': None, 'stopped': None}
  first = 0
  state_path = out / 'state.msgpack'
  if state_path.exists() and not a.timing:
    saved = serialization.msgpack_restore(state_path.read_bytes())
    params = serialization.from_state_dict(params, saved['params'])
    opt_state = serialization.from_state_dict(opt_state, saved['opt_state'])
    first = int(saved['step'])
    log = json.loads((out / 'train_log.json').read_text())
    if log.get('stopped'):
      print(json.dumps({'not_resumed': 'the run stopped itself', 'stopped': log['stopped']}), flush=True)
      return
    log['resumed_at'].append(first)
  t_window, window, run_of_skips = time.perf_counter(), [], 0
  for step, (x, t, m) in prefetch(images, labels, first, a.steps):
    params, opt_state, loss, finite = step_fn(params, opt_state, x, t, m)
    window.append(loss)
    if not bool(finite):
      log['skipped_steps'].append(step)
      run_of_skips += 1
    else:
      run_of_skips = 0
    if (reason := vitseg.stop_reason(run_of_skips)):
      log['stopped'] = {'step': step + 1, 'reason': reason}
    if (step + 1) % 100 == 0 or step + 1 == a.steps:
      losses = [float(v) for v in jax.device_get(window)]
      elapsed = time.perf_counter() - t_window
      log['losses'].append({'step': step + 1, 'mean_loss': float(np.mean(losses)),
                            'seconds_per_step': elapsed / len(losses)})
      print(json.dumps(log['losses'][-1]), flush=True)
      t_window, window = time.perf_counter(), []
    if not a.timing and ((step + 1) % a.checkpoint_every == 0 or step + 1 == a.steps or log['stopped']):
      vl = validation_loss(params)
      log['validation'].append({'step': step + 1, 'loss': vl})
      if not log['stopped'] and (reason := vitseg.stop_reason(0, vl)):
        log['stopped'] = {'step': step + 1, 'reason': reason}
      if np.isfinite(vl) and (log['best_validation_loss'] is None or vl < log['best_validation_loss']):
        log['best_step'], log['best_validation_loss'] = step + 1, vl
        blob = serialization.to_bytes(params)
        (out / 'best_params.msgpack.tmp').write_bytes(blob)
        (out / 'best_params.msgpack.tmp').replace(out / 'best_params.msgpack')
      print(json.dumps(log['validation'][-1]), flush=True)
      state = {'params': serialization.to_state_dict(params), 'opt_state': serialization.to_state_dict(opt_state),
               'step': step + 1}
      tmp = out / 'state.msgpack.tmp'
      tmp.write_bytes(serialization.msgpack_serialize(state))
      tmp.replace(state_path)
      (out / 'train_log.json').write_text(json.dumps(log, indent=1) + '\n')
    if log['stopped']:
      print(json.dumps({'stopped': log['stopped']}), flush=True)
      break
  log['skipped_fraction'] = len(log['skipped_steps']) / a.steps
  log['unstable'] = bool(log['stopped']) or log['skipped_fraction'] > vitseg.MAX_SKIPPED_FRACTION
  log['wall_seconds_this_job'] = time.time() - started
  log['platform'] = provenance()
  if not a.timing:
    blob = serialization.to_bytes(params)
    (out / 'final_params.msgpack').write_bytes(blob)
    log['final_params_sha256'] = hashlib.sha256(blob).hexdigest()
    best = out / 'best_params.msgpack'
    if best.exists():  # absent only if the run stopped before any finite validation loss
      blob = best.read_bytes()  # the selected checkpoint
      (out / 'params.msgpack').write_bytes(blob)
      log['params_sha256'] = hashlib.sha256(blob).hexdigest()
  (out / 'train_log.json').write_text(json.dumps(log, indent=1) + '\n')
  print(json.dumps({k: v for k, v in log.items() if k not in ('losses', 'platform')}), flush=True)


if __name__ == '__main__':
  main()
