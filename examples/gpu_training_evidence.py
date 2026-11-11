"""Evidence for fine-tuning SegNeuron on newer GPUs (docs/gpu_training_results.md), from the training logs.

  gpu_training_evidence.py [--out docs/gpu_training_evidence.json]

Reads only the train_log.json files of the runs listed below (in the data cache), the job scripts kept in
their run folders and the records of their training data, and writes every number the results document and
the manuscript state.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liconn_v2 import data  # pylint: disable=g-import-not-at-top

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = data.cache_dir() / 'runs'
# 200-step timing probes at batch 4 (the repository's examples/train_segneuron.py), by GPU and job.
PROBES = {
    'V100 (Wahab 7478554)': 'segneuron-ft-20261028/probe-b4',
    'A100 PCIe (Wahab 7491576)': 'segneuron-a100-probe-20261030/probe-b4',
    'A100 PCIe (MIT 27401098)': 'segneuron-mit-probe-20261030/probe-b4-w6-27401098',
    'H100 (MIT 27401100)': 'segneuron-mit-probe-20261030/probe-b4-w6-27401100',
    'B200 (MIT 27405214)': 'segneuron-b200-batch-20261030/check-b4-27405214',
}
WORKERS_16 = {  # the same probes with 16 data-loading workers instead of 6
    'A100 PCIe (MIT 27401099)': 'segneuron-mit-probe-20261030/probe-b4-w16-27401099',
    'H100 (MIT 27401101)': 'segneuron-mit-probe-20261030/probe-b4-w16-27401101',
}
# B200 batch sizes at equal data (25,600 patches), train_segneuron_bs.py, MIT jobs 27405215-27405218.
BATCH = {4: 'segneuron-b200-batch-20261030/b4-27405215', 8: 'segneuron-b200-batch-20261030/b8-27405216',
         16: 'segneuron-b200-batch-20261030/b16-27405217', 32: 'segneuron-b200-batch-20261030/b32-27405218'}
B48 = 'segneuron-b200-batch-20261030/b48-27408932'  # full length, 256,000 patches, MIT job 27408932
V100_FULL = 'segneuron-ft-20261028/train'  # the fine-tuning of the study, Wahab job 7478595
SCRIPTS = ('segneuron-b200-batch-20261030/train_segneuron_bs.py', 'segneuron-b200-batch-20261030/run.sbatch',
           'segneuron-mit-probe-20261030/probe.sbatch')


def log(run: str) -> dict:
  return json.loads((RUNS_DIR / run / 'train_log.json').read_text())


def probe(run: str) -> dict:
  x = log(run)
  return {'gpu': x['gpu'], 'torch': x['torch'], 'batch': x['args']['batch'], 'workers': x['args']['workers'],
          'steps_done': x['steps_done'], 'valid_patches': x['valid_patches'],
          'window_steps': [x['train'][-2]['step'] + 1, x['train'][-1]['step']],
          # steps 101-200, after start-up
          'seconds_per_step': x['train'][-1]['seconds_per_step'],
          'valid_loss_200': x['validation'][-1]['valid_loss']}


def long_run(run: str) -> dict:
  x = log(run)
  v = x['validation']
  return {'gpu': x['gpu'], 'torch': x['torch'], 'batch': x['args']['batch'], 'steps_done': x['steps_done'],
          'seconds': x['seconds'], 'base_lr': x['base_lr'], 'end_lr': x['end_lr'],
          'warmup_steps': x.get('warmup_steps', 0), 'valid_patches': x['valid_patches'],
          'median_seconds_per_step': statistics.median(t['seconds_per_step'] for t in x['train']),
          'max_memory_gb': max(t['max_memory_gb'] for t in x['train']),
          'patches': x['steps_done'] * x['args']['batch'],
          'validation': [[t['step'], t['valid_loss']] for t in v],
          'final_valid_loss': v[-1]['valid_loss'], 'final_diff_at_or_above_0.5': v[-1]['diff_at_or_above_0.5']}


def build() -> dict:
  probes = {k: probe(r) for k, r in PROBES.items()}
  batch = {str(b): long_run(r) for b, r in BATCH.items()}
  for b, r in batch.items():
    r['ms_per_patch'] = r['median_seconds_per_step'] / int(b) * 1000
  b48 = long_run(B48)
  b48['ms_per_patch'] = b48['median_seconds_per_step'] / 48 * 1000
  v100 = long_run(V100_FULL)
  per_step = {k: p['seconds_per_step'] for k, p in probes.items()}
  # The B200 at batch 4 in steady state: the 6,400-step batch-4 run of the batch study.
  per_step['B200 (MIT 27405214)'] = batch['4']['median_seconds_per_step']
  losses = [p['valid_loss_200'] for p in probes.values()]
  return {
      'probe_order': list(PROBES),
      'runs': {'probes': PROBES, 'workers_16': WORKERS_16, 'batch': {str(b): r for b, r in BATCH.items()},
               'batch48': B48, 'v100_full': V100_FULL},
      'scripts_sha256': {s: data.sha256_file(RUNS_DIR / s) for s in SCRIPTS},
      # Every run trained on the study's crops: the MIT probes on a copy whose manifest is the V100 run's, the
      # B200 batch study and the Wahab A100 probe through links to those folders (as recorded in the run folders).
      'data_manifest_sha256': {r: data.sha256_file(RUNS_DIR / r / 'data' / 'manifest.json') for r in
                               ('segneuron-ft-20261028', 'segneuron-mit-probe-20261030')},
      'data_links': {r: os.readlink(RUNS_DIR / r / 'data') for r in
                     ('segneuron-b200-batch-20261030', 'segneuron-a100-probe-20261030')},
      'b200_batch4_time_from': BATCH[4],
      'logs_sha256': {r: data.sha256_file(RUNS_DIR / r / 'train_log.json') for r in
                      [*PROBES.values(), *WORKERS_16.values(), *BATCH.values(), B48, V100_FULL]},
      'probes': probes, 'workers_16': {k: probe(r) for k, r in WORKERS_16.items()},
      'seconds_per_step_batch4': per_step,
      'b200_over_v100': per_step['V100 (Wahab 7478554)'] / per_step['B200 (MIT 27405214)'],
      'valid_loss_200_spread': max(losses) - min(losses),
      'batch': batch, 'batch48': b48, 'v100_full': v100,
      'v100_full_hours': v100['seconds'] / 3600,
      'b200_batch4_projected_hours': v100['steps_done'] * per_step['B200 (MIT 27405214)'] / 3600,
  }


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out', default=str(ROOT / 'docs' / 'gpu_training_evidence.json'))
  a = p.parse_args()
  Path(a.out).write_text(json.dumps(build(), indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
  main()
