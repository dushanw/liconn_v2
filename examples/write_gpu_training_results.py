"""Write docs/gpu_training_results.md from docs/gpu_training_evidence.json.

  write_gpu_training_results.py [--out docs/gpu_training_results.md]

Every number on the page is formatted from the evidence (examples/gpu_training_evidence.py, which reads
only the training logs). tests/test_gpu_training_evidence.py requires the page to equal render().
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / 'docs' / 'gpu_training_evidence.json'
PAGE = ROOT / 'docs' / 'gpu_training_results.md'


def render(e: dict) -> str:
  """The page's full text from the evidence e."""
  per = e['seconds_per_step_batch4']
  v100 = 'V100 (Wahab 7478554)'
  b200 = 'B200 (MIT 27405214)'
  batch = e['batch']
  b48, full = e['batch48'], e['v100_full']
  sizes = sorted(batch, key=int)
  job = {r: r.rsplit('-', 1)[-1] for r in e['runs']['batch'].values()}
  jobs = [job[e['runs']['batch'][b]] for b in sizes]
  workers = sorted({p['workers'] for p in e['probes'].values()})
  workers16 = sorted({p['workers'] for p in e['workers_16'].values()})
  patches = sorted({p['valid_patches'] for p in e['probes'].values()})
  win = sorted({tuple(p['window_steps']) for p in e['probes'].values()})
  probe_steps = ', '.join(str(x) for x in sorted({p['steps_done'] for p in e['probes'].values()}))
  builds = {}
  for k in e['probe_order']:
    p = e['probes'][k]
    builds.setdefault(p['torch'], []).append(k)
  L = []
  w = L.append
  w('# Fine-tuning SegNeuron on newer GPUs: results')
  w('')
  w('Evidence: docs/gpu_training_evidence.json, written by examples/gpu_training_evidence.py from the')
  w('train_log.json files of the runs below (data cache `runs/`); checked by tests/test_gpu_training_evidence.py.')
  w('This page is written by examples/write_gpu_training_results.py, and the test requires it to be exactly')
  w('that script\'s output.')
  w('')
  w('**No design was written before these runs.** They were exploratory measurements of the fine-tuning')
  w('code of docs/segneuron_finetune_design.md on other GPUs and batch sizes, recorded here afterwards from')
  w('their logs. No segmentation was scored for the probes and batch-size runs; only training speed and validation')
  w(f"loss on the {', '.join(map(str, patches))} fixed crop-12 patches of the fine-tuning design.")
  w('')
  links = '; '.join(f"{r} links to {t.rsplit('/runs/', 1)[-1] if '/runs/' in t else t}" for r, t in
                    sorted(e['data_links'].items()))
  manifests = set(e['data_manifest_sha256'].values())
  w(f"Training data: {', '.join(sorted(e['data_manifest_sha256']))} hold copies of the study's crops with "
    f"{'the same' if len(manifests) == 1 else 'different'} manifest (sha256 `{sorted(manifests)[0][:16]}...`); "
    f'{links}.')
  w('')
  w('## Seconds per training step at batch 4')
  w('')
  w(f"Short probes with the repository's examples/train_segneuron.py ({', '.join(map(str, workers))} data-loading "
    'workers); the time is')
  w(f"the last logged window (steps {', '.join(f'{a}-{b}' for a, b in win)}), after start-up. The B200 row is the "
    'median over the')
  w(f"{batch['4']['steps_done']:,}-step batch-4 run of the batch study below, in steady state.")
  w('')
  w(f'| GPU (cluster, job) | torch | seconds per step | V100 / this GPU | validation loss after {probe_steps} steps |')
  w('|---|---|---|---|---|')
  for k in e['probe_order']:
    p = e['probes'][k]
    w(f"| {k} | {p['torch']} | {per[k]:.4f} | {per[v100] / per[k]:.2f}x | {p['valid_loss_200']:.5f} |")
  w('')
  w(f"The B200 row's seconds per step come from {e['b200_batch4_time_from'].rsplit('/', 1)[-1]} (the batch-4 run of "
    'the batch study); its validation loss from the probe named in the row.')
  w('')
  w(f"B200 against V100: {e['b200_over_v100']:.2f}x per step. The validation losses at the end of the probes span "
    f"{e['valid_loss_200_spread']:.6f} across the {len(e['probes'])} probes. The B200 probe row's {probe_steps}-step time is "
    f"{e['probes'][b200]['seconds_per_step']:.4f} s per step (start-up still settling).")
  w('')
  w(f"With {', '.join(map(str, workers16))} data-loading workers instead of {', '.join(map(str, workers))}: "
    + '; '.join(f"{k} {p['seconds_per_step']:.4f} s per step" for k, p in e['workers_16'].items()) + '.')
  w('')
  w('## B200: batch size at equal data')
  w('')
  base = batch[sizes[0]]['base_lr']
  warm = sorted({round(batch[b]['warmup_steps'] / batch[b]['steps_done'], 4) for b in sizes[1:]})
  ends = sorted({batch[b]['end_lr'] for b in sizes})
  w(f"Each run saw {batch[sizes[0]]['patches']:,} training patches, with a base learning rate of {base:g} x "
    f"(batch/{sizes[0]})^0.5, a")
  w(f"{', '.join(f'{100 * x:g}%' for x in warm)} linear warm-up for batches above {sizes[0]} and polynomial decay to "
    f"{', '.join(f'{x:g}' for x in ends)} over its own length (a copy of the")
  w(f"training script with these three options, kept in the run folder; MIT jobs {', '.join(jobs)}).")
  w('')
  w('| batch | steps | median seconds per step | ms per patch | peak GPU memory (GiB) | base learning rate '
    '| final validation loss | different-neuron pairs >= 0.5 |')
  w('|---|---|---|---|---|---|---|---|')
  for b in sizes:
    r = batch[b]
    w(f"| {b} | {r['steps_done']:,} | {r['median_seconds_per_step']:.4f} | {r['ms_per_patch']:.1f} | "
      f"{r['max_memory_gb']:.1f} | {r['base_lr']:.4g} | {r['final_valid_loss']:.4f} | "
      f"{r['final_diff_at_or_above_0.5']:.3f} |")
  w('')
  w(f"Time per patch fell {batch[sizes[0]]['ms_per_patch'] / batch[sizes[-1]]['ms_per_patch']:.2f}x from batch "
    f"{sizes[0]} to batch {sizes[-1]}; validation loss rose with batch size.")
  w('')
  w('## Full-length runs')
  w('')
  w('| run | GPU | batch | patches | hours | final validation loss | different-neuron pairs >= 0.5 |')
  w('|---|---|---|---|---|---|---|')
  w(f"| the study's fine-tuning ({e['runs']['v100_full']}) | {full['gpu']} | {full['batch']} | {full['patches']:,} | "
    f"{e['v100_full_hours']:.2f} | {full['final_valid_loss']:.4f} | {full['final_diff_at_or_above_0.5']:.3f} |")
  w(f"| batch {b48['batch']} (MIT {e['runs']['batch48'].rsplit('-', 1)[-1]}) | {b48['gpu']} | {b48['batch']} | "
    f"{b48['patches']:,} | {b48['seconds'] / 3600:.2f} | {b48['final_valid_loss']:.4f} | "
    f"{b48['final_diff_at_or_above_0.5']:.3f} |")
  w('')
  w(f"A full batch-4 run on a B200 at its steady-state step time would take about "
    f"{e['b200_batch4_projected_hours']:.2f} h ({full['steps_done']:,} steps); this is a projection, not a run.")
  w('')
  w('## Disclosures')
  w('')
  w('- One run per setting; run-to-run spread is not measured. The probes ran on whichever node Slurm')
  w('  assigned; the B200 step time comes from a different, longer run than the B200 probe.')
  w('- The batch study compares batch sizes under one learning-rate rule (square-root scaling with warm-up),')
  w('  not re-tuned per batch size, so it does not show that large batches cannot match batch 4.')
  w('- Validation loss is not segmentation accuracy; the probes and batch-size runs were not scored on crop 12')
  w('  or any other crop. The study\'s V100 fine-tuning, listed above for comparison, is the model the study used;')
  w('  it was scored in docs/segneuron_finetune_results.md.')
  w('- PyTorch builds, as logged: ' + '; '.join(f"{t} ({', '.join(ks)})" for t, ks in sorted(builds.items()))
    + f"; the batch study {', '.join(sorted({batch[b]['torch'] for b in sizes}))}.")
  return '\n'.join(L) + '\n'


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out', default=str(PAGE))
  a = p.parse_args()
  Path(a.out).write_text(render(json.loads(EVIDENCE.read_text())))


if __name__ == '__main__':
  main()
