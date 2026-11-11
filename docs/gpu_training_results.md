# Fine-tuning SegNeuron on newer GPUs: results

Evidence: docs/gpu_training_evidence.json, written by examples/gpu_training_evidence.py from the
train_log.json files of the runs below (data cache `runs/`); checked by tests/test_gpu_training_evidence.py.
This page is written by examples/write_gpu_training_results.py, and the test requires it to be exactly
that script's output.

**No design was written before these runs.** They were exploratory measurements of the fine-tuning
code of docs/segneuron_finetune_design.md on other GPUs and batch sizes, recorded here afterwards from
their logs. No segmentation was scored for the probes and batch-size runs; only training speed and validation
loss on the 32 fixed crop-12 patches of the fine-tuning design.

Training data: segneuron-ft-20261028, segneuron-mit-probe-20261030 hold copies of the study's crops with the same manifest (sha256 `1a065a8771a96a4c...`); segneuron-a100-probe-20261030 links to /scratch/dwadduwa/liconn_runs/segneuron-ft-20261028/data; segneuron-b200-batch-20261030 links to segneuron-mit-probe-20261030/data.

## Seconds per training step at batch 4

Short probes with the repository's examples/train_segneuron.py (6 data-loading workers); the time is
the last logged window (steps 101-200), after start-up. The B200 row is the median over the
6,400-step batch-4 run of the batch study below, in steady state.

| GPU (cluster, job) | torch | seconds per step | V100 / this GPU | validation loss after 200 steps |
|---|---|---|---|---|
| V100 (Wahab 7478554) | 2.7.1+cu126 | 1.0843 | 1.00x | 0.70162 |
| A100 PCIe (Wahab 7491576) | 2.7.1+cu126 | 0.6795 | 1.60x | 0.70147 |
| A100 PCIe (MIT 27401098) | 2.7.1+cu126 | 0.6728 | 1.61x | 0.70173 |
| H100 (MIT 27401100) | 2.7.1+cu126 | 0.2061 | 5.26x | 0.70177 |
| B200 (MIT 27405214) | 2.7.1+cu128 | 0.1407 | 7.71x | 0.70177 |

The B200 row's seconds per step come from b4-27405215 (the batch-4 run of the batch study); its validation loss from the probe named in the row.

B200 against V100: 7.71x per step. The validation losses at the end of the probes span 0.000302 across the 5 probes. The B200 probe row's 200-step time is 0.1666 s per step (start-up still settling).

With 16 data-loading workers instead of 6: A100 PCIe (MIT 27401099) 0.6727 s per step; H100 (MIT 27401101) 0.2030 s per step.

## B200: batch size at equal data

Each run saw 25,600 training patches, with a base learning rate of 0.001 x (batch/4)^0.5, a
5% linear warm-up for batches above 4 and polynomial decay to 1e-05 over its own length (a copy of the
training script with these three options, kept in the run folder; MIT jobs 27405215, 27405216, 27405217, 27405218).

| batch | steps | median seconds per step | ms per patch | peak GPU memory (GiB) | base learning rate | final validation loss | different-neuron pairs >= 0.5 |
|---|---|---|---|---|---|---|---|
| 4 | 6,400 | 0.1407 | 35.2 | 12.8 | 0.001 | 0.4596 | 0.774 |
| 8 | 3,200 | 0.2373 | 29.7 | 24.9 | 0.001414 | 0.4751 | 0.779 |
| 16 | 1,600 | 0.4355 | 27.2 | 49.2 | 0.002 | 0.4914 | 0.781 |
| 32 | 800 | 0.8002 | 25.0 | 97.7 | 0.002828 | 0.5134 | 0.782 |

Time per patch fell 1.41x from batch 4 to batch 32; validation loss rose with batch size.

## Full-length runs

| run | GPU | batch | patches | hours | final validation loss | different-neuron pairs >= 0.5 |
|---|---|---|---|---|---|---|
| the study's fine-tuning (segneuron-ft-20261028/train) | Tesla V100-SXM2-16GB | 4 | 256,000 | 19.17 | 0.3980 | 0.717 |
| batch 48 (MIT 27408932) | NVIDIA B200 | 48 | 256,032 | 1.82 | 0.4248 | 0.754 |

A full batch-4 run on a B200 at its steady-state step time would take about 2.50 h (64,000 steps); this is a projection, not a run.

## Disclosures

- One run per setting; run-to-run spread is not measured. The probes ran on whichever node Slurm
  assigned; the B200 step time comes from a different, longer run than the B200 probe.
- The batch study compares batch sizes under one learning-rate rule (square-root scaling with warm-up),
  not re-tuned per batch size, so it does not show that large batches cannot match batch 4.
- Validation loss is not segmentation accuracy; the probes and batch-size runs were not scored on crop 12
  or any other crop. The study's V100 fine-tuning, listed above for comparison, is the model the study used;
  it was scored in docs/segneuron_finetune_results.md.
- PyTorch builds, as logged: 2.7.1+cu126 (V100 (Wahab 7478554), A100 PCIe (Wahab 7491576), A100 PCIe (MIT 27401098), H100 (MIT 27401100)); 2.7.1+cu128 (B200 (MIT 27405214)); the batch study 2.7.1+cu128.
