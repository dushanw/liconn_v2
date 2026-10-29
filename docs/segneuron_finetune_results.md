# SegNeuron fine-tuned on LICONN crops 0-11: results on crop 12

Design: `docs/segneuron_finetune_design.md`. Evidence: `docs/segneuron_finetune_evidence.json`,
checked by `tests/test_segneuron_ft_evidence.py` against the run files in the data cache
(`runs/segneuron-ft-20261028/`). Zero-shot reference: `docs/segneuron_results.md`.

## Summary

- **Fine-tuning taught the network much of the boundary between touching neurons.** On crop 12 the
  AUC of same-neuron against different-neuron pairs rose from 0.67-0.69 (zero-shot) to 0.85 in
  each channel. The recall our pipeline loses to other segments fell from 37% to 15% of each
  object, and the recall lost to background from 8.5% to 5.3%. This is the threshold-free reading
  the design asked for first.
- **Best mean F1 over the 457 tuning objects rose from 0.500 to 0.823 with our pipeline**
  (recall 0.801, precision 0.926; threshold 0.6). FRMC with predicted background removed reached
  0.763 (beta 0.25), up from 0.530. The ViT trained on the same crops scored 0.531, and the
  pipeline with label-derived outputs (the oracle) 0.924 on the same objects. Paired by object,
  419 of 457 objects scored higher than zero-shot and 38 lower. The median change was +0.28 and
  the largest drop 0.86.
- **Crop 12 chose the threshold and beta, so these numbers are optimistic and are not test
  results.** Crop 12 also chose the checkpoint, but the lowest validation loss was at the last
  step, so the choice was the same as "train to the end". The test crops 13-40 were not read.
- **The boundaries are still not sharp.** 76-78% of the pairs that straddle two different neurons
  still get affinity >= 0.5 (87% zero-shot), though their median fell from 0.95 to 0.76. The
  segmentation works at threshold 0.6 because same-neuron pairs sit higher (median 0.97). Which
  part of the pipeline turned the AUC gain into the F1 gain was not isolated.
- **One run of 64,000 steps (19.2 h on one V100), no failures.** Validation loss fell from 2.05
  (the released model) to 0.52 by step 1,000, and then slowly to 0.398 at step 64,000. It was
  still falling at the end. Whether a longer run would help was not tested.

## What was run

| step | job | output (data cache `runs/segneuron-ft-20261028/`) |
|---|---|---|
| timing probe, batch 8 | Slurm 7478237, `timed-gpu` | out of memory on the 16 GB V100 (not in the evidence) |
| timing probe, batch 4, 200 steps | Slurm 7478554, `timed-gpu`, 4.5 min | `probe-b4/train_log.json` (12.75 GB, 1.09 s per step) |
| training, then the authors' `inference.py` with `best.ckpt` on crop 12 | Slurm 7478595, `gpu` (V100), 19 h 15 min | `train/train_log.json`, `native/infer/inference.json`, `7478595.out`, `env-train.txt`, `env-infer.txt`, `gpu.txt`, `data/manifest.json` |
| FRMC at 7 betas, our pipeline, scoring | Slurm 7478596, `main` (CPU), 50 min | `segneuron_ft_crop12.json`, `7478596.out`, `env-frmc.txt` |

- Both jobs ran from the staged source `/scratch/dwadduwa/liconn_v2_segneuron_ft`. Its manifest
  (`source_sha256.txt`) matches the repository's fine-tuning files and the zero-shot functions
  they reuse (checked by the evidence test).
- The run used SegNeuron commit `ea6f0c96`, unmodified. Each job checked the released
  checkpoint's sha256 before any work. The fine-tuned checkpoint's sha256 is in
  `inference.json` and the evidence.
- Training settings: batch 4 (batch 8 did not fit) and 64,000 iterations, chosen from the
  probe's 1.09 s per step to fill about 20 h. The learning rate went from 1e-3 to 1e-5 (power
  1.5), with seed 666. Training used crops 0-11, and validation used 32 fixed patches of crop 12
  every 1,000 steps, 65 times in all. GPU memory peaked at 12.75 GB. The run used torch
  2.7.1+cu126 on a Tesla V100-SXM2-16GB.
- The large arrays stay on Wahab under `/scratch/dwadduwa/liconn_runs/segneuron-ft-20261028/`:
  training crops, checkpoints, network outputs and FRMC segmentations. Only records were copied.

## Validation during training (32 fixed crop-12 patches)

| step | validation loss | same-neuron pairs < 0.5 | different-neuron pairs >= 0.5 |
|---|---|---|---|
| 0 (released) | 2.053 | 7.4% | 82.8% |
| 1,000 | 0.521 | 7.6% | 75.0% |
| 2,000 | 0.486 | 7.7% | 73.9% |
| 10,000 | 0.423 | 5.9% | 76.0% |
| 20,000 | 0.411 | 6.2% | 72.4% |
| 32,000 | 0.406 | 6.1% | 71.6% |
| 48,000 | 0.400 | 5.7% | 72.4% |
| 64,000 (chosen) | 0.398 | 5.7% | 71.7% |

The pair columns are diagnostics on the patches only. They did not choose anything.

## Network outputs against the labels (crop 12, native)

| | zero-shot | fine-tuned |
|---|---|---|
| foreground AUC | 0.917 | 0.976 |
| foreground precision / recall at 0.5 | 0.814 / 0.935 | 0.929 / 0.953 |
| predicted foreground at 0.5 (labelled: 59.0%) | 67.8% | 60.5% |
| foreground: median on unlabelled voxels | 0.226 | 0.008 |
| AUC same vs different neuron (z, y, x) | 0.666, 0.688, 0.688 | 0.851, 0.852, 0.854 |
| AUC same neuron vs neuron-background (z, y, x) | 0.842, 0.878, 0.866 | 0.934, 0.943, 0.931 |
| same-neuron pairs below 0.5 (z, y, x) | 7.8%, 7.2%, 7.3% | 4.9%, 4.9%, 4.9% |
| different-neuron pairs at or above 0.5 (z, y, x) | 87.4%, 86.6%, 85.7% | 77.6%, 77.7%, 75.9% |
| median affinity, different-neuron pairs (z, y, x) | 0.954, 0.948, 0.947 | 0.764, 0.761, 0.754 |

## Segmentation (457 tuning objects)

| | zero-shot native | fine-tuned | ViT (crops 0-11) |
|---|---|---|---|
| ours: threshold, mean F1 (recall / precision) | 0.8, 0.500 (0.544 / 0.674) | 0.6, 0.823 (0.801 / 0.926) | 0.5, 0.531 (0.502 / 0.826) |
| ours: recall lost to other segments / background | 37.1% / 8.5% | 14.6% / 5.3% | 40.5% / 9.3% |
| ours: mean pieces >= 1% (median) | 4.9 (5) | 2.2 (2) | 5.7 (5) |
| ours: seeds in predicted background | 4 | 3 | |
| ours: F1 at threshold 0.5 | 0.031 | 0.808 | 0.531 |
| FRMC beta (chosen by masked F1) | 0.4 | 0.25 | |
| FRMC masked: mean F1 (recall / precision) | 0.530 (0.534 / 0.717) | 0.763 (0.690 / 0.951) | |
| FRMC as released: mean F1 (recall / precision) | 0.553 (0.572 / 0.721) | 0.785 (0.727 / 0.951) | |

FRMC by beta, fine-tuned (segments; masked F1 / as-released F1):

| beta | 0.1 | 0.25 (authors' default) | 0.4 | 0.5 | 0.6 | 0.75 | 0.9 |
|---|---|---|---|---|---|---|---|
| | 2,173; 0.755 / 0.777 | 9,042; 0.763 / 0.785 | 53,935; 0.609 / 0.626 | 118,314; 0.476 / 0.487 | 171,752; 0.295 / 0.301 | 214,558; 0.036 / 0.037 | 224,317; 0.021 / 0.022 |

Our pipeline is now much less sensitive to its threshold than zero-shot. F1 is 0.78 or more from
0.45 to 0.75, against a best of 0.50 zero-shot. FRMC still depends strongly on beta. Our pipeline
beat FRMC by 0.06 masked F1. As in the zero-shot run, both use fixed foreground and watershed
settings, and no other post-processing was tried.

Each row's numbers come from the checked zero-shot and ViT evidence (`docs/segneuron_evidence.json`,
`docs/vitseg_evidence.json`), on the same 457 objects and with the same scoring.

## Reading for the next decision

- The question in the design: does supervised fine-tuning on crops 0-11 teach the boundary
  between touching neurons? On crop 12 the answer is largely yes. The threshold-free
  same-vs-different AUC and the recall lost to other segments both moved by a large margin. The
  F1 gain is not just the foreground fraction moving towards the labelled 59%.
- The remaining loss is mostly to other segments (14.6%, against 5.3% to background), and most
  different-neuron pairs are still above 0.5. Separation between neurons is still the main
  weakness, at a smaller size.
- Crop 12 is the only held-out crop that has been scored. Its numbers chose the threshold and
  beta, so they overstate what a fresh crop would give. The test crops are the way to measure
  that.
- The design does not compare against FFN, and the numbers here do not support one. FFN's
  reference F1 (0.894, `docs/vitseg_results.md`) is over different objects and seeds, and FFN's
  network was trained on proofread data from this volume.

## Disclosed

- Crop 12 chose the threshold and beta. It also chose the checkpoint, which was the last one.
  The ViT's reference number is on the crop it was validated on. All three are optimistic.
- One training run with one seed. GPU training is not bitwise reproducible (cuDNN), so a repeat
  would give somewhat different numbers. The run-to-run spread was not measured.
- The learning rate (1e-3 to 1e-5) and the run length are choices, not tuned values. The authors'
  dataset-mixing augmentations were left out (design).
- Label 0 is background in training and scoring. Unproofread neurites inside label 0 are taught
  as background, and precision ignores label-0 voxels (design). The predicted foreground fell to
  60.5% against 59.0% labelled. Recall lost to background fell, so this did not cost the
  labelled objects recall on average.
- Crop 12 is at least 500 voxels from each training crop, but a neuron can pass through both.
- The batch-8 probe that ran out of memory is reported from its log. Its files are not in the
  evidence.
