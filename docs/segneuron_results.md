# SegNeuron zero-shot on crop 12: results

Design: `docs/segneuron_design.md`. Evidence: `docs/segneuron_evidence.json`, checked by
`tests/test_segneuron_evidence.py` against the run files in the data cache
(`runs/segneuron-zeroshot-20261026/`).

## Summary

- **Zero-shot SegNeuron is on par with our trained ViT on crop 12, and far from the oracle.**
  Best mean F1 over the 457 tuning objects at native resolution: 0.553 for FRMC as released
  (beta 0.4), 0.530 for FRMC with predicted background removed, and 0.500 for our pipeline
  (threshold 0.8). For comparison, on the same objects and scoring, the ViT trained on crops 0-11
  scored 0.531 and the pipeline with label-derived outputs (the oracle) 0.924. Crop 12 chose the
  threshold and beta, so all of these are optimistic.
- **The outputs separate neuron from background much better than neuron from neuron.** At
  native resolution the foreground output separates labelled from unlabelled voxels with AUC
  0.917 (precision 0.814, recall 0.935 at 0.5). But 86-87% of pairs that straddle two different
  neurons still get affinity >= 0.5 (median 0.95), and the AUC of same-neuron against
  different-neuron pairs is only 0.67-0.69 per channel, which is a merge risk. Our pipeline
  loses 37% of each object's recall to other segments and 8% to background, close to the ViT's
  split (40% and 9%). The accounting cannot say what causes the loss to other segments
  (affinities, the foreground cut, watershed fragmentation, or object pieces that are separate
  within the crop); no experiment here isolates it.
- **Upsampling 2x lowered the threshold-free scores and the best F1** (foreground AUC 0.871,
  same/different AUC 0.64-0.67, best F1 0.459 ours and 0.442 FRMC as released), though not every
  measure: foreground recall, the same-neuron pairs below 0.5 and our recall lost to background
  were slightly better. Native gave the better zero-shot result; which resolution is the better
  starting point for fine-tuning was not tested.
- **Both post-processing methods are very sensitive to their merge setting on these outputs.**
  Our pipeline at the ViT's threshold 0.5 gives F1 0.031 (native); its best threshold is 0.8.
  FRMC's F1 ranges from 0.02 to 0.55 over the seven betas (native; 0.002 to 0.44 for xy2). This is the behaviour of the pipelines
  on these outputs; whether the outputs' probabilities are calibrated was not measured (no
  comparison of predicted probabilities with observed frequencies was made).
- No pass/fail verdict: this milestone was zero-shot on crop 12 only. The test crops 13-40 were
  not read.

## What was run

| step | job | output (data cache `runs/segneuron-zeroshot-20261026/`) |
|---|---|---|
| inputs, inference at both resolutions, xy2 to the crop grid | Slurm 7463133, `gpu` (V100), 22 min | `native/infer/inference.json`, `xy2/infer/inference.json`, `7463133.out`, `env-infer.txt`, `gpu.txt` |
| FRMC at 7 betas x 2 resolutions, our pipeline, scoring | Slurm 7463134, `main` (CPU), 14 h 13 min | `segneuron_crop12.json`, `7463134.out`, `env-frmc.txt` |

- Both jobs ran from the staged source `/scratch/dwadduwa/liconn_v2_segneuron`; its manifest
  (`source_sha256.txt`) matches the repository's `liconn_v2/segneuron.py`, `liconn_v2/vitseg.py`,
  `examples/run_segneuron.py` and `cluster/wahab/segneuron.sbatch` (checked by the evidence test).
- SegNeuron code at commit `ea6f0c96`, unmodified; the job checked the commit, a clean tree and
  the checkpoint's sha256 before any work, and both inference manifests record the same hash.
- The jobs were submitted with longer limits than the sbatch header's example commands: GPU on
  `gpu` with 8 h (header: `timed-gpu`, 2 h), CPU on `main` with 24 h (header: 12 h, which the
  14 h 13 min run would have exceeded). The header is left as it ran; the commands to use are in
  the design's "Jobs" section.
- Model time on one V100: 217 s native (2,450 tiles), 991 s xy2 (11,250 tiles).
- FRMC's run times differed widely (from the files' modification times on Wahab; not part of the
  evidence test): the 7 native betas took 32 min together; xy2 at beta 0.1 took about 4 h 40 min,
  at beta 0.25 about 7 h 10 min, and the remaining 5 betas 33 min together. Scoring took 76 min.
  The cause of the slow cases was not investigated.
- The large arrays (inputs, network outputs, FRMC segmentations) stay on Wahab under
  `/scratch/dwadduwa/liconn_runs/segneuron-zeroshot-20261026/`; only records were copied.

## Network outputs against the labels (crop 12)

| | native | xy2 |
|---|---|---|
| foreground AUC | 0.917 | 0.871 |
| foreground precision / recall at 0.5 | 0.814 / 0.935 | 0.716 / 0.945 |
| foreground: median on unlabelled voxels | 0.226 | 0.546 |
| same-neuron pairs below 0.5 (z, y, x) | 7.8%, 7.2%, 7.3% | 5.3%, 7.0%, 7.0% |
| different-neuron pairs at or above 0.5 (z, y, x) | 87.4%, 86.6%, 85.7% | 91.9%, 86.7%, 86.9% |
| AUC same vs different neuron (z, y, x) | 0.666, 0.688, 0.688 | 0.637, 0.668, 0.662 |
| AUC same neuron vs neuron-background (z, y, x) | 0.842, 0.878, 0.866 | 0.785, 0.846, 0.833 |

Labelled voxels make up 59.0% of crop 12; the model calls 67.8% (native) and 77.9% (xy2)
foreground at 0.5. Without fine-tuning, the affinities separate same-neuron pairs from
neuron-background pairs (AUC 0.84-0.88 native) much better than from different-neuron pairs
(0.67-0.69). How much of the segmentations' lost recall this explains was not measured.

## Segmentation (457 tuning objects)

| | native | xy2 |
|---|---|---|
| ours: threshold, mean F1 (recall / precision) | 0.8, 0.500 (0.544 / 0.674) | 0.8, 0.459 (0.472 / 0.662) |
| ours: recall lost to other segments / background | 37.1% / 8.5% | 47.3% / 5.5% |
| ours: mean pieces >= 1% (median) | 4.9 (5) | 6.6 (6) |
| ours: seeds in predicted background | 4 | 7 |
| FRMC beta (chosen by masked F1) | 0.4 | 0.6 |
| FRMC masked: mean F1 (recall / precision) | 0.530 (0.534 / 0.717) | 0.426 (0.390 / 0.710) |
| FRMC as released: mean F1 (recall / precision) | 0.553 (0.572 / 0.721) | 0.442 (0.411 / 0.720) |

FRMC by beta (segments; masked F1 / as released F1):

| beta | native | xy2 |
|---|---|---|
| 0.1 | 465; 0.154 / 0.158 | 59; 0.002 / 0.002 |
| 0.25 (authors' default) | 2,506; 0.433 / 0.452 | 624; 0.114 / 0.118 |
| 0.4 | 28,646; 0.530 / 0.553 | 2,103; 0.323 / 0.334 |
| 0.5 | 97,242; 0.476 / 0.495 | 9,998; 0.417 / 0.433 |
| 0.6 | 168,295; 0.397 / 0.410 | 66,787; 0.426 / 0.442 |
| 0.75 | 249,146; 0.104 / 0.106 | 230,291; 0.212 / 0.218 |
| 0.9 | 278,850; 0.020 / 0.021 | 325,781; 0.017 / 0.018 |

FRMC as released scores a little higher than masked partly because precision ignores label-0
voxels, so absorbing background costs it nothing (design, "Disclosed"); the masked numbers are
the comparable ones. The two post-processing methods are within 0.03 F1 of each other at native
resolution, so choosing between these two made little difference. That does not show that
post-processing is not limiting: both use fixed foreground and watershed settings, and no other
post-processing was tried.

## Reading for the fine-tuning decision

- Supports fine-tuning: with no LICONN data, the model already reaches the ViT's crop-12 F1 and
  separates foreground from background well (AUC 0.92), so its features carry over in part.
- The weakest measured output is the affinity between touching neurons (AUC 0.67-0.69, 86-87%
  of cross-boundary pairs above 0.5). Whether it is what limits the segmentation was not
  isolated. The ViT trained on crops 0-11 also lost most recall to other segments
  (`docs/vitseg_results.md`), so fine-tuning is not guaranteed to close the gap.
- Native resolution gave the better zero-shot F1 and AUCs and is about 4.6 times cheaper to
  infer; whether it would also fine-tune better was not tested.

## Disclosed

- Crop 12 chose our threshold and FRMC's beta, and the ViT's reference number is on the crop it
  was validated on; all three are optimistic, and none is a test result.
- The ViT was trained on crops 0-11 of this volume; SegNeuron has seen no LICONN data.
- No intensity inversion or other adaptation was tried (design); a different input polarity
  might change the zero-shot numbers.
- The FRMC run-time figures come from file modification times, not from a timer in the job.
