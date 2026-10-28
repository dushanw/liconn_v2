# Single-pass ViT pilot: results (stopped at the tuning crop)

Design: `docs/vitseg_design.md` (revision 3). Evidence: `docs/vitseg_evidence.json`, checked by
`tests/test_vitseg_evidence.py` against the run files in the data cache (`runs/`).

## Summary

- **Status: stopped by the owner before the test crops (2026-10-26).** Crops 13-40 were never
  segmented, so there is no verdict under the pass rule. No ViT was run or scored on them, so
  they can still be used once by a later model. (Their labels were not entirely unseen before
  this pilot: the design's crop-ceiling note compared labels and FFN traces on 37 seeds in crops
  13-15.)
- **Training run 2 stopped itself** at step 67,344 of 170,000. From step 66,247 (index 66,246 in
  the log's 0-based `skipped_steps`; other step numbers here count completed steps) almost every
  step had a non-finite gradient, and the 200-in-a-row rule ended the run. The selected model is the
  step-66,000 checkpoint (validation loss 0.268), saved before the failure. The log marks the run
  `unstable`.
- **On crop 12 (the tuning crop) the model is far from FFN:** mean F1 0.531 over the 457 large
  objects at the chosen merge threshold 0.5 (recall 0.502, precision 0.826). For comparison, FFN
  (fp16_head32) averages F1 0.894 (recall 0.850, precision 0.985) over its 300 many-seed seeds in
  crops 0-12. These are different populations (paired only for object 346093, F1 0.646 against
  FFN's 0.869), so the difference is context, not a measured gap; see "The FFN reference".
- **Background works; most lost recall is in other segments.** On average 50% of a neuron is in the seed's
  segment, 40% in other segments and 9% is called background. Neurons break into a median of 5
  pieces holding at least 1% each.
- **Neither output alone is the whole problem.** With the labels' neuron mask and the model's
  affinities, F1 rises to 0.744; with the model's mask and the labels' affinities, to 0.778. With
  both from the labels the same pipeline reaches 0.924 (oracle check in the design). These are two
  particular substitutions, not bounds: the mask and the affinities act in series, the first
  hybrid peaks at the lowest threshold tried, and the second replaces affinities only between two
  labelled neurons. They show that neither substitution alone came close to the oracle, not how
  much each output contributes or how far improving one could go.

## What was run

| step | job | input | output (data cache `runs/`) |
|---|---|---|---|
| training run 2 | Slurm 7448381, `gpu`, 14 h 29 min | crops 0-11; validation blocks of crop 12 | `vitseg-train-r3-20261025/` (`train_log.json`, `params.msgpack` = step 66,000) |
| tuning | Slurm 7451650, `timed-gpu`, 36 min | crop 12, `params.msgpack` | `vitseg-tune-r3-s66000-20261026/vitseg_tuning.json` |
| breakdown | Slurm 7452001, `timed-gpu`, 44 min | crop 12, `params.msgpack` | `vitseg-breakdown-s66000-20261026/breakdown.json` |
| 3D meshes | Slurm 7452015, `timed-gpu`, 35 min | crop 12, `params.msgpack` | `vitseg-breakdown-s66000-20261026/vit_scenes.json`, `meshes/` |

All four ran from the staged source `/scratch/dwadduwa/liconn_v2_m18d`; its manifest
(`vitseg-train-r3-20261025/source_sha256.txt`) matches the repository's `liconn_v2/vitseg.py`,
`examples/train_vitseg.py`, `examples/run_vitseg.py`, `examples/check_vitseg.py`,
`tests/test_vitseg.py` and `cluster/wahab/vitseg.sbatch` (checked by the evidence test). The
breakdown and mesh scripts are exploratory and live with their outputs (`breakdown.py`,
`meshes.py`). The tuning, the breakdown (network outputs, per-object accounting, hybrids) and the
meshes all used the float outputs (checked by the evidence test: the breakdown's only use of
`uint8` is writing the export, the mesh script never reads it, and the two jobs give the same mean
F1 to all digits, 0.5313875, within 5e-7 of the tuning run's 0.5313870; the cause of the small difference
was not isolated; run-to-run GPU non-determinism is a likely but untested explanation). Only the export `aff_uint8.npy` (whole-crop outputs rounded to 1/255) is rounded;
it feeds the demo page's slice example and the local Neuroglancer view, not any number in this
document. (The task contract's limits sentence, "the breakdown and 3D/viewer exports read model
outputs rounded to 1/255 where stated", was written imprecisely: only the exports are rounded.)

## Training

- Validation loss (8 fixed crop-12 blocks): 0.395 at step 2,000, 0.290 at 20,000, 0.278 at
  50,000, 0.268 at 66,000 (the minimum), then 0.515 at 67,344 after the failure. It improved by
  only about 0.02 over the last 46,000 steps.
- 0 skipped steps in the first 66,246; 972 skipped steps in all (0.57% of the planned 170,000).
  Training loss per 100 steps went from about 0.26 to about 1.0 within 300 steps of the failure.
- Speed: 0.77 s per step on a V100.
- Cause of the failure: **not diagnosed.** Losses stayed finite while gradients were not, which
  points to a float16 overflow in the backward pass under the static loss scale (x1,024); the
  revision-3 LayerNorms and clipping held for 66,000 steps but did not prevent it. The learning
  rate was still about 2e-4 (cosine schedule at 39%). The 972 skipped steps were not all
  consecutive, so some finite updates landed between them; once the parameters were in a state
  where nearly every batch overflowed, the static scale could not back off. Hypotheses, untested:
  attention and the residual stream run in float16 without float32 accumulation; and
  `balanced_bce` gives each negative pair weight 0.5 / (number of negatives), so in a batch with
  few neuron-neuron contacts (1.1% of boundary pairs) a channel's whole negative gradient lands on
  a handful of voxels. The loss and the output head are float32, and that gradient is bounded
  there (at most about 0.5 / 7 x 1,024 = 73 per logit), so it cannot overflow by itself; the
  hypothesis is that such concentrated gradients grow past the float16 range in the backward pass
  through the float16 decoder and transformer.

## Crop 12: tuning

Threshold table (`vitseg_tuning.json`): mean F1 rises from 0.097 at t = 0.05 to its maximum
0.531 at t = 0.50 and falls to 0.298 at t = 0.95. Merging harder does not recover recall without
merges: at t = 0.75 recall is 0.348 at precision 0.953; at t = 0.05 recall is 0.854 at precision
0.113.

- Background against label 0 (voxel level): 41.1% predicted, 41.0% labelled; precision 0.886,
  recall 0.889. 7 of the 457 seeds fall in predicted background.
- Time for the crop: network 32 s (V100), watershed 267 s, agglomeration 12 s (CPU); 59,303
  fragments.

## Crop 12: where the accuracy goes (breakdown)

**Network outputs against the labels.**

- Neuron channel: precision 0.923, recall 0.920. The errors sit on boundaries: 18.2% of neuron
  voxels next to a label change are missed (1.9-2.8% further in), and 14.0% of background voxels
  next to a label change are called neuron. Deep background (4+ voxels from any change, 0.8 M
  voxels) is called neuron 48% of the time.
- Affinities on neuron-neuron pairs: 14.3-14.6% of same-neuron nearest-neighbour pairs are below
  0.5 (mean 0.81), and 11.7-12.5% of different-neuron pairs are 0.5 or above. The long-range
  channels are better (7.2-7.9% each way). 14.0% of labelled interior voxels predicted as neuron
  are unconfident (boundary map >= 0.5); the watershed breaks neurons there.

**Per object at t = 0.5** (means over the 457 objects):

| | |
|---|---|
| in the seed's segment (recall) | 50.2% |
| in other segments | 40.5% |
| called background | 9.3% |
| pieces holding >= 1% of the object | 5.7 (median 5) |

"Other segments" is not all model error. The seed's 6-connected piece of the labelled object
holds on average 88.7% of the object, because objects leave and re-enter the crop (oracle check in
the design). A segmentation that stays inside the labelled object cannot reach the other 11.3%.
That is a reference point, not an unavoidable loss: a predicted segment can reach other pieces
through voxels falsely called neuron (and precision ignores label-0 voxels), so how much of the
ViT's 40.5% is this structural part was not measured. And a single missed neuron voxel across a
thin neck cuts the object in two; the voxel accounting books the far side as "other segments" although the cause is the neuron channel.

By thickness (median distance from a label change, terciles): F1 0.468 for the thinnest 161
objects (<= 24 nm), 0.517 for the middle 195, 0.660 for the thickest 101. Split edges occur
at similar rates along z, y and x (1.06-1.11% of same-neuron pairs each). This is a descriptive
aggregate: it does not show a z-specific excess, but it does not by itself rule out effects of
anisotropy that differ by object geometry or orientation.

**Hybrids** (each scored over t = 0.3-0.9):

| outputs | best t | recall | precision | F1 |
|---|---|---|---|---|
| model (tuning run) | 0.50 | 0.502 | 0.826 | 0.531 |
| labels' neuron mask, model's affinities | 0.30 | 0.657 | 0.979 | 0.744 |
| model's neuron mask, labels' affinities | 0.80 | 0.696 | 0.962 | 0.778 |
| labels for both (oracle, design doc) | 0.80 | 0.887 | 0.997 | 0.924 |

The first hybrid's best is at the lowest threshold tried, so it may rise slightly lower down.
How to read the hybrids (they are not additive; the mask and the affinities act in series):

- The labels' mask hands the pipeline 98.9% of crop 12's boundaries (neuron-background) for free,
  and hides every affinity on an edge touching true background, which training never constrained.
  That it still reaches only 0.744 means the affinities themselves split neuron interiors.
- In the second hybrid the labels' affinities replace the model's only on pairs of two labelled
  neurons (the training mask); on edges touching label 0 the model's values are kept, so no
  label-derived wall is added around false neuron voxels.
- The mask matters more than its 92% voxel accuracy suggests: its errors sit on boundaries, where
  false neuron voxels bridge neighbours (precision 0.83 to 0.98 with the labels' mask) and missed
  ones cut thin neurites.

## The FFN reference: what differs

The 0.89 is context, not a paired comparison:

- Different objects: FFN's 300 seeds in crops 0-12 (one in crop 12) against the ViT's 457 crop-12
  objects; paired only for object 346093.
- Crop 12 chose the ViT's checkpoint and threshold, so the ViT figure is optimistic.
- Precision (`benchmark.score`) ignores label-0 voxels in a segment. FFN's segments contain
  label-0 voxels at no cost (mean 13.6% over its 300 many-seed seeds; quartiles 10.6% and 16.0%) and can pass through background to bridge thin necks; the ViT
  was designed to output background and its segments cannot cross it.
- FFN can bridge small gaps beyond the 6-connected ceiling (it did on 10 of 37 seeds checked).
- FFN's network was trained by the volume's authors on proofread data from this volume, possibly
  including crop 12; the ViT was trained from scratch on crops 0-11.

## Examples (crop 12, picked by F1 rank)

From `vit_scenes.json`; shown in 3D on the demo page (wadduwagelab.com/demos/liconn-ffn/#vit-result).

| object | rank (1 = worst) | recall | precision | F1 | pieces >= 1% |
|---|---|---|---|---|---|
| 90th percentile | 411 | 0.906 | 0.925 | 0.915 | 4 |
| median | 229 | 0.389 | 1.000 | 0.560 | 11 |
| 10th percentile | 46 | 0.044 | 1.000 | 0.084 | 8 |
| 346093 (the crop-12 FFN seed) | 283 | 0.523 | 0.845 | 0.646 | 5 |
| 346093, FFN fp16_head32 | | 0.768 | 0.9995 | 0.869 | |

## What is known and what is not

- Known: on the tuning crop this checkpoint is far below FFN, and most of the lost recall is in
  other segments rather than in predicted background; replacing either the neuron channel or the
  affinities by the labels (as done here) gave F1 0.744 and 0.778, short of the oracle's 0.92.
- Not known: how a fully trained model (170,000 steps) would do. The validation loss had nearly
  flattened, but the learning rate was still about two thirds of its peak (the cosine tail never
  came), and the validation loss also averaged the three long-range channels that segmentation
  never reads, so it is an imperfect guide. A large gain is unlikely but not ruled out. Not known whether a CNN backbone, a
  larger model or more data would close the gap (the SegNeuron paper reports CNNs far ahead of
  ViT encoders such as UNETR for affinity prediction). No result exists on the test crops.
- Crop 12 chose the checkpoint and the threshold, so its numbers are optimistic for this model.

## How to resume

Files on Wahab: `/scratch/dwadduwa/liconn_runs/vitseg-train-r3-20261025/`.

- `state.msgpack` is the state **at step 67,344, after the failure** (broken parameters and
  optimiser state). Do not resume from it. `train_vitseg.py` also refuses to resume a run whose log
  has `stopped` set.
- `params.msgpack` (= `best_params.msgpack`, step 66,000) holds the model weights only
  (`flax.serialization.to_bytes(params)`), without the optimiser state; continuing from it needs a
  fresh optimiser state (and a warm-up). `train_vitseg.py` has no option for this: it starts from
  random weights at step 0 or resumes `state.msgpack` (a dict of params, optimiser state and
  step). Starting from these weights needs a small code change (load the weights, set the first
  step).
- Batches depend only on (`RNG_SEED`, step). A run that starts from these weights at step 0 with
  the same seed sees batches 0-65,999 again; choose the start step or the seed deliberately.
- `breakdown.py` and `meshes.py` are not in the repository; they live with their outputs in
  `vitseg-breakdown-s66000-20261026/` (data cache and Wahab) and are run as
  `python breakdown.py PARAMS OUT_DIR` (likewise `meshes.py`) from the staged source tree.

Steps:

1. Fix the float16 failure first: a dynamic loss scale (halve on overflow and retry), float32
   accumulation in attention and LayerNorm (bfloat16 is an option only on A100s; these runs used
   V100s, which lack it), and possibly a floor on the negative count in `balanced_bce` (which
   reweights the classes, so the 0.5 thresholds and the merge threshold must be re-tuned). An
   attempt to reproduce the failure by loading the step-66,000 weights and replaying batches
   66,000-66,300 is approximate only: the step-66,000 optimiser state (Adam moments and step
   count, which drives the learning-rate schedule) was not saved, so the exact trajectory cannot
   be replayed.
2. Cheap tests on the existing step-66,000 model, no training: a softer boundary map (second-lowest
   or mean incident edge instead of the lowest), and a neuron mask dilated by one voxel or a lower
   neuron threshold during segmentation, to see how much of the splitting is post-processing.
3. Then retrain (or continue from step 66,000 with a new optimiser state), select checkpoints on a
   validation measure that uses only the nearest-neighbour channels and the neuron channel, and
   repeat the tuning (crop 12) and the breakdown (`breakdown.py`) before reading the test crops
   once. No ViT model has been run or scored on the test crops 13-40.
4. Targets suggested by the breakdown: boundary-voxel accuracy of the neuron channel, and
   same-neuron affinities inside thin neurites.

## Disclosed

- Run 1 (Slurm 7433740) diverged at step 22,777 and produced no model (design doc).
- This is a stopped pilot, not a negative result for single-pass segmentation in general.
- The FFN comparison figure is on different objects (FFN's 300 seeds in crops 0-12, of which one
  is in crop 12); the per-object pairing exists only for object 346093.
