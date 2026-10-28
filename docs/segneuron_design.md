# SegNeuron zero-shot on crop 12: design

Owner decisions 2026-10-26 (after the ViT pilot stopped at crop 12, `docs/vitseg_results.md`):
try the pretrained SegNeuron model; post-process its outputs **both** with our pipeline and with
the authors' FRMC; run it at **both** the native resolution and x-y upsampled 2x; the first
milestone is **zero-shot on crop 12 only**. Fine-tuning on crops 0-11 is a later milestone,
decided after this one. Sources and verbatim excerpts: `docs/segneuron_sources.md`.

## What SegNeuron is

A generalist neuron segmentation model for volume EM (Zhang et al., MICCAI 2024; MIT licence).
MNet (a CNN mixing 2D and 3D convolutions, `MNet(1, kn=(32,64,96,128,256), FMU='sub')`) predicts
3 nearest-neighbour affinities and an auxiliary foreground probability, both through a sigmoid.
It was trained on 16 EM datasets at 5-10 nm in x-y. It has never seen light microscopy.

- Affinity convention (checked in the authors' `seg_to_affgraph` and `mknhood3d(1)`, which
  evaluates to offsets (-1,0,0), (0,-1,0), (0,0,-1)): channel c at voxel v is the pair
  (v, v - o_c), 1 when both voxels carry the same nonzero label. This is our convention
  (`vitseg.OFFSETS[:3]`, pairs (v, v - o)). Unlike our ViT, SegNeuron is trained with
  affinity 0 on every pair that touches label 0, so its affinities also mark neuron-background
  boundaries.
- Foreground target: `label != 0` (file name `boundaries.tif`, but it is not a membrane map).

## Pinned inputs

- Code: github.com/yanchaoz/SegNeuron, commit `ea6f0c963a2f60f0f130244da27e2b2d753f8ee9`.
  Run unmodified through their command lines: `Train_and_Inference/inference.py` (20x128x128
  tiles, stride 10x64x64, Gaussian blending, input / 255) and `Postprocess/FRMC_post.py`.
- Weights: huggingface.co/yanchaoz/SegNeuron, revision `b6b92dbcdc654a1aba119b3cb40c069f15662491`,
  `SegNeuronModel.ckpt`, 161,669,157 bytes, sha256
  `3767a3793cbbdbf98041b8c4c270b92d2355213d11605568c003322bb759bef5` (checked before use;
  the inference manifest records the hash again).
- Environments on Wahab, outside the repository, under `/scratch/$USER/segneuron/`:
  - inference: a uv virtual environment with PyTorch (CUDA 12 wheel), numpy and tifffile, per
    the repo's `requirements.txt`;
  - FRMC: micromamba (static binary) environment from `environment-postprocess.yml`
    (python-elf 0.8.1, nifty, vigra, numpy < 2).
  Package versions are written next to each run. The existing `~/envs/liconn_v2` environment
  runs our pipeline and the scoring; it is not changed.

## Input

Crop 12 raw image (`manyseed.load_raw_xyz(12)`, uint8), transposed to (z, y, x) as the README
requires. No intensity change: the model divides by 255 itself. LICONN cell interiors are bright
with dark extracellular space between them; EM shows dark membranes in lighter cytoplasm. Polarity
is not inverted (that would be a design choice the paper does not cover); this is reported, not
tuned.

- **native**: 500 x 500 x 500 at 24 x 18 x 18 nm (z, y, x).
- **xy2**: y and x upsampled 2x by linear interpolation with pixels treated as areas
  (`scipy.ndimage.zoom(order=1, grid_mode=True, mode='nearest')` in float32, then rounded to
  uint8 because the inference script requires uint8) to 500 x 1000 x 1000 at 24 x 9 x 9 nm,
  inside the training range. Fine voxels 2i and 2i + 1 lie within coarse voxel i.

## Back to the crop grid (xy2 only)

Outputs of xy2 are mapped back to the 500^3 grid before any post-processing or scoring, so both
resolutions are post-processed and scored on the same voxels (`segneuron.to_crop_grid`):

- foreground: mean over each 1x2x2 block of fine voxels.
- z affinity of coarse voxel v (pair v, v - z): mean of the 4 fine z affinities in v's block.
- y affinity of v (pair v, v - y): along each of the block's 2 fine columns, the minimum of the
  three fine y edges between the centres of v - y and v: inside v - y's block (fine 2i - 1 to
  2i - 2), crossing (2i to 2i - 1) and inside v's block (2i + 1 to 2i); then the mean over the 2
  columns. x likewise. The minimum keeps a boundary that the model places inside a block (a
  one-voxel dark gap at 18 nm becomes two fine voxels, whose centre is the edge inside the
  block); using only the crossing edge would erase it (method check, 2026-10-26). A boundary
  inside block i therefore lowers both of i's x-y pairs. At i = 0 the pair leaves the crop;
  nothing reads it as an edge.

## Post-processing A: ours (as for the ViT)

The outputs are put in our 7-channel layout (`segneuron.vitseg_layout`): channels 0-2 =
SegNeuron's z, y, x affinities; channels 3-5 (our long-range offsets, which SegNeuron does not
predict and segmentation does not read) = 0.5; channel 6 = foreground. Then `vitseg.fragments`,
`vitseg.fragment_edges`, `vitseg.agglomerate` unchanged (neuron = foreground >= 0.5), and the
threshold chosen on crop 12 by mean F1 over the 457 tuning objects (`vitseg.choose_threshold`,
thresholds 0.05-0.95).

## Post-processing B: the authors' FRMC

`FRMC_post.py` as released, on the crop grid: affinities restricted by the foreground
(elementwise minimum), per-slice 2D distance-transform watershed, region adjacency graph and
Kernighan-Lin multicut with bias beta. beta is the analogue of our threshold and is chosen on
crop 12 the same way, from {0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9} (0.25 is the authors' default),
by mean F1 of the masked scoring below. FRMC labels every voxel, background included. It is
scored twice at every beta:

- **as released**: the segmentation unchanged (predicted background fraction 0).
- **masked**: voxels with foreground < 0.5 set to 0 afterwards. This is the same background
  rule as post-processing A for scoring, but not the same connectivity: segments that FRMC joined
  through background keep one ID after masking, whereas our pipeline never connects through
  background.

FRMC's per-slice watershed has no foreground mask, so its fragments take in surrounding
background; its foreground restriction `min(aff, fg)` uses the foreground of v only. Its fixed
watershed settings (`threshold=0.25`, `sigma_seeds=2.0` voxels) were set for 6-9 nm pixels and
act on the 18 nm crop grid here at both resolutions. If a slice gives the watershed no seed,
`FRMC_post.py` stops with an error; that would be reported as a failure, not worked around.

## Scoring (crop 12 only; `segneuron.score`)

For each of the 457 tuning objects (`vitseg.tuning_objects`), `vitseg.seed_records` of the
segment at its seed point; mean recall, precision and F1 (the `benchmark.score` definitions:
precision ignores label-0 voxels in a segment); recall lost to background (seg = 0) versus to
other segments; pieces holding at least 1% of the object; `vitseg.background_scores`. Network
outputs against the labels: foreground channel precision and recall against label > 0; for each
nearest-neighbour channel, the fraction of same-neuron pairs below 0.5 and of different-neuron
pairs at or above 0.5 (pairs with both labels nonzero), and the fraction of neuron-background
pairs (exactly one voxel labelled; background-background pairs not counted) at or above 0.5.
Pairs leaving the crop are not counted.

Threshold-free diagnostics (method check, 2026-10-26): the 0.5 cuts suit a model calibrated on
this data; a zero-shot EM model may separate the classes well with its outputs shifted. So the
foreground channel is also reported at 0.3 and 0.7 (not used for segmentation), with its AUC
(label > 0 against label 0) and class quantiles; each affinity channel with its AUC of
same-neuron against different-neuron pairs and against neuron-background pairs, and quantiles.
AUCs are computed from 1000-bin histograms (ties within a bin count half).

Reference points: ViT step 66,000 F1 0.531 and 6-connected oracle 0.924 (both crop 12, same
objects and scoring); FFN fp16_head32 0.894 (different objects, `docs/vitseg_evidence.json`).
No pass/fail verdict: crop 12 chooses the threshold and beta, and the test crops 13-40 are not
read.

## Jobs

`cluster/wahab/segneuron.sbatch`, driver `examples/run_segneuron.py`:

1. GPU (V100): `prep` writes the native and xy2 input volumes; the authors'
   `inference.py` runs on each; `grid` writes the xy2 outputs on the crop grid
   (`affinities.npy`, `boundaries.tif`).
2. CPU (`main`, after 1): the authors' `FRMC_post.py` for each resolution and beta; then
   `score` runs post-processing A and scores everything into `segneuron_crop12.json`.

The time limits in the sbatch header's example commands (2 h on `timed-gpu`, 12 h for the CPU
job) are too short for the recorded run (22 min GPU, 14 h 13 min CPU; `docs/segneuron_results.md`).
The header is left as it ran, so that its hash matches the run's source manifest. Submit as the
recorded run did:

    j=$(sbatch --parsable --export=ALL,MODE=infer -p gpu --gres=gpu:1 -t 8:00:00 -o "$RUN/%j.out" cluster/wahab/segneuron.sbatch)
    sbatch --export=ALL,MODE=post -p main -t 24:00:00 -d afterok:$j -o "$RUN/%j.out" cluster/wahab/segneuron.sbatch

## Decision after this milestone (owner)

Fine-tune on crops 0-11 if zero-shot shows that the pretrained features carry over; otherwise
report and stop. The segmentation F1 against the ViT's 0.531 is one indication; the
threshold-free AUCs are the more direct one, because a zero-shot model with uncalibrated outputs
can fail at fixed thresholds while its features still separate the classes.

## Disclosed

- The ViT's numbers come from a model trained on crops 0-11 of this volume; SegNeuron here has
  seen no LICONN data. The comparison is about the starting point for fine-tuning, not a contest.
- Up-sampling invents no detail; at native resolution neurons look half the size the model was
  trained on. The xy2 input is rounded to uint8 after interpolation (at most 0.5 grey level).
- Crop 12 chooses our threshold and FRMC's beta, so the numbers are optimistic for both.
- Precision ignores label-0 voxels, so FRMC as released (no background at all) pays nothing for
  absorbing background; seeds that land in predicted background score 0 under the masked rule
  and ours. Both are reported (seeds in background, background scores).
- Our boundary map takes the lowest affinity among a voxel's edges to other neuron voxels.
  SegNeuron was trained with affinity 0 on neuron-background pairs, so where its foreground
  extends past a neuron's edge, rim voxels become unconfident; thin neurites may fragment for
  this reason rather than because of the features.
