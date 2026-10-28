# Training a cheaper FFN network: feasibility and data sources (2026-10-21)

Owner request: explore whether training (option b) is feasible, and in particular
where training data could come from. This is a desk study: nothing was trained. It
is not a plan the owner has approved.

What was read:
- the bucket listing and each volume's metadata (info files);
- the ISTA repository's web listing and readme;
- one coarse level of the ExPID82_1 proofread segmentation (288 x 288 x 384 nm;
  240 x 305 x 50 voxels; about 29 MB of public label data), to measure label coverage.

That last read went beyond the "metadata only" wording of the first task contract
for this milestone. The owner approved it on 2026-10-21. No image data was downloaded.

## Why training is on the table

Training is one route to further gains, not the only one. The no-training changes
tested so far, judged against the owner-confirmed 1% tolerance:
- skip/postpone policies failed (docs/postpone_results.md);
- the two benchmarked residual-block drops failed; the other 18 drops were not
  benchmarked (docs/block_drop_results.md);
- float16 evaluation passed on the 12 benchmark seeds, and held-out confirmation is
  pending (docs/precision_results.md).

The block drops changed about 1-2% of the network's decisions per evaluation and lost
3-11 recall points over full traces. Structural changes that remove work from the
network (fewer blocks, a smaller network) would probably need training to come
close enough to the reference. That is a hypothesis these experiments suggest, not
something they establish.

## What data exists (checked 2026-10-21)

examples/survey_training_data.py repeats the bucket listing, volume metadata and
coverage measurement below; the coverage step reads the coarse label level described
above. Its output
(`data/runs/training-survey-20261021/training_data_survey.json`) is hashed and
checked by tests/test_precision_evidence.py. The ISTA repository contents were read
from its web listing and readme, and are not checked by a test.

Public bucket gs://liconn-public (listed with the storage API):

| dataset | contents | voxel size (finest) | extent (voxels, x/y/z) | usable as |
|---|---|---|---|---|
| ExPID82_1 (paper Fig. 2; our benchmark volume) | image; **proofread segmentation** 231030_agg_240123; automatic agglomeration | 9 × 9 × 12 nm (FFN uses 18 × 18 × 24) | 7,650 × 9,740 × 1,590 (3,825 × 4,870 × 795 at 18 nm) | labels and image |
| ExPID124 (paper Fig. 1a) | image; automatic FFN segmentation + agglomeration (not proofread) | 10 × 10 × 25 nm | 47,000 × 13,000 × 1,000 | image; weak labels |
| ExPID146 (Ext. Data Fig. 10) | image only | 9 × 9 × 25 nm | 11,098 × 11,094 × 2,300 | image |
| ExPID150 (multi-round fusion) | image only | 10 × 10 × 25 nm | 11,000 × 11,000 × 12,000 | image |
| ExPID09_01 | image only | 10 × 10 × 25 nm | 2,304 × 2,304 × 969 | image |
| models/ffn/axde_59110972 | the authors' FFN checkpoints (ckpt-257, -1110, -2116) | | | **teacher network** |

Proofread-label coverage in ExPID82_1, measured at the 288 × 288 × 384 nm level:
- 32.7% of the whole volume is labeled; about 47% within the imaged z-range (z bins
  2-8 of 10). The rest is empty or unimaged.
- Coverage is fairly uniform in x and y.
- The volume holds 50,691 labeled objects at that resolution.
- At 18 × 18 × 24 nm that is about 4.8 billion labeled voxels. Our benchmark crop is
  0.125 billion voxels (about 0.8% of the volume).

ISTA data repository (doi 10.15479/AT:ISTA:18697 and
https://pub.ista.ac.at/group_danzl/data/LICONN/), read from the listing and readme:
- figure images and test images;
- a copy of the Fig. 2a segmentation;
- synapse-prediction volumes;
- `train07_20K`: a PyTorch checkpoint (279 MB), most likely the synapse-detection
  network from danzllab/LICONN; its layer names do not match an FFN.

**Not found:** the manual annotations the paper used to train its FFN models (162
fragments, 103 agglomerated fragments, and the proofread axon/dendrite sets of Model
3), and the webKnossos skeletons used for evaluation. The paper does not list them as
released. Requesting them from the authors is possible.

## Three possible sources of training data

1. **Distillation from the authors' network (no labels needed).** The authors' network
   is the teacher; a smaller or pruned network learns to reproduce its outputs on the
   same inputs (image box + seed box).
   - Inputs can come from teacher traces on any LICONN image. All four other datasets
     plus ExPID82 outside the benchmark crop give effectively unlimited data.
   - It optimises the right target. The owner's tolerance is defined against the
     reference run of this network, so matching the teacher is matching the reference.
   - The block-drop lesson applies: small per-step differences accumulate along a trace.
     Training only on the teacher's own trajectories may not be enough. A student that
     drifts sees inputs the teacher never produced. A standard remedy is on-policy
     distillation: the student traces, and the teacher labels the inputs the student
     actually visits (DAgger-style).
   - Code: we already record network inputs along traces (liconn_v2/pruned.InputRecorder).
     The loss would be a per-voxel regression or cross-entropy on teacher logits.
2. **Supervised FFN training on the proofread segmentation of ExPID82.** This is
   the authors' own recipe. The pinned google/ffn commit contains a JAX training
   pipeline (`ffn/jax/train.py`, `main.py`, `input_pipeline.py`) for this network
   type.
   - Caveats: the labels are agglomerated and proofread FFN output, not independent
     dense ground truth (proofreading targeted specific errors), so a model learns the
     pipeline's own errors too.
   - The benchmark crop and a margin must be excluded from training. The authors' own
     final model (Model 3) was trained on proofread data from this volume, so the
     reference network may already have seen the crop.
3. **Weak labels from ExPID124's automatic segmentation.** Unproofread FFN output,
   so it is close to distillation with extra noise. Not recommended over (1).

## Compute (order of magnitude; must be measured in a pilot)

- The authors trained from scratch on 32 V100s for up to 30 days (about 8.8 million
  steps, batch 128).
- Distilling into a network initialised from the teacher's own weights (for example
  the teacher with some blocks removed) should need far less, but how much is
  unknown. A pilot on a few Wahab V100s (the `gpu` partition allows 14-day jobs; 16
  nodes × 4 V100, shared) would measure training speed and how fast the
  per-evaluation flip rate falls.
- Any trained variant is judged as before: the benchmark against the reference under
  the 1% tolerance, then new held-out seeds (for example, axons not in the benchmark,
  or a second crop) before it is called a speedup.

## Assessment

- Feasible in principle. Distillation is the best fit: it needs no labels, can use all
  public LICONN images, and targets exactly what the tolerance measures. The authors'
  pipeline also gives a supervised route with their proofread labels.
- Main risks: (1) whether any cheaper network can reach the closeness needed for the
  1% rule over hundreds of steps (block drops at about 1.2% flips per evaluation were
  far off); (2) compute cost; (3) leakage between training data and the benchmark.
- Suggested first step if the owner allows training: a pilot that distils a teacher
  with 2-4 blocks removed on teacher-trajectory inputs from ExPID82 outside the
  benchmark crop. Track the screen flip rate on the benchmark inputs during training,
  and benchmark only if it falls well below 1%.
