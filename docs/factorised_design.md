# Factorised FFN pilot: design

Owner decisions on 2026-10-31:
- build a factorised FFN pilot directly (before batched multi-seed tracing);
- train it by distillation from the authors' FFN network (the teacher), not from labels;
- port the network to PyTorch and train on MIT B200 GPUs;
- pilot signal fixed before training: on the 457 crop-12 objects, mean F1 >= 0.86 and crop
  time at least 2x lower than FFN on the same GPU. This is a pilot signal, not a test result.

## Why

The authors' FFN network joins the image and the object mask at its first convolution
(`liconn_v2/fasttrace.py`: `concatenate([image, seed])`), so all 42 convolutions depend on the
object and on the step. Nothing can be reused between steps (consecutive 33^3 boxes overlap by
25 of 33 voxels per axis) or between objects. On crop 12, FFN took 137,723 steps for 457 objects
(`docs/headtohead_results.md`), about 11.2 PFLOP at 81.6 GFLOP per step; one pass of the same
network over the 500^3 crop would be about 0.28 PFLOP.

The factorised FFN splits the network:
- an **image encoder** that sees only the image and runs once per volume (dense, tiled);
- a **mask decoder** that takes the encoder features of the current box and the current object
  mask, and runs at every step.

Tracing (seed, 33^3 box, face-maximum moves of 8 voxels, move and segment thresholds 0.6) stays
as in the authors' notebook; only the network changes. This is the design of image-encoder /
prompt-decoder segmenters (SAM) and of video object segmentation models, applied to FFN. We
found no published factorised or transformer FFN (literature search of 2026-10-31).

## Architecture (from the teacher, `connectomics.jax.models.convstack.ResConvStack`)

Teacher: 32 features; `pre_a` (3^3 conv, 2 -> 32), LayerNorm, ReLU, `pre_b` (3^3, 32 -> 32);
20 residual modules, each x + conv(ReLU(conv(ReLU(LayerNorm(x))))); LayerNorm, ReLU, 1x1x1
`output`. Input channels are [image, mask logit]. See docs/factorised_sources.md.

Factorised student, with encoder depth K (pilot: K = 16):
- **Encoder(image)** = `pre_a` restricted to its image channel, LayerNorm, ReLU, `pre_b`, residual
  modules 0..K-1. Initialised from the teacher. With the mask channel equal to 0 (a logit of 0 is
  FFN's "unknown", the pad value 0.5), the teacher's residual stream after module K-1 is exactly
  the encoder's output on the same box, so the initialisation is a faithful slice of the teacher.
- **Decoder(features, mask)** = features + MaskStem(mask), then residual modules K..19, LayerNorm,
  ReLU, `output`, initialised from the teacher. MaskStem is conv 3^3 (1 -> 32), ReLU, conv 3^3
  (32 -> 32) with the second convolution zero-initialised, so at initialisation the decoder
  ignores the mask and the student equals the teacher with an all-unknown mask.
- **Decoder dilations.** The mask reaches each output only through the decoder. With four
  undilated modules the mask's reach is about 10 voxels, less than the 16 voxels from the box
  centre to its faces. The pilot uses dilations (1, 2, 4, 1) in the four decoder modules
  (reach about 18 voxels). Dilation changes what the teacher's weights compute, so training must
  repair it; undilated decoders are a control if training allows.
- **Dense encoding.** The encoder's receptive radius is r = 2 + 2K voxels (34 for K = 16). Dense
  features are defined as the encoder applied to the image zero-padded by r, with the padding
  cropped off, so each feature depends only on the image within r voxels (zeros outside the
  volume). Tiles with a halo of r then give exactly the same features as one pass. Unlike the teacher, which zero-pads at the box
  border, the student's box features include real image context beyond the box. Exact agreement
  with the teacher's traces is therefore impossible by design; the student is judged against the
  proofread labels.

## Cost (multiply-adds x 2; LayerNorm and ReLU ignored)

Per voxel: a 3^3 32 -> 32 convolution is 55,296 FLOP.

| | per voxel | per 33^3 box (35,937 voxels) |
|---|---|---|
| teacher (2 pre + 40 residual convs + output) | 2,270,656 | 81.6 GFLOP |
| encoder, K = 16 (dense, once per voxel of the volume) | 1,826,496 | — |
| decoder, K = 16 (stem + 8 residual convs + output) | 499,456 | 17.9 GFLOP |

Crop 12 at the recorded 137,723 steps: teacher 11.2 PFLOP; decoder 2.47 PFLOP. The encoder runs
on each tile plus its halo of r = 34 voxels (`factorised.encoder_input_voxels`): 0.33 PFLOP in one
pass over the padded 568^3 crop, 0.47 with 250^3 tiles, 0.84 with 125^3 tiles. The student
total is therefore 2.8-3.3 PFLOP, 3.4-4.0x less than the teacher. (A first count of 0.23 PFLOP for
the encoder ignored the halo; the independent method check pointed this out.) FLOPs are not
time: the dilated decoder, tiling and per-step host overheads must be measured (part 3).

## Risks for part 2 (from the independent method check; to be handled in its design)

- **Mask reach at the first step.** At step 0 only the seed voxel is known; the mask stem and four
  decoder modules must carry it to the box faces at +-8 voxels above the move threshold, or the
  trace stops at once. Reach with dilations (1, 2, 4, 1) is about 18 voxels, so this is
  possible but must be checked early in training (first-step move rate on training seeds).
- **Dilation across thin structures.** Dilation 4 samples voxels 72 nm apart laterally;
  membranes and thin axons are 1-3 voxels wide, so dilated taps may skip boundaries and favour
  merges. The undilated decoder is the control.
- **Box-edge targets.** The teacher zero-pads at the box border, so its logits near the faces
  are made without outside context; the student has that context. Distillation loss should be
  weighted toward the box interior, or the difference reported.
- **Mask stem start.** With its last convolution at zero, the first convolution receives no
  gradient on the first update (only the last one does); the ReLU directly after a single-channel
  convolution of logits can leave channels inactive. A unit test checks that a gradient step makes
  the output depend on the mask; training curves must show it too.
- **Wall-clock speed.** At batch 1 on a B200 a 33^3 decoder evaluation may be dominated by kernel
  launches and the per-step device-to-host transfer, not FLOPs; CUDA graphs or torch.compile and
  the fast tracer's bookkeeping are needed before timing.
- **Encoder trained or frozen.** Not fixed yet: a frozen, teacher-initialised encoder allows
  caching features of crops 0-11; joint training is more flexible. Part 2 decides and states it.

## Parts

1. **Port and model (this milestone; local CPU, no training).**
   `liconn_v2/factorised.py`: a PyTorch port of the teacher loaded from the authors' checkpoint;
   the student (encoder, decoder, teacher initialisation); tiled dense encoding; FLOP counts.
   Pre-registered checks:
   - the port matches the JAX teacher in float32 to a maximum absolute logit difference of 1e-3
     on random inputs and on boxes of the notebook crop with realistic masks;
   - the student with undilated decoder at initialisation equals the teacher with mask 0 on the
     same box (same tolerance);
   - tiled dense encoding equals one untiled pass over the same region (same tolerance), and its
     box crop equals the encoder applied to the box with an r-voxel halo of real image (zeros
     outside the volume), cropped back to the box;
   - added after the method check: the port also matches on network inputs recorded along the
     notebook's reference trace (every 24th step), with that trace checked against the recorded
     reference fingerprint;
   - the FLOP count of the teacher reproduces the 81.6 GFLOP per step used in the manuscript.
2. **Distillation on MIT B200 (next milestone, separate approval of the run).** Teacher
   trajectories on crops 0-11 (seeds from the proofread labels, as for FFN's earlier seed sets);
   loss on teacher logits at visited boxes; later rounds let the student trace and the teacher
   label the boxes the student visits (on-policy). Crop 12 is used only for scoring; crops
   13-40 are not read.
3. **Scoring (after part 2).** Student traces of the 457 crop-12 objects with the authors'
   movement policy, scored with `benchmark.score` exactly as FFN and SegNeuron in
   `docs/headtohead_design.md`; crop time on the same GPU as an FFN run; pilot signal as above.

## Limits stated in advance

- The teacher may have been trained on proofread data that include crop 12; the student
  inherits any such leak. Crop-12 numbers are tuning numbers, not test numbers.
- Part 1 shows only that the port and the student are built correctly; it says nothing about
  whether the student can learn to trace.
- CPU numerics differ from GPU numerics; the B200 run uses TF32 or lower precision and is
  checked against the teacher there separately.
