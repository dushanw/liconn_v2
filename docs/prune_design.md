# Pruned FFN, distilled: design

Owner decision on 2026-11-02, after the timing study (docs/distil_results.md): "what about
pruning FFN, while finetuning the pruned model?", then "yes, go with 8 and 4, pruning first".
A pilot on crop 11 (validation). Crop 12 is not read; crops 13-40 are never read.

## Question

The factorised student (docs/factorised_design.md, docs/distil_design.md) runs 4 residual blocks
per step after a 16-block image encoder that runs once per volume. Could a plain FFN with as few
blocks per step do as well, without the encoder? A pruned FFN keeps FFN's own input (image and
mask joined at the first layer) and its tracing, but has fewer blocks, and is trained by the same
distillation as the student. P4 runs the same residual blocks per step as the student, so it asks
mainly what the encoder's features buy in accuracy; P8 asks what twice the blocks per step buy.
Measured on crop 11:
- mean F1, recall, precision, merges and steps against the authors' FFN (teacher) and the student;
- crop time on one B200 against FFN F5 and student S10 (the fastest settings of the timing study).

No pass mark is set; the comparisons are reported as observed. They choose which network the
runaway-trace work (the next owner-approved step) starts from.

## Earlier evidence

- Dropping blocks without retraining failed (docs/block_drop_results.md): dropping block 12 cost
  one seed 24 recall points, blocks 12 and 17 one seed 40. No single block's removal left
  decisions nearly unchanged (lowest flip rate 1.18%, block 12; highest 7.0%, block 1).
- Distillation with one on-policy round brought the student to crop-11 mean F1 0.876 against the
  teacher's 0.898; validation loss on teacher states did not track trace quality.
- Per step (job D, fp16_head32, stream x 1/16, 32 at once): FFN 1.44 ms (81.6 GFLOP), student
  0.58 ms (17.9 GFLOP). A straight line through these two points gives about 0.3 ms per step plus
  0.013 ms per GFLOP; this is a rough guide from two different networks, not a prediction.

## Networks

Both are `factorised.ConvStack` with the authors' stem (image and mask channels) and head, and a
subset of the authors' residual blocks, initialised from the authors' weights:

| name | blocks kept (0-based) | dilations | receptive radius | GFLOP per 33^3 box |
|---|---|---|---|---|
| P8 | 12-19 | 1 x 8 | 2 + 2 x 8 = 18 | 33.9 |
| P4 | 16-19 | 1, 2, 4, 1 | 2 + 2 x 8 = 18 | 18.0 |

- **Which blocks.** The last K. P4 keeps the student's decoder blocks with the student's dilations,
  so the two run the same residual blocks per step (18.0 against 17.9 GFLOP). They differ in the
  image they see, not only in where it enters (independent method check of this task):
  - the student's box features come from 16 frozen blocks run over the whole crop, so each feature
    carries image context from up to 50 voxels away, beyond the box; P4 sees only the 33^3 image box
    through a 2-convolution stem, with zeros outside the box (as FFN does);
  - P4's stem mixes image and mask before a LayerNorm; the student adds a mask stem (no LayerNorm)
    to the features;
  - the student's encoder is frozen and its features are stored in float16; all of P4 trains, on
    float32 image boxes.
  P4 skips the encoder and reads a 1-channel image box instead of a 32-channel feature box. We
  expect (a hypothesis, to be measured; no pruned network has been timed) its step to cost about
  the same as the student's and its crop time to be lower, unless it takes more steps. If so, the
  question P4 answers is mainly one of accuracy: **at the same per-step
  network cost, what do the student's deep, wide-context image features buy?** P8 extends P4 with
  the four blocks before it. Both reach 18 voxels from the box centre, beyond the 16 to the faces
  (the student's mask reach is the same). The block screen ranked single-block removals from the
  full network; it does not say which subset is best to keep, and training changes all weights.
- **What changes at initialisation.** The kept blocks receive the stem's output instead of the
  output of the dropped blocks (block 12 or 16 was trained on the sum of the stem and 12 or 16
  residual updates), and P4's dilations change what its weights compute (the student's dilated
  blocks at least act on the teacher's own deep features). The pruned networks therefore start far
  from the teacher; their validation loss at step 0 is logged, their crop-11 F1 at initialisation
  is not measured. A poor start may need more training than the student did; this is a risk, not a
  tested claim.
- **Cost.** A 3^3 32 -> 32 convolution is 55,296 FLOP per voxel; the stem (2 -> 32 and 32 -> 32)
  and head as in the teacher. P8: 8 blocks x 2 convs + stem + head. Dilation does not change the
  count. GFLOP above: (stem + 2K convs + head) x 35,937 voxels; the teacher is 81.6 and the
  student's decoder 17.9 by the same count.

## Training (as the student, docs/distil_design.md)

- **Samples.** The same teacher samples (crops 0-10 for training, 4,096 fixed crop-11 samples for
  validation, all step-0 samples included). A sample stores the mask box, the teacher's logits and
  the box centre; the pruned network's input is the image box at that centre (normalised crop,
  float32, on the GPU) and the mask box.
- **Loss and optimiser.** Unchanged: binary cross-entropy against the teacher's probabilities over
  the whole box; AdamW, peak 3e-4, 500 warm-up steps, cosine decay to 0, no weight decay, batch 32,
  TF32 allowed in training; validation every 1,000 steps; the checkpoint with the lowest
  validation loss is kept as best.pt, the final one as last.pt. All weights train (stem, blocks,
  head).
- **Steps.** The same as the student for both networks: 400,000 in round 0 and 150,000 in round 1,
  with the learning-rate schedule over the full count. If the 200-step probe shows that a round
  does not fit in one 6-h job, it runs as consecutive jobs: each job stops before the time limit
  and saves `resume.pt` (weights, optimiser, step, the batch generator's state, the log), and the
  next job continues from it (`--resume`), so the schedule and the batches are those of one
  uninterrupted run. (The student's flat validation loss after 100k steps does not justify fewer
  steps: that loss did not track trace quality.)
- **Round 1 (on-policy), as for the student.** The round-0 best.pt traces crops 0-10 from the
  teacher's seeds; the teacher labels the pruned network's own boxes and masks on the kept steps
  (same sampling rule); 150,000 further steps from round-0 best.pt on teacher plus on-policy
  samples (learning rate restarted, as for the student). In round 1, step 0 (the round-0
  checkpoint itself) is validated and logged but cannot become round 1's best.pt; otherwise, since
  validation uses teacher states only, best.pt could stay the round-0 network. (The student's
  round 1 did not hit this: its best was at step 139,000.) Round-0 best.pt remains a candidate in
  the choice below anyway. The share of on-policy samples in round 1 depends on how long each
  network's round-0 traces are, as it did for the student; it is reported, not equalised.
- **Choice.** best.pt and last.pt of round 1, and best.pt of round 0, trace crop 11; the highest
  mean F1 is selected per network (the student's rule). Crop-11 numbers are tuning numbers.

## Measurement

- Crop-11 traces (float32, one at a time, TF32 off; `trace_student.py`, which handles both kinds of
  checkpoint): per object score against the labels and against the teacher's mask, steps, merges.
- One timing job (`time_crop11.py`, one B200): F1, F5, S10 and, per pruned network, float32 one at
  a time and fp16_head32 with the stream x 1/16 and 32 at once. Every timed run is also scored
  (mean F1, recall, precision, merges, steps, per-object records) and reports both ms per step and
  crop seconds, so per-step cost is separated from trace length. Non-finite outputs are counted;
  the reduced-format runs are compared with the float32 run of the same network per object, as
  for FFN and the student, since non-finite counts do not show silent accuracy changes.
  The scaled stream (the stem's last convolution and every block's last convolution x 1/16) leaves
  the network unchanged except through LayerNorm's epsilon, which becomes 256 times larger relative
  to the stream's variance; it is approximate, as for FFN and the student, and its effect on a
  trained pruned network is measured by the per-object comparison with the float32 run, not
  assumed.

## Code (all new behaviour optional; the student and teacher paths are unchanged)

- `factorised.ConvStack(depth, dilations=None)`: optional per-block dilations (default 1).
- `factorised.pruned_from_teacher(teacher, keep, dilations)`: the stem, blocks `keep` and head of
  the teacher, copied.
- `distil.load_checkpoint(teacher, ckpt)`: a student (decoder checkpoint) or a pruned network
  (checkpoint with `keep`), so that on-policy collection, crop-11 traces and timing take either.
- `examples/train_pruned.py`: training as `train_distil.py`, with image boxes instead of features,
  `--resume` for rounds split over jobs, and round 1's step 0 excluded from best.pt.
- `cluster/mit/distil.sbatch`: modes `prune-probe` and `prune-train` (KEEP, DIL), `onpolicy` with an
  output name, `trace` for either checkpoint kind.

## Compute

Per network on one B200: probe (minutes), round 0 (<= 5 h), round-0 traces of crop 11 and
on-policy collection of crops 0-10 (about 2 h), round 1 (<= 5 h, 320 GB memory as for the
student), crop-11 traces (minutes). P8 and P4 run in parallel (two GPUs per user). Then one timing
job (about 1.5 h). About 25 B200-hours in all, in `mit_normal_gpu`.

## Limits stated in advance

- Crop 11 selects checkpoints and is the only crop scored; its numbers are tuning numbers.
- One training run per network, no seed repeats; no error bars on F1 or times.
- P8 and P4 may get fewer round-0 steps than the student if the probe says so.
- Distillation targets the teacher, so the pruned networks imitate FFN's errors as well; no
  proofread labels are used in training.
- The block choice (last K) is one choice among many; other subsets are not tried. The early
  blocks, whose single removal changed the most decisions, are all dropped.
- P4 against the student differs in image context, depth of image processing, frozen versus
  trained weights, feature precision and how the mask is combined (Networks, above); a difference
  in accuracy cannot be assigned to one of these.
- The FLOP estimate and the per-step guide above are not times; the timing job measures them.
