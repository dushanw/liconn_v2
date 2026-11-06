# Factorised FFN pilot, part 2: distillation on MIT B200 (design)

Part 1 (docs/factorised_port_results.md) built the PyTorch port of the authors' network (the
teacher) and the factorised student. This part trains the student's mask decoder to reproduce the
teacher's outputs along the teacher's own traces, and compares student and teacher traces on a
held-out training crop. Crop 12 is not read in this part; crops 13-40 are never read.

Owner decisions on 2026-10-31: distil from FFN; PyTorch on MIT B200 (about 1-2 B200-days approved);
"start the next steps" after the recommendation below (frozen encoder, dilated and undilated
decoders compared, first-step reach checked early).

## Teacher traces (crops 0-11)

- **Objects and seeds.** In each of crops 0-11, the large objects by the rule used for crop 12
  (`vitseg.large_objects`), each seeded at `benchmark.seed_point`. Crops 0-10 train; crop 11 is
  the validation crop (not trained on).
- **Teacher.** The PyTorch port in float32 with TF32 off. Its weights are exported once from the
  authors' checkpoint to a torch state file whose sha256 is recorded, and the export is checked to
  equal `factorised.teacher_from_params` exactly.
- **Tracer.** `distil.TorchTracer` repeats `fasttrace.FastTracer` with a PyTorch network: the
  same input assembly (NaN -> pad logit 0), the same canvas update (FFN's seed-consistency rule),
  the same six face maxima with first-NaN/first-maximum ties, and FFN's own
  `FaceMaxMovementPolicy` and `MovementRestrictor` on the host via `fasttrace.queue_moves`.
  The network is either the teacher (image box + mask box) or the student (encoder features of
  the box + mask box). Checks (CPU): the canvas update, maxima and positions equal FastTracer's
  JAX functions exactly on random canvases; the torch teacher reproduces the notebook's reference
  trace (same step count and history fingerprint as `expected_reference.json`).
- **Samples kept.** Steps 0-3 of every trace (the first-step reach risk) and each later step
  with probability 1/4 (a generator seeded by crop and trace). A sample is the input mask box
  (float16 logits), the teacher's output logits (float16), the box centre, the step index and
  the trace. About 1.8 M teacher steps and 0.45 M samples (about 65 GB) are expected.
- **Per trace:** the teacher's object mask (flat voxel indices of canvas >= logit 0.6) and
  `benchmark.score` against the proofread labels.

## Student training

- **Encoder frozen.** The teacher's stem (image channel) and modules 0-15, as initialised in part
  1. Features of each crop are computed on the GPU by `factorised.encode_volume` (float32, TF32
  off) and kept on the GPU in float16 (8 GB per crop). The float16 rounding error is reported.
- **Two decoders**, trained the same way in separate jobs: dilations (1, 2, 4, 1) (the pilot)
  and (1, 1, 1, 1) (the control; equals the teacher with an unknown mask at initialisation).
- **Loss.** Binary cross-entropy between the student's logits and the teacher's probabilities,
  sigmoid(teacher logit), averaged equally over all 33^3 voxels: every voxel of the box is written
  to FFN's canvas and feeds later steps. The teacher zero-pads at the box border and the student
  sees real context there; the outer 4-voxel shell is 56.5% of the box (the method check pointed
  this out), so loss and agreement are reported separately for the shell and the interior. The
  move faces (+-8) are in the interior. The gradient with respect to the student logit is
  sigmoid(s) - sigmoid(t). Where student and teacher are both confident on the same side, it is
  small (logits 5 and 8 differ by under 0.01 in probability), so logit scale far from the threshold
  is trained only weakly. A confident wrong logit is still penalised fully. The crop-11 traces show
  whether the weak training of logit scale matters. (Wording corrected after the runs on Codex
  review; the method is unchanged.)
- **Optimisation.** AdamW, learning rate 3e-4 with 500 warm-up steps and cosine decay to 0,
  weight decay 0, batch 32, TF32 allowed during training. Step count set from a 200-step timing
  probe so that a run fits in 5 h (the 6 h partition limit), the same for both decoders.
- **Validation** every 1,000 steps on 4,096 fixed crop-11 samples (all crop-11 step-0 samples,
  the rest drawn at random): loss (whole box, interior, shell); voxel agreement with the teacher
  at the 0.6 threshold (interior and shell); first-step moves (for step-0 samples: faces whose
  maximum reaches the move threshold for the teacher, for both, for both with face maxima within
  2 voxels of each other, and for the student only). The checkpoint with the lowest whole-box
  validation loss is kept.
- Steps 0-3 are about 5% of the samples (step 0 about 1%). Batches are drawn uniformly, so the
  first steps are not over-weighted; the first-step metrics show whether that is a problem.

## Student traces on crop 11 (selection)

Each trained student traces the crop-11 seeds with `TorchTracer` (decoder in float32 with TF32
off, on the same float16 encoder features as in training), the authors' movement policy and
thresholds. Per object: `benchmark.score` against the labels, F1 against the teacher's mask,
steps, and trace time (a diagnostic; formal timing is part 3). Mean F1 is the mean over objects
of 2PR/(P+R) (0 if P + R = 0), as `vitseg.mean_f1`. `benchmark.score` leaves unlabelled voxels out
of precision, so the unlabelled fraction, merges (pieces of other objects >= 1,000 voxels),
merged voxels and total steps are reported for both networks next to F1.

Fixed now, before any training: **part 3 scores the student with the higher crop-11 mean F1
against the labels.** If that student's mean F1 is more than 0.05 below the teacher's on crop 11,
part 3 waits for an owner decision. The gap is first located (first-step stops, which an
on-policy round cannot fix because step 0 is the same for both; later divergence; merges); an
on-policy round (the teacher labels boxes the student visits) is proposed only for later
divergence. A student that takes many more steps than the teacher is reported as such, since part
3 also needs a 2x lower crop time.

## On-policy round (approved 2026-11-01)

**Why.** The dilated student (round 0) is 0.067 mean F1 below the teacher on crop 11 (limit 0.05).
It stops tracing too early on its own traces (docs/distil_results.md). It was trained only on
boxes the teacher visited, with the teacher's own canvas as the mask input. On its own traces the
mask input comes from its own earlier outputs and the boxes come from its own moves, which it has
never been trained on. Owner decision on 2026-11-01: "yes to recommended approach to train
further".

**Collecting samples.**
- **Tracing.** The round-0 student (`train-dilated/best.pt`) traces the same seeds as the teacher
  in crops 0-10, read from the teacher's `traces.json`. The student decides every move. Crop 11 is
  not collected (it stays the validation crop). Crop 12 is not read, and crops 13-40 are never
  read.
- **Kept steps.** The same rule as before (steps 0-3 and 1/4 of later steps), with its own random
  stream (generator seeded by crop, trace and 1).
- **Labels.** On each kept step the teacher (float32, TF32 off) is run on the same box with the
  same mask input (the student's canvas), and its output is stored as the target.
- **Move diagnostics** (kept steps; a face "moves" if its maximum reaches the move threshold):
  - faces where the teacher moves and where the student moves;
  - moves the student misses (the teacher moves, the student does not);
  - for missed moves, how far the student's face maximum is below the threshold (median, and
    the share within 0.5 logit).

  These counts use the raw network outputs, before FFN's canvas rule, and are descriptive only.
  The collection does not change the threshold.
- **Per-trace results.** The student's `benchmark.score` on each training crop, next to the
  teacher's.

**Training.**
- **Start.** From `train-dilated/best.pt`.
- **Data.** The round-0 teacher samples of crops 0-10 plus the new samples, drawn uniformly from
  the union.
- **Unchanged from round 0.** Loss, optimiser, batch and validation (the same 4,096 crop-11
  teacher-state samples).
- **Steps.** 150,000, with 500 warm-up steps and cosine decay from 3e-4. Round 0's validation loss
  changed little after about 100k steps, and the new samples are about 40% of the set.

**Method check (Cursor, 2026-11-01) and what changed.**
- *Accepted.* The move counts above cannot separate a static offset of the student's face scores
  from drift along the trace:
  - they use raw outputs, before FFN's canvas rule;
  - they record the gap to the threshold, not to the teacher;
  - they are not split by step.

  A separate diagnostic (`examples/move_diagnostics.py`, `distil.MoveGaps`) recomputes the
  student's output on stored samples, applies FFN's canvas rule to both outputs (`distil.written`;
  the stored mask input equals the old canvas box) and reports, per step bin (0, 1-3, 4-49, 50+),
  the moves of each network and the quantiles of (teacher face maximum - student face maximum).
  Following the second method check:
  - the same numbers are also reported for **frontier faces**, where the mask input's face
    maximum is below the move threshold. This includes untraced faces but also faces with low or
    negative values written earlier. Faces outside this set are not necessarily moving: FFN's
    canvas rule keeps only negative old values, so a face with a high old maximum can still end
    below the threshold;
  - quantiles of (teacher - student) are reported on frontier faces whose **mean** of teacher and
    student maxima lies within 1 logit of the threshold. These are conditional descriptive
    numbers, not an unbiased offset estimate: choosing by the mean still biases the difference if
    the two maxima vary by different amounts (Codex review). A consistent sign across step bins and
    against the on-policy samples is what would point to an offset.
  - **On crop-11 teacher samples** (no drift) with the round-0 student, a consistently positive
    difference is consistent with a static offset (descriptive; see below).
  - **On the on-policy samples** of crops 0-10, the same numbers include drift.

  Both runs take a few B200-minutes and do not change any training or selection rule.
- *Recorded as limits of this round, not changed.*
  - **Labels on drifted states.** On a student state that has already lost part of the object
    (FFN's canvas rule keeps voxels below 0 once written), the teacher's answer may itself stop.
    The label then confirms the early stop instead of correcting it.
  - **Short traces are under-represented.** Uniform mixing gives them few samples.
  - **Learning-rate restart.** The run restarts the learning rate at 3e-4 from a converged
    checkpoint.
  - **Duplicated step-0 samples.** Step-0 samples are the same as the teacher's (about 1% of the
    set, counted twice).
  - **Checkpoint choice.** `best.pt` is chosen on teacher-state validation loss and will favour
    little change from round 0, which is why `last.pt` is also traced.
  - **Crop 11 reused.** Crop 11 has now been used for several choices, so its numbers are
    optimistic.

**Selection, fixed now.**
- **Two candidates.** `best.pt` (lowest validation loss) and `last.pt` of the new run both trace
  crop 11. The one with the higher crop-11 mean F1 against the labels is the round-1 student.
- **If the gap closes.** If it is at most 0.05 below the teacher (0.898), part 3 can score it on
  crop 12.
- **If not.** The result is reported and the next step is an owner decision; there is no automatic
  second round and no threshold change.
- **Round 0 stays a candidate.** If round 1 is worse than round 0 (0.831), round 0 remains the
  candidate.

**Compute.** About 1.5 B200-hours of collection (student about 2.4 ms per step and teacher about
8 ms on the kept quarter of steps, about 1.2 M student steps), about 2 h of training, and 2 x 10
min of crop-11 traces: about 4-5 B200-hours.

**Limits.**
- Crop 11 chooses between candidates; its scores are tuning numbers.
- One round only: the new student will visit boxes the round-0 student did not.
- Training samples from the teacher's and the student's traces are mixed equally per sample, not
  per crop or trace.

## Part 3: scoring on crop 12 (approved 2026-11-01)

Owner decision on 2026-11-01 ('yes run the crop-12 test'), after round 1 closed the crop-11 gap to
0.021 (docs/distil_results.md). Written before crop 12 is read in this pilot.

**Networks.**
- **Student:** round-1 `train-dilated-r1/best.pt`, chosen by the fixed crop-11 rule. Encoder
  features in float16, decoder in float32, TF32 off, as on crop 11.
- **FFN:** the torch port of the authors' network (`export/teacher.pt`), float32, TF32 off. It
  is the teacher that the student was trained on, and it is checked against the authors' JAX
  model on the B200 (docs/distil_results.md).
- **Same tracer for both:** `distil.TorchTracer`, with the authors' movement policy and
  thresholds; only the network differs. The speed ratio therefore measures what the
  factorisation saves, not other implementation differences.

**Objects and scoring.** The 457 large objects of crop 12 (`vitseg.tuning_objects`), one seed
each, the same objects and seeds as in docs/headtohead_design.md. Each mask is
`benchmark.object_mask` of the canvas, scored with `benchmark.score`. Reported for FFN and the
student:
- mean recall, precision and F1, and the mean unlabelled fraction of the mask;
- paired per-object F1 (wins, mean, quantiles);
- merges of at least 1,000 voxels and merged voxels;
- steps;
- stops at the first step.

**Timing (one job, one B200, FFN first).**
- **Not timed:** loading the model and the crop from disk, moving the networks to the GPU, one
  warm-up trace (the first object) per network, and one encoder pass on a small tile. Timers
  synchronise the GPU at start and stop. Taking and scoring each mask is not timed.
- **FFN crop time:** copying the normalised image to the GPU, plus the wall time of the 457
  traces.
- **Student crop time:** encoding the crop (image to GPU, tiled encoder, float16 features), plus
  the wall time of the 457 traces.
- Each trace's own clock (`trace_seconds`) is reported as well.

**Pilot signal, fixed on 2026-10-31 (docs/factorised_design.md).** The student passes if both
hold:
- its mean F1 on the 457 objects is at least 0.86;
- FFN's crop time divided by the student's crop time is at least 2.

Both comparisons are inclusive and made after rounding to 12 decimals: in floating point an F1
of exactly 0.86 computes as 0.8599999999999999, and the rounding removes only that error. The
signal is decided only when all 457 objects were scored; a smoke run on fewer objects gives no
verdict. Both are reported whatever the outcome. Nothing is tuned on crop 12; there is no second run with
other settings.

**For context only, not part of the signal.**
- The recorded crop-12 result of FFN in `fp16_head32` with the JAX fast tracer on a V100
  (docs/headtohead_results.md: mean F1 0.891). FFN's F1 here is compared with it per object
  after the run, on the local copies of `crop12.json` and the recorded `ffn_crop12.json`; the
  GPU job does not read the recorded result. The
  networks' number formats and GPUs differ, so these numbers are not expected to match exactly.
- The speed of that fast tracer is not compared: it ran on another GPU, and float16 was not
  tried for the student.

**Limits stated in advance.**
- Crop 12 is a tuning crop for earlier work in this project (the SegNeuron threshold and
  checkpoint) and FFN's network may have been trained on it. The student, trained to copy FFN,
  inherits that. Crop-12 numbers are a pilot signal, not test results.
- Crops 13-40 are not read.
- One job, one B200; no repeat to measure the spread of the timings.
- The speed comparison is in float32 for both networks. A float16 FFN (3.0-3.2x faster per step
  on a V100 here) is not compared, and a float16 student was not tried.

**Data.** Crop 12 (image and proofread labels) is copied to MIT from the local cache and checked
against the sha256 in docs/manyseed_crops.json by `manyseed.verify_crop` at load.

## Timing study on crop 11 (approved 2026-11-01)

Owner request on 2026-11-01 ('yes set up the timing job ... run these tests and have the results
ready'), after the crop-12 pass (docs/distil_results.md, part 3). A measurement, not a test: no
pass mark, nothing is selected on crop 12, and crop 12 is not read. Crop 11 (the validation crop)
is used so that crop 12 is not looked at again.

**Question.** How fast are the student and FFN on one B200 when (a) needless copies between GPU
and CPU are removed, (b) the networks run in float16, or in float16 with a float32 output layer,
and (c) several objects are traced at once; and what does each change do to accuracy?

**Changes to the code, all optional (the defaults keep the earlier behaviour).**
- **Encoder output straight to the GPU.** `fz.encode_volume(..., out_device=...)` writes each tile
  into GPU memory instead of CPU memory followed by one copy of the whole crop. The features must
  be bitwise equal to the earlier path.
- **Copy back only the visited box.** `TorchTracer.trace(..., box_only=True)` copies back only the
  bounding box of the boxes the trace evaluated (and its seed), instead of the whole 500^3 canvas.
  `distil.full_canvas` rebuilds the full canvas afterwards, outside the timer; it must equal the
  full copy exactly (outside the visited boxes the canvas is NaN).
- **Number formats.** `distil.MixedNet` runs a network in `fp32`, `fp16` or `fp16_head32`:
  - in `fp16`, convolutions, ReLUs and residual additions run in float16 with float16 weights;
  - LayerNorm (statistics and its scale and offset) is computed in float32 in all formats, because
    squares of the encoder features (up to 25,584) overflow float16. Flax applies the scale and
    offset in float16; the difference is small and not measured;
  - the residual stream stays in float16, as in the JAX layout, so values above 65,504 would
    overflow: each run counts non-finite network outputs on the GPU and reports the count;
  - `fp16_head32` is like `fp16`, except that the head (LayerNorm, ReLU, 1x1x1 convolution) runs
    in float32, the variant accepted for FFN in docs/head32_design.md;
  - inputs and outputs are float32.

  For the student, the encoder is unchanged: float32 compute, features stored in float16.
- **Several objects at once.** `distil.BatchTracer` keeps up to W traces open, evaluates their
  boxes in one network call per round, and copies the face maxima of all W back in one transfer.
  Each trace keeps its own canvas and FFN's own movement policy on the CPU, so the order of steps
  within a trace is FFN's. When a trace ends, the next object takes its place in the same round.
  Like the one-at-a-time tracer, it still reads the seed's canvas value from the GPU before each
  step (FFN's own check); removing those small reads, and reusing canvases instead of allocating
  one per object, are left for later and would help both tracers. With a network whose output for a box does not depend on the batch,
  each trace must equal its single-object trace exactly; on the GPU, batched convolutions may
  round differently, so identity is reported, not assumed.

**Runs (one job, one B200; same 512 crop-11 objects and seeds as before; order as listed).**

| run | network | format | tracer | canvas copy |
|---|---|---|---|---|
| S1 | student | fp32 | one at a time | whole canvas (as on crop 12) |
| S2 | student | fp32 | one at a time | visited box |
| S3 | student | fp16 | one at a time | visited box |
| S4 | student | fp16_head32 | one at a time | visited box |
| S5 | student | fp32 | 8 at once | visited box |
| S6 | student | fp16_head32 | 8 at once | visited box |
| S7 | student | fp16_head32 | 32 at once | visited box |
| F1 | FFN | fp32 | one at a time | visited box |
| F2 | FFN | fp16 | one at a time | visited box |
| F3 | FFN | fp16_head32 | one at a time | visited box |
| F4 | FFN | fp16_head32 | 8 at once | visited box |

- **Encoder:** timed twice, by the earlier path (CPU then GPU) and straight to the GPU.
- **Not timed:** loading, seeds, one warm-up trace before each run (for batched runs, one batched
  warm-up of W traces), and scoring.
- **Timed:** the wall time of all 512 traces of each run.
- **Crop time:** the student's is encoding (straight to the GPU) plus a run's traces; FFN's is the
  image copy plus a run's traces.

**Reported for each run.**
- Accuracy: mean F1, recall and precision against the labels, merges (summed over traces), steps.
- Effect of the format: per-object F1 change against the fp32 run of the same network
  (quantiles), and the objects whose F1 falls by more than 0.1.
- Speed: total wall time, ms per step, and the speed-up against F1 (FFN fp32).
- Identity, compared by step count and a sha256 of each mask:
  - S1 against S2 (must be identical: only the copy differs);
  - S5 against S2, S6 and S7 against S4, F4 against F3 (batched against one at a time).

**Limits stated in advance.**
- Crop 11 chose the student; its F1 here is a tuning number. Speed does not depend on that.
- One job, one GPU, no repeat.
- FFN here is the torch port, not the authors' JAX fast tracer; float16 here is torch's, not
  XLA's.
- No run here changes the student's or FFN's weights. The movement queue stays on the CPU (a GPU
  queue is not part of this study).

**After the smoke run (job 27621107, 10 objects; 2026-11-01).** The float32 student runs
behaved as designed (S1, S2 and S5 gave the same scores and steps). The float16 student runs
failed: mean F1 0.02-0.03 against 0.96, hundreds of merges, and about 100 s per object, so S3,
S4, S6 and S7 cannot finish 512 objects in a 6 h job. Cause, measured on one crop-11 box (the
first seed): the student's residual stream reaches 70,661 after the fourth decoder block, above
float16's largest value (65,504); it becomes infinite and the logits become NaN. Changes:
- **Scaled stream (added runs S8, S9).** `MixedNet(..., scale=1/16)` multiplies the residual
  stream by 1/16: the features, the mask stem's last convolution and each block's last
  convolution are scaled (exact for a power of two, unless a weight becomes subnormal in float16).
  Every LayerNorm ignores the scale of its input, so the network is the same in exact arithmetic
  apart from LayerNorm's epsilon (1e-6, against a stream variance in the millions). S8 is
  fp16_head32 one at a time, S9 and S10 the same 8 and 32 at once; their accuracy is compared
  with S2, and S9 and S10 are checked for identity against S8. This is
  not the JAX layout; the plain runs S3, S4, S6 and S7 are reported from the smoke run only.
- **Jobs.** Job A: S1 S2 S5 F1 F3 F4 (512 objects). Job B (after a 20-object smoke run of S8 and
  S9: F1 0.962 for both S2 and S8, no non-finite outputs): S2 S8 S9 S10 F1 F2, F2 last because FFN's plain float16 may overflow too. The two jobs run on
  different B200 nodes; speeds are compared within a job.
- `timing.json` is rewritten after every run, so a job stopped by its time limit keeps the runs it
  finished.

**Follow-up (approved 2026-11-02, task 20261102-053834-77cc06cd).** Run F5: FFN in fp16_head32 with
the stream scaled by 1/16, 32 objects at once, box-only copy. Job C runs S10, F1, F4 and F5 on the
same 512 objects, one B200, so that:
- student against FFN at the same settings is S10 against F5, within one job;
- the effect of the scale on FFN is F5 against F4 (F4 is unscaled, 8 at once, so width also
  differs; no scaled one-at-a-time FFN run is made);
- F5's accuracy is compared with F1, and its non-finite outputs are counted as for every run.
S10, F1 and F4 repeat runs of jobs A and B; their run fingerprints show whether the traces repeat
exactly on another node. No pass mark.

**Plain float16 first (owner direction 2026-11-02, task 20261102-113319-447b4113).** "The goal is
to increase the speed of the tracing. If float16 works fine for FFN then it should be shown so;
only if that doesn't work we can motivate the need to change the head precision." Job D, one
B200, 512 objects: S2 S10 S11 F1 F5 F6 F7, where
- F6: FFN, plain fp16 (head included), stream x 1/16, 32 at once; F7: the same one at a time;
- S11: student, plain fp16, stream x 1/16, 32 at once;
- F6 against F5 and S11 against S10 differ only in the head's format; F7 against F2 (job B,
  unscaled) differs only in the scale (across nodes; F1 and S2 repeat exactly across nodes);
- F6 against F7 is an identity pair (batching).
'Works' is judged from mean F1 and its per-object change against F1 or S2, merges and non-finite
outputs, stated as observed; no pass mark is set in advance.

Written down while job D was running, before its results were seen (after the Cursor consult of
this task). Plain float16 counts as **not working** for a network, and the float32 head as
motivated, if plain float16 shows any of these and the float32-head run of the same job (F5 or
S10) does not:
- non-finite outputs;
- F6 and F7 not identical on all 512 objects (batching changing traces);
- a one-sided loss against the float32 run: a lower mean F1 change, more objects falling by over
  0.1 than rising, or clearly more merges, beyond the spread the float32-head run shows against
  the same float32 run (F5 against F1: +0.004, 4 falls and 8 rises; S10 against S2: -0.0003, 4
  falls and 6 rises).
The last point needs judgement, since one run per setting gives no error bar; the comparison is
reported in full either way. Paired comparisons F6-F5, S11-S10 and F7-F2 (job B) are computed from
the full records after the job. Not run: the student in plain float16 one at a time, so student
batching identity in plain float16 is not checked.

## GPU numerics (B200)

- The torch teacher on the B200 (float32, TF32 off) against the authors' JAX logits recorded on
  the CPU for the 15 reference-trace inputs of part 1: maximum absolute difference < 1e-3.
- The torch teacher traces the notebook's seed on the notebook crop on the B200: step count and
  history compared with the recorded reference (reported; GPU arithmetic may change a near-tie).

## Compute and data

- Teacher traces about 2-3 B200-hours (crops split over jobs), training 2 x <= 5 h, crop-11
  traces 2 x <= 1 h: about 15 B200-hours in `mit_normal_gpu` (6 h limit, 2 GPUs per user).
- Copied to MIT `/home/dushanw/orcd/pool/liconn`: crops 0-11 (image and proofread labels, from the
  local cache, sha256 checked against docs/manyseed_crops.json), the teacher state file and the
  numerics-check inputs. Crops 12-40 are not copied in this part.
- Environment on MIT: a new venv with this repository's pinned requirements (CPU JAX, needed for
  FFN's movement policy) plus torch 2.7.1+cu128.

## Limits stated in advance

- Off-policy distillation: the student is trained only on states the teacher visits; its own
  errors lead to states it has not seen. On-policy rounds are not part of this milestone.
- The teacher may have been trained on data that include these crops.
- Crop-11 trace scores choose between the two decoders; they are tuning numbers.
- Student traces cannot match the teacher's exactly by design (box context, dilation).
- Seeds sit at the deepest point of large labelled objects, as in the crop-12 protocol; the
  student sees no other starting points.
- The notebook crop is used only for the GPU numerics check; nothing from it enters training.
