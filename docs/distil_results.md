# Factorised FFN pilot, part 2: distillation results (MIT B200)

Design and fixed rules: docs/distil_design.md. Evidence (copied from MIT
`/home/dushanw/orcd/pool/liconn/runs/distil-20261031`) is in
`~/ResearchFlow/data/liconn-v2/runs/distil-20261031`. Crop 12 was not read; crops 13-40 were never
read. All crop-11 numbers are validation (tuning) numbers, not test results.

The controller check of this task covers the code and the local CPU tests only. The GPU numbers
below come from the MIT job logs and records listed in each section. The controller's Codex review
did not see them, so they have not been independently verified.

## Outcome

- **Timing study (crop 11): at the same settings, the student is 1.5x faster than FFN.** Both in
  float16 with a float32 head and the stream scaled by 1/16, 32 objects at once, on one B200 (job
  C): student 150 s for the crop, FFN 228 s. Against FFN in float32 one at a time (1,181 s) the
  student is 7.9x faster, but FFN gains 5.2x from the same changes. Per step the student is 2.5x
  cheaper; it takes 17% more steps, and its float32 encoder (45 s) is 30% of its crop time. Accuracy
  barely moves (student mean F1 0.876 in both formats; FFN 0.898 -> 0.902). Unscaled float16
  overflows in the student's residual stream; with the stream divided by 16 (a change not in the
  approved plan) no run has non-finite outputs.
- **Plain float16 (head included), job D: for FFN it is slower and less accurate than float16
  with a float32 head; for the student it is slightly faster but less accurate.** With the stream
  divided by 16 and 32 at once, FFN in plain float16 (F6) takes 241 s against 230 s with the
  float32 head (F5): its step is 1.6% cheaper but it takes 6.8% more steps. Its mean F1 is 0.893
  against 0.898 in float32 (17 objects fall by over 0.1, 11 rise; 133 merges against 112), where
  F5 gains (+0.004, 4 falls, 8 rises). The student in plain float16 (S11) takes 149 s against
  154 s (S10) and has fewer merges (307 against 377), but loses 0.003 mean F1 against float32 (13
  falls, 9 rises; lower recall) where S10 loses 0.0003. Both meet the "not working" criteria
  written before the results, which favour accuracy; for the student this is a trade-off of 3%
  speed against a small accuracy loss.

- **Part 3 (crop 12): the pilot signal passed, by small margins, with many more merges.** On the
  457 crop-12 objects the student scores mean F1 0.866 (threshold 0.86; FFN 0.888) and is 2.24x
  faster than FFN on the same B200 (threshold 2), both in float32. It has 751 merges of at least
  1,000 voxels against FFN's 102. 637 of them come from 16 runaway traces.

- **Round 1 (on-policy) closes most of the gap.** The selected decoder is round 1 best.pt
  (dilated, 1, 2, 4, 1): crop-11 mean F1 0.876 against 0.898 for the teacher. The gap of 0.021 is
  within the fixed 0.05 limit, so under the design part 3 may score it on crop 12.
- **New failure in round 1: lower precision and more merges.** Precision fell from 0.984 to 0.952
  (teacher 0.972). The student takes 17% more steps than the teacher and has three times as many
  merges of at least 1,000 voxels (339 vs 112).
- **Round 0** (teacher samples only): the dilated decoder reached 0.831, the undilated one 0.696.
  The gap of 0.067 was above the limit. Recall was lower (0.753 vs 0.868), and 85% of the gap
  came from the 159 objects where the student took fewer than 0.8x the teacher's steps. This split
  by step count describes where the gap sits; it is not a test of its cause. First-step stops were
  rare (one for the teacher, one for the dilated student). Whether the traces split at the first move's
  position (13% location agreement) or later was not tested.

## GPU numerics of the teacher (job 27535229; first run 27493125)

The torch teacher on the B200 (float32, TF32 off) on the 15 recorded inputs:
- maximum absolute difference to the authors' JAX logits is 4.48e-5 (limit 1e-3);
- 0 voxels change side of the threshold.

The notebook reference trace on the GPU matches the recorded reference hash for hash:
- the same 359 steps and step history;
- the same visited region (`predicted_mask_sha256` hashes the non-NaN canvas);
- the same thresholded masks: seed component (94,534 voxels) and notebook mask (94,905 voxels).

The first run (job 27493125, kept in `check-27493125/`) compared only the visited region and the
seed-component size. The Codex review pointed this out, and the check was rerun with the mask
hashes. The teacher runs at 7.5-7.8 ms per step. Evidence: `check/gpu_port.json`.

## Teacher traces (jobs 27493886, 27493887)

| crop | traces | steps | samples | teacher mean F1 |
|---|---|---|---|---|
| 0 | 526 | 166,444 | 43,337 | 0.867 |
| 1 | 508 | 172,098 | 44,516 | 0.902 |
| 2 | 503 | 174,072 | 44,963 | 0.891 |
| 3 | 474 | 178,715 | 45,994 | 0.868 |
| 4 | 509 | 159,197 | 41,370 | 0.900 |
| 5 | 464 | 170,205 | 43,644 | 0.904 |
| 6 | 434 | 154,276 | 40,011 | 0.884 |
| 7 | 440 | 186,552 | 48,190 | 0.895 |
| 8 | 523 | 178,895 | 46,559 | 0.901 |
| 9 | 497 | 159,590 | 41,581 | 0.897 |
| 10 | 468 | 160,814 | 41,481 | 0.893 |
| 11 (validation) | 512 | 158,257 | 41,290 | 0.898 |

- 481,646 training samples (crops 0-10) and 58 GB of samples in total.
- 7.9-8.0 ms per teacher step, including sample recording.
- Evidence: `teacher/crop-KK/done.json` and `traces.json`.

## Training (jobs 27516950 dilated, 27516951 undilated)

- **Evidence.** `train-NAME/train_log.json` and the job logs. The checkpoints (`best.pt`,
  `last.pt`) stay on MIT.
- **Steps.** 400,000 steps for each decoder (batch 32, about 27 passes over the training samples).
  The count was set from a 200-step probe (job 27512280: 0.034 s per step, 100 GB peak GPU memory)
  to fit in 5 h. The runs took 4.0 h and 3.9 h.
- **Float16 encoder features.** Every stored value in all 12 crops is finite (largest 25,584;
  float16 limit 65,504). On 8 crop-11 boxes, the largest difference from float32 is 8.0 on
  features up to 22,631 (relative 3.5e-4, the float16 resolution).
- **First run failed.** The first submission (jobs 27514969, 27514970) stopped at the first
  validation: `distil.faces` could not reshape an empty batch, i.e. a validation batch with no
  step-0 sample. It was fixed with an explicit face size and a test, and both runs were
  resubmitted. No training result came from the failed jobs.

Validation on 4,096 crop-11 samples (all 512 step-0 samples included):

| | dilated (1,2,4,1) | undilated (1,1,1,1) |
|---|---|---|
| loss at step 0 / 10k / 100k / 400k | 1.521 / 0.276 / 0.265 / 0.263 | 1.532 / 0.280 / 0.272 / 0.269 |
| best checkpoint (lowest loss) | step 383,000, loss 0.2631 | step 386,000, loss 0.2688 |
| voxel agreement at 0.6, interior / shell | 0.980 / 0.981 | 0.976 / 0.976 |
| step-0 teacher face moves the student also makes (of 3,066) | 99.7% | 100% |
| step-0 face moves only the student makes | 2 | 6 |

- **Loss levels off.** Validation loss on teacher states changes little after about 100k steps for
  both decoders. Trace quality was measured only for the best checkpoint, so whether longer
  training would change the trace gap is untested; the loss curve gives no sign that it would.
- **First-step face maxima.** The "both move with face maxima within 2 voxels" figure (13% and 7%)
  is hard to interpret. The teacher's own moving faces are flat near their maximum (crop-11 step-0
  samples, script `face_flatness.py`):
  - the median face has 132 of its 289 voxels within 0.01 logit of the maximum;
  - the median face maximum is 2.94.

  A small change in logits can therefore move the selected position far along a face, and a low
  location agreement is expected even for a close student. It does not show that the trajectories
  are equivalent either.
- **Float16 ties.** 58% of faces have an exact tie in the stored float16 logits. The tracer
  selects positions from float32 logits, so those ties are partly a storage effect, and they
  lower the teacher-side location figure.

## Student traces on crop 11 (jobs 27520420 dilated, 27520421 undilated)

Same 512 seeds, the authors' movement policy and thresholds, decoder in float32 with TF32 off.

| | teacher | dilated | undilated |
|---|---|---|---|
| mean F1 (labels) | 0.898 | **0.831** | 0.696 |
| mean precision / recall | 0.972 / 0.868 | 0.984 / 0.753 | 0.956 / 0.591 |
| mean F1 against the teacher's mask | - | 0.824 | 0.690 |
| objects better / worse than teacher | - | 110 / 402 | 38 / 474 |
| paired F1 difference, median (10th pct) | - | -0.017 (-0.281) | -0.145 (-0.537) |
| stops at the first step | 1 | 1 | 3 |
| total steps | 158,257 | 116,153 | 98,813 |
| merges >= 1,000 voxels / merged voxels | 112 / 6.27 M | 59 / 0.73 M | 66 / 3.65 M |
| mean unlabelled fraction | 0.146 | 0.166 | 0.152 |
| trace time (diagnostic) | 1,265 s with recording | 279 s | 249 s |

- **Time per step:** about 2.4 ms for the student against 8.0 ms for the teacher; the teacher
  time includes sample recording. Fewer steps also lower the student's time. Neither number is
  the part-3 timing, which needs FFN and student timed the same way on the same GPU.

**Where the dilated student's gap comes from.** The analysis uses the per-object records in
`trace-dilated/student_crop11.json`; d is the teacher's F1 minus the student's.

| student / teacher steps | objects | share of gap | mean d |
|---|---|---|---|
| < 0.25 | 24 | 4% | 0.061 |
| 0.25-0.5 | 39 | 39% | 0.343 |
| 0.5-0.8 | 96 | 42% | 0.151 |
| 0.8-1.25 | 339 | 18% | 0.019 |
| >= 1.25 | 14 | -3% | -0.084 |

- 137 objects lose more than 0.1 F1; together they make up the whole gap. The student took fewer
  steps in 130 of them.
- The student's fewer merges come with this early stopping. They are not evidence of better
  boundaries.

## Face moves on teacher states (job 27552531)

Script `examples/move_diagnostics.py`, round-0 best.pt, all 41,290 crop-11 teacher samples.
FFN's canvas rule is applied to both outputs. A face counts as a move when its maximum is at or above
the threshold. "Frontier" faces are those where the mask input is below the threshold over the
whole face. T-S is the teacher's face maximum minus the student's, on faces where the teacher
moves.

| step bin | faces | teacher moves | student moves | both | missed | median T-S |
|---|---|---|---|---|---|---|
| 0 | all = frontier | 3,066 | 3,059 | 3,057 | 9 | -0.09 |
| 1-3 | all | 8,280 | 8,280 | 8,249 | 31 | -0.02 |
| 1-3 | frontier | 23 | 15 | 13 | 10 | +0.99 |
| 4-49 | all | 30,650 | 30,649 | 30,611 | 39 | -0.02 |
| 4-49 | frontier | 30 | 29 | 23 | 7 | +0.41 |
| 50+ | all | 167,086 | 167,024 | 166,706 | 380 | -0.02 |
| 50+ | frontier | 233 | 181 | 134 | 99 | +0.67 |

- Over all faces the student misses 459 of 209,082 teacher moves. On all faces (the "all" rows)
  where the teacher moves, the median T-S is between -0.09 and -0.02 in every step bin; on
  frontier faces after step 0 it is positive (next point). These conditional medians do
  not rule out an offset near the threshold or in a subset of faces.
- On frontier faces after step 0, it misses about 40% of the teacher's moves (116 of 286) and
  makes 55 moves of its own (47 at 50+), on the teacher's own states. On the faces where the
  teacher moves, the median T-S is positive.
  These are errors without drift. Whether they caused the lost voxels in the round-0 traces was
  not tested: there was no check of which faces the tracer would have queued, no paired
  trajectory comparison and no intervention. The counts are small and the quantiles are
  descriptive.

## On-policy round 1 (jobs 27550344-5 collection, 27550350 training, 27550351-2 traces)

Design: "On-policy round" in docs/distil_design.md. Evidence: `onpolicy/crop-KK/traces.json` and
`done.json` (the samples stay on MIT), `train-dilated-r1/`, `trace-dilated-r1/`,
`trace-dilated-r1-last/` and the job logs.

**Collection.** The round-0 dilated student (`train-dilated/best.pt`) traced crops 0-10 from the
teacher's seeds (5,346 traces, 1.23 M steps; the teacher took 1.86 M). At each kept step the
teacher labelled the student's box and mask:
- 323,801 new samples;
- the student's mean F1 per crop was 0.791-0.836, the teacher's 0.867-0.904;
- of 1,692,545 teacher face moves at these steps, the student also made 98.1%, missed 32,990
  (half within 0.5 logit of the threshold) and added 9,434 of its own;
- 556-659 s per crop on one B200.

**Training.** 150,000 steps from `train-dilated/best.pt` on 805,447 samples (481,646 teacher plus
323,801 on-policy), 1.7 h. The validation set is unchanged: 4,096 crop-11 teacher samples.
- The validation loss hardly moves: 0.2631 at the start, best 0.2628 at step 139,000, 0.2628 at
  step 150,000. Voxel agreement is 0.980 / 0.982 (interior / shell) at the
  best checkpoint and stays within 0.979-0.980 / 0.980-0.982 over the run.
- The trace F1 rose by 0.046 (below), so the loss on teacher states does not track trace quality.
  The best.pt/last.pt choice by this loss is close to arbitrary; both were traced.

**Crop-11 traces** (same 512 seeds and settings as round 0):

| | teacher | round 0 | round 1 best.pt | round 1 last.pt |
|---|---|---|---|---|
| checkpoint step | - | 383,000 | 139,000 | 150,000 |
| mean F1 (labels) | 0.898 | 0.831 | **0.876** | 0.875 |
| gap to teacher | - | 0.067 | **0.021** | 0.022 |
| mean precision / recall | 0.972 / 0.868 | 0.984 / 0.753 | 0.952 / 0.856 | 0.951 / 0.855 |
| mean F1 against the teacher's mask | - | 0.824 | 0.865 | 0.862 |
| objects better / worse than teacher | - | 110 / 402 | 227 / 285 | 227 / 285 |
| stops at the first step | 1 | 1 | 1 | 1 |
| total steps | 158,257 | 116,153 | 184,606 | 182,330 |
| merges >= 1,000 voxels / merged voxels | 112 / 6.27 M | 59 / 0.73 M | 339 / 15.6 M | 337 / 15.4 M |
| objects with precision < 0.9 | - | 16 | 40 | - |
| trace time (diagnostic) | 1,265 s with recording | 279 s | 416 s | 412 s |
| plus encoder, once per crop | - | 54 s | 58 s | 58 s |

- **Selected by the fixed rule: round 1 best.pt** (higher crop-11 mean F1). Its gap, 0.021, is
  within the 0.05 limit, so under the design part 3 may score it on crop 12.
- **New failure: lower precision.** Recall is close to the teacher's, but precision fell below it
  (0.952 vs 0.972). The student now takes 17% more steps than the teacher and merges 2.5x as many
  voxels into other labelled objects (339 merges of at least 1,000 voxels, against 112).

Where the round-1 gap comes from (d = teacher F1 - student F1, script `locate_gap_r1.py`):

| student / teacher steps | objects | share of gap | mean d |
|---|---|---|---|
| < 0.25 | 6 | 4% | 0.075 |
| 0.25-0.5 | 12 | 20% | 0.186 |
| 0.5-0.8 | 37 | 40% | 0.120 |
| 0.8-1.25 | 408 | 1% | 0.000 |
| >= 1.25 | 49 | 34% | 0.076 |

- The split is by the final step counts of each trace. It describes where the gap sits, not
  what caused it: no trajectory was compared step by step.
- Objects where the student took fewer than 0.8x the teacher's steps hold about two-thirds of
  the gap (55 objects; 159 in round 0). Objects where it took at least 1.25x hold about a third
  (49 objects; 14 in round 0).
- The bins mix cases. In the < 0.25 bin, two objects have d below -0.6: the teacher took
  1,724 and 1,449 steps and scored far lower than the student.
- 53 objects lose more than 0.1 F1: in 33 the student took fewer steps, in 20 more.

**Limits of round 1.**
- No control. Round 1 both added on-policy samples and trained 150,000 more steps (learning
  rate restarted). No run trained the same extra steps on teacher samples alone, so the gain is
  that of the combined round, not of the on-policy samples by themselves.
- One round only. The labels are on states of the round-0 student; the round-1 student's own
  states were not relabelled.
- Crop 11 was used to choose round 0's decoder, both round-1 checkpoints, and to decide that a
  round was needed. Its 0.876 is a tuning number. Crop 12 is the first unseen test.
- The merge count matters more for a whole-crop segmentation than for single-object F1. Part 3
  will measure it on crop 12.
- The times are diagnostic. The student's extra steps raise its trace time; the speed pass mark
  (>= 2x lower crop time than FFN on the same GPU) is measured only in part 3.

## Part 3: crop 12 (job 27604027; smoke run 27602699)

Protocol: "Part 3: scoring on crop 12" in docs/distil_design.md. Evidence:
`score12-dilated-r1/crop12.json` (summary, per-object records of both networks),
`ffn_masks.npz`, `student_masks.npz`, `analysis.json` (from `score12_analysis.py`), and the job
log. One B200; job time 40 min. Crops 13-40 were not read.

**Pilot signal: passed, by small margins.**

| | threshold | result |
|---|---|---|
| student mean F1 (457 objects) | >= 0.86 | **0.866** |
| FFN crop time / student crop time | >= 2 | **2.24** (1,074 s / 479 s) |

| | FFN (float32 port) | student (round 1 best.pt) |
|---|---|---|
| mean F1 / recall / precision | 0.888 / 0.860 / 0.965 | 0.866 / 0.850 / 0.942 |
| objects with F1 >= 0.9 | 320 | 296 |
| objects better / worse than the other | 265 | 192 |
| merges >= 1,000 voxels (objects with a merge) | 102 (62) | **751 (91)** |
| merged voxels | 3.7 M | **20.3 M** |
| mean unlabelled fraction | 0.153 | 0.207 |
| steps | 137,696 | 191,794 (1.39x) |
| stops at the first step | 0 | 1 |
| ms per step (trace clock) | 7.79 | 2.21 |
| traces, wall time | 1,073 s | 424 s |
| one-time cost | 0.06 s (image to GPU) | 54 s (encoder) |
| crop time | 1,074 s | 479 s |

- **Speed.** Each student step is 3.5x cheaper, but the student takes 39% more steps and pays
  54 s once for the encoder; together 2.24x. The median object takes 0.62 s against 2.06 s.
- **Paired F1.** Median student minus FFN -0.002; 10th percentile -0.148, 90th +0.028.
- **FFN check.** The same 457 objects and seeds as the recorded crop-12 run of FFN in
  `fp16_head32` on a V100 (`headtohead-20261029/ffn_crop12.json`, mean F1 0.891). Per object,
  float32 here minus that run: median 0.000, mean -0.002; 177 objects within 0.001. Different
  number formats and GPUs, so exact agreement is not expected.

**Runaway traces.** 16 student traces took at least 3x FFN's steps (descriptive grouping, chosen
after seeing the results):
- they hold 637 of the 751 merges and 18.4 M of the 20.3 M merged voxels;
- they took 68,183 steps and 137 s of the student's 424 s of tracing (FFN: 29 s on them);
- on the other 441 objects, mean F1 is 0.887 for the student and 0.899 for FFN, with 114 merges
  for the student (FFN 102 on all 457).

The F1 gap split by step ratio (as for crop 11; d = FFN F1 - student F1; it shows where the gap
sits, not its cause):

| student / FFN steps | objects | share of gap | mean d |
|---|---|---|---|
| < 0.25 | 5 | -7% | -0.143 |
| 0.25-0.5 | 13 | -11% | -0.085 |
| 0.5-0.8 | 33 | 44% | 0.136 |
| 0.8-1.25 | 360 | 12% | 0.003 |
| >= 1.25 | 46 | 62% | 0.139 |

**Limits.**
- Crop 12 is a tuning crop for earlier work, and FFN's network may have been trained on it; the
  signal is a pilot signal, not a test result.
- Both networks ran in float32. Our fastest FFN (float16 network, float32 head, fast tracer) was
  3.0-3.2x faster per step than float32 FFN on a V100; this run does not show that the student
  is faster than that FFN.
- One job, one GPU, no repeat: the spread of the timings is unknown. The speed margin (2.24 vs 2)
  is about 12%.
- The F1 margin is 0.006. On the 16 runaway traces the student's mean F1 is 0.288 against
  FFN's 0.596. That difference alone accounts for 0.011 of the 0.022 gap in mean F1, more than
  the margin, so a few traces decide whether the signal passes.
- Per-object F1 does not count merges. The 751 (and FFN's 102) are summed over independent,
  possibly overlapping traces: a neighbour reached by several traces counts several times. How
  many distinct merges a consistent whole-crop segmentation would have was not measured.

## Timing study on crop 11 (jobs 27628922 A, 27630568 B; smoke runs 27621107, 27628923)

Design: docs/distil_design.md, "Timing study on crop 11", including the note added after the first
smoke run. Evidence extract: docs/distil_timing_evidence.json (job summaries, identity checks,
format effects, sha256 of the full records, logs and sources). Full records:
`timing-dilated-r1-a/timing.json` and `timing-dilated-r1-b/timing.json`. Student: round 1
best.pt. All 512 crop-11 objects, the same seeds as before, one B200 per job, TF32 off.

| run | network, format | objects at once | ms per step | crop seconds | speed-up vs F1 | mean F1 | merges |
|---|---|---|---|---|---|---|---|
| F1 (A) | FFN fp32 | 1 | 7.46 | 1,180.5 | 1.00 | 0.898 | 112 |
| F3 (A) | FFN fp16_head32 | 1 | 3.65 | 585.6 | 2.02 | 0.900 | 118 |
| F4 (A) | FFN fp16_head32 | 8 | 1.52 | 244.1 | 4.84 | 0.900 | 118 |
| F2 (B) | FFN fp16 | 1 | 3.65 | 593.6 | 1.98 | 0.893 | 110 |
| S1 (A) | student fp32, whole-canvas copy | 1 | 2.26 | 462.9 | 2.55 | 0.876 | 339 |
| S2 (A) | student fp32 | 1 | 1.98 | 411.9 | 2.87 | 0.876 | 339 |
| S5 (A) | student fp32 | 8 | 1.38 | 301.3 | 3.92 | 0.876 | 339 |
| S8 (B) | student fp16_head32, stream x 1/16 | 1 | 1.37 | 301.3 | 3.91 | 0.876 | 377 |
| S9 (B) | student fp16_head32, stream x 1/16 | 8 | 0.62 | 161.8 | 7.28 | 0.876 | 377 |
| S10 (B) | student fp16_head32, stream x 1/16 | 32 | 0.58 | 154.5 | 7.62 | 0.876 | 377 |

Crop seconds: the student's include the encoder (45.7 s, straight to the GPU); FFN's include the
image copy (0.05 s). Speed-ups in the table are within each job (B's F1: 1,177.8 s, 0.2% from A's). Merges are
summed over the 512 independent traces.

- **Jobs A and B: between the fastest settings measured there, the student was 1.6x faster.** FFN
  F4 (float16 with a float32 head, 8 at once; job A) takes 244 s; the student S10 (float16 with the
  scaled stream, 32 at once; job B) 155 s, and S9 (8 at once, as for F4) 162 s, 1.5x. The settings
  are not the same: FFN was not run 32 at once or with a scaled stream, and F4 and S9/S10 ran on
  different nodes (the float32 runs F1 and S2 took 0.2-0.4% longer in A than in B and gave
  identical traces on both). The 7.6x is against FFN in float32 one object at a time. The encoder,
  run once in float32, is now 30% of the student's crop time. Job C (below) replaces this
  comparison with one at equal settings.
- **Each change, student.** Copying back only the visited box: 2.26 -> 1.98 ms per step. Eight
  objects at once: -> 1.38. Float16 with the scaled stream: 1.37 one at a time, 0.62 eight at once,
  0.58 at 32. The study's starting point (S1, the crop-12 setup) is 2.55x faster than F1, close to
  crop 12's 2.24x.
- **Identity.** Every pair that should match did, on all 512 objects (same steps, same mask, same
  history): S1-S2, S5-S2, S9-S8, S10-S8, F4-F3. Batching on the B200 did not change a single
  trace, and the box-only copy is exact. S2 and F1, run in both jobs, are also identical across
  the two nodes (run fingerprints in the evidence file).
- **Encoder.** Writing tiles straight to the GPU gives bitwise-equal features (45.7 s against
  51.0 s); no feature is non-finite.
- **Float16 changes accuracy little once the stream fits.** Student S8 against S2: mean F1 change
  -0.0003, median 0.000, 10th and 90th percentiles -0.004 and +0.003. Four objects fell by more
  than 0.1 (65230, 102735, 102804, 103622) and six rose by more than 0.1. Merges 339 -> 377 and
  steps 184,606 -> 186,072. FFN F3 against F1: +0.003 mean, five falls and eight rises over 0.1.
  FFN F2 (plain float16): -0.0045 mean, 15 falls and 10 rises over 0.1.
- **Plain float16 breaks the student (smoke run, 10 objects).** S3 and S4 reached mean F1 0.019
  and 0.032, against 0.963 in float32, with 739k and 581k steps instead of 2.7k. One crop-11 box
  (job 27626254) shows why: the decoder's residual stream reaches 70,661 after the fourth block,
  above float16's 65,504, and the logits become NaN. Dividing the stream by 16 (the features, the
  mask stem and each block's last convolution; LayerNorm ignores the scale) removes the overflow:
  no non-finite output in S8, S9 or S10. S3, S4, S6 and S7 were not run on all 512 objects (about
  100 s per object).
- **FFN in float16 has non-finite outputs too.** F3 and F4 counted 4,472 non-finite output values,
  F2 6,094 (warm-up included). Their scores stay at the float32 level, but which traces were
  affected is not recorded. F5 (job C: rescaled and 32 at once) has none.

**Limits.**
- Crop 11 chose the student; its F1 here is a tuning number. No crop-12 or test-crop number here.
- One run per setting, no repeats; two nodes (A and B). The 1.6x and 1.5x figures compare runs
  from different nodes and different batch widths or stream handling.
- FFN here is the torch port with this repository's tracer, not the authors' JAX fast tracer;
  float16 is torch's.
- The 1/16 scale was not in the approved plan. It was chosen from one box, after the smoke run.
  It makes 187 of 138,400 scaled student weights and 1,119 of 581,280 scaled FFN weights subnormal
  in float16 (none become zero; worst relative rounding of those weights 1.9% and 10.8%).
- Not tried: float16 encoder, a movement queue on the GPU, removing the per-step reads of the seed
  value, reusing canvases.

### Follow-up: job C (27656204), student and FFN at the same settings

Design: docs/distil_design.md, "Follow-up (approved 2026-11-02)". Task 20261102-053834-77cc06cd.
One B200, the same 512 objects and seeds; job C ran exactly the current sources (sha256 in the
evidence file).

| run | network | format | objects at once | ms per step | steps | crop seconds | speed-up vs F1 | mean F1 | merges | non-finite outputs |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | FFN | fp32 | 1 | 7.46 | 158,257 | 1,181.3 | 1.00 | 0.898 | 112 | not counted |
| F4 | FFN | fp16_head32 | 8 | 1.51 | 160,371 | 242.2 | 4.88 | 0.900 | 118 | 4,472 |
| F5 | FFN | fp16_head32, stream x 1/16 | 32 | 1.43 | 159,191 | 227.8 | 5.19 | 0.902 | 106 | 0 |
| S10 | student | fp16_head32, stream x 1/16 | 32 | 0.57 | 186,072 | 150.3 | 7.86 | 0.876 | 377 | 0 |

- **Student against FFN at equal settings (S10 against F5): 1.52x faster for the crop.** The
  student's step is 2.5x cheaper, and its traces take 2.2x less time, but it takes 17% more steps
  and pays 45.0 s for the encoder (30% of its 150.3 s). Its mean F1 is 0.026 lower (0.876 against
  0.902), with 377 merges against 106.
- **F5 against F4: width and scale change together, so neither effect is isolated.** F5 (32 at
  once, scaled) takes 1.43 ms per step against F4's 1.51 (8 at once, unscaled), and has no
  non-finite outputs against F4's 4,472. Every F5 trace differs somewhere from F4's (none
  identical). No FFN run holds one of the two fixed.
- **F5's accuracy matches float32.** Against F1, mean F1 +0.004; four objects fall by more than
  0.1 (60991, 64752, 102947, 148097) and eight rise.
- **Repeats are exact.** S10, F1 and F4 in job C give the same run fingerprints (steps, masks and
  histories of all 512 objects) as in jobs A and B, on another node.
- **Encoder.** Straight to the GPU again bitwise equal to the earlier path (45.0 s against 51.1 s).

**Caveats.**
- The student's encoder runs in float32; float16 there was not tried. Its share of the crop time
  depends on how many objects are traced per crop (512 here, 457 on crop 12).
- The student's path rescales its 32-channel feature box at every step, while FFN's scale sits in
  its weights; this small cost works against the student. Per-step reads of the seed's value and
  one new canvas per object are shared overheads, a larger share of the student's shorter step.
- MixedNet casts features to float16 before scaling; float32 features above 65,504 would overflow
  at the cast. Crop-11 features are float16 already (no non-finite value).
- LayerNorm's epsilon becomes 256 times larger relative to the stream. F5's accuracy (no loss
  against F1) is consistent with this not mattering for FFN; it was not measured separately.
- One run per setting; crop 11 chose the student and the scale; torch port, not the authors'
  JAX tracer.

### Plain float16 first: job D (27684986)

Design: docs/distil_design.md, "Plain float16 first (owner direction 2026-11-02)", including the
"not working" criteria written while job D ran, before its results were seen. Task
20261102-113319-447b4113. One B200, the same 512 objects and seeds. All float16 runs here divide
the stream by 16; "plain float16" means the head is float16 too (LayerNorm statistics are still
taken in float32 and the output is cast to float32, as in all float16 runs).

| run | network | format | objects at once | ms per step | steps | crop seconds | speed-up vs F1 | mean F1 | merges | non-finite outputs |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | FFN | fp32 | 1 | 7.46 | 158,257 | 1,180.6 | 1.00 | 0.898 | 112 | not counted |
| F5 | FFN | fp16_head32 | 32 | 1.44 | 159,191 | 229.7 | 5.14 | 0.902 | 106 | 0 |
| F6 | FFN | plain fp16 | 32 | 1.42 | 169,988 | 241.2 | 4.89 | 0.893 | 133 | 0 |
| F7 | FFN | plain fp16 | 1 | 3.61 | 169,988 | 614.1 | 1.92 | 0.893 | 133 | 0 |
| S2 | student | fp32 | 1 | 1.97 | 184,606 | 410.6 | 2.88 | 0.876 | 339 | not counted |
| S10 | student | fp16_head32 | 32 | 0.58 | 186,072 | 153.9 | 7.67 | 0.876 | 377 | 0 |
| S11 | student | plain fp16 | 32 | 0.57 | 179,359 | 148.7 | 7.94 | 0.873 | 307 | 0 |

Per-object comparisons (F1 change; objects falling or rising by more than 0.1; from the full
records):

| pair | what differs | mean F1 change | falls | rises | identical traces |
|---|---|---|---|---|---|
| F5 - F1 | float16 with float32 head, 32 at once | +0.0039 | 4 | 8 | 0 |
| F6 - F1 | plain float16, 32 at once | -0.0045 | 17 | 11 | 0 |
| F6 - F5 | head format only | -0.0084 | 19 | 9 | 1 |
| F7 - F2 (job B) | stream scale only (one at a time; other node) | +0.0000 | 14 | 12 | 0 |
| S10 - S2 | float16 with float32 head, 32 at once | -0.0003 | 4 | 6 | 0 |
| S11 - S2 | plain float16, 32 at once | -0.0031 | 13 | 9 | 0 |
| S11 - S10 | head format only | -0.0028 | 12 | 7 | 1 |

- **Against the pre-written criteria.** Plain float16 has no non-finite outputs, and F6 and F7
  are identical on all 512 objects (batching does not change its traces). It does show a
  one-sided loss against float32 that the float32-head run does not, for both networks: FFN -0.0045
  mean F1, 17 falls against 11 rises and 133 merges against 112 (merged voxels 9.69 million
  against 6.27 million; precision 0.965 against 0.972), where F5 shows +0.004 with 4 falls and 8
  rises; the student -0.0031 with 13 falls against 9 rises (recall 0.850 against 0.856; merges
  fewer, 307 against 339), where S10 shows -0.0003 with 4 falls and 6 rises. By those criteria
  plain float16 is **not working** for either network, and the float32 head is motivated. The
  losses are small, and one run per setting gives no error bar.
- **Dropping the float32 head buys almost no speed.** Per step, plain float16 is 1.6% cheaper
  for FFN (1.42 against 1.44 ms) and 1.3% for the student (0.570 against 0.577 ms). FFN's crop
  is 5% slower (241 against 230 s) because it takes 6.8% more steps; the student's is 3% faster
  (149 against 154 s) because it takes 3.6% fewer.
- **The FFN accuracy loss persists without observed non-finite outputs.** Unscaled plain
  float16 (F2, job B) had 6,094 non-finite outputs; scaled (F7) has none, and its mean F1 is about
  the same (0.8932 both). This does not show what caused F2's errors: no trace is identical, 14
  objects fall and 12 rise by over 0.1 between the two, so their errors differ and may cancel in
  the mean, and scaling also changes rounding and LayerNorm's effective epsilon. The two runs are
  on different nodes; F1 and S2 repeat exactly across nodes.
- **Repeats are exact.** S2, S10, F1 and F5 give the same run fingerprints as in jobs A, B and C.
- **Encoder.** Straight to the GPU again bitwise equal to the earlier path (46.5 s against 51.1 s).

**Caveats.** The student was not run in plain float16 one at a time, so its batching identity in
plain float16 is not checked. The stream scale was chosen after the smoke run, not in the
original plan. Non-finite counts include warm-up steps. Crop 11 is a tuning crop.

## Interpretation and limits (round 0)

- **Divergence along the trace, not first-step stops.** The student agrees with the teacher on 98%
  of voxels on teacher states, yet on its own traces it stops expanding earlier in about a third
  of objects. This fits off-policy drift (the student's slightly different canvas feeds its next
  inputs, and the error compounds), but that mechanism has not been measured directly.
  - **Divergence point not measured.** The traces could split as early as the first move's
    position, which differs often. No paired trajectory comparison was made to find where they
    split.
  - **Next step under the fixed rule.** For divergence after the first step, the rule names an
    on-policy round (the teacher labels boxes the student visits). That round also covers boxes
    reached after a different first move.
- **Not tested here.** Whether a calibration offset at the move threshold (the student's face
  maxima sitting slightly below the teacher's) causes the early stops has not been measured.
- **Dilation matters.** The undilated control has a smaller receptive field and loses three times
  as much F1 (0.20).
- **Tuning numbers only.** Crop-11 scores chose the decoder. They are not test results. The teacher
  may have been trained on data including these crops.
