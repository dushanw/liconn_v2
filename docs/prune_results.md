# Pruned FFN, distilled: results (runs 2026-11-02/03)

Design: docs/prune_design.md. Evidence: docs/prune_evidence.json (checked by
tests/test_prune_evidence.py). MIT Engaging, one B200 per job, run directory distil-20261031. Crop 11
is the validation crop and chose the checkpoints; every number here is a tuning number. Crop 12 was
not read; crops 13-40 were never read. One training run per network; no error bars.

## Outcome

- **P8 (the last 8 of FFN's 20 blocks), without the encoder, traced crop 11 with a mean F1 close to
  the factorised student's, and at the fastest settings it was faster than both the student and FFN.**
  Crop-11 mean F1 0.874 in float32 (student 0.876, FFN 0.898). At the fastest settings (float16
  with a float32 head, stream x 1/16, 32 objects at once) it traced the 512 objects in 130 s
  against the student's 151 s and FFN's 229 s, at mean F1 0.870 (student 0.876, FFN 0.902), with
  268 merges of at least 1,000 voxels (student 377, FFN 106). The means are close, but the two
  networks differ object by object (50 objects fall and 38 rise by more than 0.1 from the student
  to P8), and this is one crop and one training run with no pass mark: it does not show that the
  two trace equally well.
- **P4 (the last 4 blocks, dilated as the student's decoder) is faster again and less accurate.**
  98 s at mean F1 0.841 (321 merges). It runs the same residual blocks per step as the student; its
  F1 is 0.035 lower than the student's at the fastest settings (0.032 lower in float32). This is
  the difference between the two inputs and training set-ups (docs/prune_design.md, Networks),
  which cannot be assigned to one cause.
- **Round 0 alone was not enough, as for the student.** It gave F1 0.741 (P8) and 0.729 (P4),
  with traces that stopped early (58-59% of FFN's steps). One on-policy round brought them to 0.874
  and 0.844; it also added 150,000 training steps, and no run separates the on-policy samples from
  the extra steps.
- **The fastest settings cost the pruned networks a little accuracy against their own float32
  runs**, unlike FFN: P8 -0.0041 mean F1 (11 objects fall by more than 0.1, 6 rise), P4 -0.0028
  (7 fall, 6 rise); FFN +0.0039 (4 fall, 8 rise) in the same job. The losses are small and come
  from one run each.

## Crop-11 traces and the choice of checkpoint

Float32, one object at a time, TF32 off (`trace_student.py`), 512 objects, seeds at the deepest
point of each large labelled object. FFN (teacher): mean F1 0.898, 158,257 steps, 112 merges. "F1
vs FFN" scores each trace against FFN's own mask for the same seed instead of the labels.

| checkpoint | step | mean F1 | recall | precision | F1 vs FFN | steps / FFN's | merges >= 1,000 voxels | traces >= 3x FFN's steps |
|---|---|---|---|---|---|---|---|---|
| P8 round 0, best | 387,000 | 0.741 | 0.634 | 0.993 | 0.746 | 0.58 | 11 | 1 |
| **P8 round 1, best** | 124,000 | **0.874** | 0.842 | 0.965 | 0.865 | 1.05 | 260 | 11 |
| P8 round 1, last | 150,000 | 0.866 | 0.845 | 0.951 | 0.863 | 1.15 | 342 | 13 |
| P4 round 0, best | 390,000 | 0.729 | 0.623 | 0.982 | 0.727 | 0.59 | 55 | 3 |
| P4 round 1, best | 142,000 | 0.840 | 0.809 | 0.946 | 0.830 | 1.11 | 338 | 18 |
| **P4 round 1, last** | 150,000 | **0.844** | 0.812 | 0.948 | 0.837 | 1.12 | 337 | 15 |
| factorised student (reference; docs/distil_results.md) | 139,000 | 0.876 | 0.856 | 0.952 | 0.865 | 1.17 | 339 | 11 |

The rule fixed in the design (highest crop-11 mean F1 among the three) selects P8 round-1 best and
P4 round-1 last (bold). The selected checkpoints were then timed on the same crop and objects
(below); the timed float32 runs reproduce these traces (same steps and exactly the same F1 for
all 512 objects).

## Timing and accuracy in one job (job E, 27770944)

One B200, the same 512 crop-11 objects and seeds, TF32 off. Float16 runs use a float32 output
head and the residual stream x 1/16 (as F5 and S10 in docs/distil_results.md). Crop time includes
the student's encoder (45.6 s) and FFN's image transfer.

| run | network | format, objects at once | crop time | vs F1 | ms per step | steps | mean F1 | recall | precision | merges | merged voxels | non-finite outputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | FFN | float32, 1 | 1,181 s | 1.00x | 7.46 | 158,257 | 0.898 | 0.868 | 0.972 | 112 | 6.27 M | not counted |
| F5 | FFN | fp16_head32, 32 | 229 s | 5.17x | 1.44 | 159,191 | 0.902 | 0.870 | 0.974 | 106 | 6.55 M | 0 |
| S10 | student | fp16_head32, 32 | 151 s | 7.84x | 0.56 | 186,072 | 0.876 | 0.855 | 0.952 | 377 | 15.9 M | 0 |
| P1 | P8 | float32, 1 | 564 s | 2.09x | 3.40 | 165,693 | 0.874 | 0.842 | 0.965 | 260 | 11.3 M | not counted |
| P2 | P8 | fp16_head32, 32 | 130 s | 9.06x | 0.77 | 168,628 | 0.870 | 0.841 | 0.961 | 268 | 12.7 M | 0 |
| Q1 | P4 | float32, 1 | 362 s | 3.26x | 2.04 | 177,183 | 0.844 | 0.812 | 0.948 | 337 | 14.8 M | not counted |
| Q2 | P4 | fp16_head32, 32 | 98 s | 12.06x | 0.56 | 173,606 | 0.841 | 0.810 | 0.946 | 321 | 13.9 M | 0 |

F1, F5 and S10 repeat job D's runs exactly (same steps, masks and visit histories for all 512
objects); their crop times differ from job D's by 0.1%, 0.5% and 2.1%.

Per-object comparisons (F1 change; objects falling or rising by more than 0.1; no trace is
identical in any pair):

| pair | mean F1 change | falls | rises |
|---|---|---|---|
| F5 - F1 (FFN, fastest against float32) | +0.0039 | 4 | 8 |
| P2 - P1 (P8, fastest against float32) | -0.0041 | 11 | 6 |
| Q2 - Q1 (P4, fastest against float32) | -0.0028 | 7 | 6 |
| P1 - F1 (P8 against FFN, float32) | -0.0239 | 56 | 27 |
| Q1 - F1 (P4 against FFN, float32) | -0.0536 | 92 | 22 |
| P2 - F5 (P8 against FFN, fastest) | -0.0320 | 60 | 27 |
| Q2 - F5 (P4 against FFN, fastest) | -0.0604 | 95 | 23 |
| P2 - S10 (P8 against student, fastest) | -0.0063 | 50 | 38 |
| Q2 - S10 (P4 against student, fastest) | -0.0347 | 75 | 29 |

- **Per step, P8 costs more than the student and P4 the same.** 0.77 and 0.56 ms against the
  student's 0.56 ms (FFN 1.44). P8 is faster per crop because it needs no encoder (the student's
  45.6 s is 30% of its crop time) and takes fewer steps (168,628 against 186,072).
- **One at a time in float32, the student is faster than P8.** The student took 411 s in job D
  (S2, 1.97 ms per step, 184,606 steps; not run in job E, so this compares two jobs), P8 564 s and
  P4 362 s in job E. The order changes at the fastest settings, where the encoder's fixed cost weighs
  more.
- **P8 and the student differ object by object** (50 objects fall and 38 rise by more than 0.1),
  although their means are close.
- **No non-finite outputs** in any float16 run with the scaled stream.
- **Against FFN's own masks** (crop-11 traces above), P8 scores 0.865 and P4 0.837, the factorised
  student 0.865.

## Training

| network | blocks kept (0-based), dilations | parameters | GFLOP per box | round 0: step-0 / best validation loss | round 1: on-policy samples added / best loss | B200 time (round 0 + round 1) |
|---|---|---|---|---|---|---|
| P8 | 12-19, 1 x 8 | 472,993 | 33.9 | 1.420 / 0.2634 (step 387,000) | 249,851 / 0.2630 (step 124,000) | 6.7 h (two jobs; resumed at step 314,408) + 2.6 h |
| P4 | 16-19, 1, 2, 4, 1 | 251,297 | 18.0 | 0.745 / 0.2654 (step 390,000) | 266,270 / 0.2648 (step 142,000) | 4.0 h + 1.6 h |

Probes (200 steps; time per step over steps 101-200): P8 0.059 s per step (13.5 GB), P4 0.035 s
(10.5 GB). 481,646 teacher samples from crops 0-10 in round 0; 4,096 fixed crop-11 samples for
validation. Validation loss
again did not track trace quality: from round 0 to round 1 it fell by 0.0004-0.0006 while
crop-11 F1 rose by 0.11-0.13. Round 1's step-0 checkpoint was excluded from best.pt as designed;
both round-1 best checkpoints are later steps. The on-policy sample counts are those the round-1
training logs report; the on-policy sample files themselves are not part of the evidence.

## Caveats

- Crop 11 chose the checkpoints and is the only crop scored. These are tuning numbers.
- One run per setting; no seed repeats; no error bars. Timings are from one job on one node.
- The fastest settings include the stream scaling chosen after the crop-11 smoke run of the
  timing study (docs/distil_results.md); it is not a pre-registered setting.
- Merges are summed over traces that may overlap.
- P4 against the student differs in image context, depth of image processing, frozen against
  trained weights, feature precision and mask combination; the F1 difference cannot be assigned
  to one of these (docs/prune_design.md, Limits).
- Only the last-K block subsets were tried.
