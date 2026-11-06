# Unseen crops 41-50: pre-registered plan (owner-approved 2026-11-04)

Nothing in this plan has been run. No image or label of any crop numbered 13 or higher was read to
write it; only crop positions and labelled fractions (docs/fresh_crops.json and the pinned coarse
coverage map that chose every crop so far).

## Questions

On crops that nothing in this project has read:
1. Is FFN's fast mode (`fp16_head32`) as accurate as float32 FFN, object by object?
2. How accurate and how fast are FFN and the fine-tuned SegNeuron?
3. How accurate and how fast are the three smaller FFNs (distilled student, P8, P4) against FFN?

## Crops (owner decisions 2026-11-04)

- **Crops 41-50**: positions 41 to 50 of the crop order fixed on 2026-10-21 (`manyseed.select_crops`,
  `default_rng(0)` over grid crops with at least 50% labelled coarse cells, not overlapping crops A
  and B). They are the only qualifying positions left: crops 0-40 took the first 41 of 51, and that
  match was checked. None of them has been fetched.

  | crop | x0, y0, z0 | labelled fraction |
  |---|---|---|
  | 41 | 100, 4083, 111 | 0.509 |
  | 42 | 2600, 1583, 111 | 0.570 |
  | 43 | 3100, 4083, 111 | 0.519 |
  | 44 | 100, 3083, 111 | 0.534 |
  | 45 | 600, 3583, 111 | 0.585 |
  | 46 | 1600, 3583, 111 | 0.584 |
  | 47 | 2600, 3583, 111 | 0.580 |
  | 48 | 2100, 1583, 111 | 0.584 |
  | 49 | 600, 4083, 111 | 0.523 |
  | 50 | 2100, 583, 111 | 0.591 |

- Ten crops, as decided; all ten are used, so no draw is needed.
- Crops 13-40 are not read here and stay in reserve. They are no longer clean for fast mode (see
  "Disclosed in advance") but are unseen by SegNeuron and the smaller FFNs.
- All crops of this project lie in one volume (ExPID82_1) at one depth (z0 = 111). The test says
  nothing about other depths, samples or volumes.

## Fetching (outside data; owner approval needed)

`manyseed.fetch_crop` from the authors' public bucket, as for crops 0-40: raw image (uint8) and
proofread labels (uint32), 500^3 each, about 0.6 GB per crop, 6 GB in all, cached locally next to
crops 0-40. Each file's sha256 is pinned in a new docs/unseen_crops.json at fetch time, before any
method runs. The fetch writes only files; it computes no statistics on them. The crops are then
copied to MIT scratch and to Wahab.

## Objects and scoring (as on crops 11 and 12)

- Objects: `vitseg.tuning_objects` of each crop, which seeds every object of `vitseg.large_objects` at
  `benchmark.seed_point` (the code that chose the crop-11 and crop-12 objects), in object-ID order.
  Every object is traced; there is no object cap in the test runs. The number of objects per crop is
  reported (about 450-520 on crops 11 and 12).
- Each FFN-family method traces each seed on its own canvas, scored with `benchmark.score`.
- SegNeuron: the authors' `inference.py`, then our pipeline at the fixed threshold 0.6; each object
  scored by `vitseg.seed_records`.
- Per object: recall, precision, F1, steps, and merges as in `benchmark.score` (another object with
  at least 1,000 voxels in the mask). Both the number of such merged partners and the number of
  objects with at least one are reported.
- Runaway traces (descriptive, not a pass/fail rule): a trace with at least 3x the steps of FFN
  float32 (F1) on the same object, the grouping used on crop 12 (docs/distil_results.md, where it
  was chosen after seeing that result). Counted for S10, P2, Q2 and F5 against F1, and for S10, P2
  and Q2 against F5 as well. Objects on which the reference stopped at its first step are listed,
  since 3x of one step is not a meaningful bound.
- Classes: the volume's proofread classes (`benchmark.load_classes`, 855 axons and dendrites in the
  whole volume) label only some objects; the rest are unclassified.

## Methods and frozen settings (nothing is chosen on these crops)

| run | method | number format | objects at once | residual stream scale | checkpoint (sha256) |
|---|---|---|---|---|---|
| F1 | FFN | float32, TF32 off | 1 | 1 | export/teacher.pt (123c8e92...52389) |
| F5 | FFN | `fp16_head32` | 32 | 1/16 | export/teacher.pt (123c8e92...52389) |
| S10 | student | `fp16_head32` | 32 | 1/16 | train-dilated-r1/best.pt, step 139,000 (1cef5c3d...77ae8) |
| P2 | P8 | `fp16_head32` | 32 | 1/16 | train-p8-r1/best.pt, step 124,000 (01eac60f...cf055) |
| Q2 | P4 | `fp16_head32` | 32 | 1/16 | train-p4-r1/last.pt, step 150,000 (fc88a63b...c4533) |
| V | FFN on the V100 | `fp16_head32` (reference port, fast tracer) | 1 | 1 | the authors' checkpoint, as on crop 12 |
| - | SegNeuron | threshold 0.6 | - | - | fine-tuned train/best.ckpt (e726bb43...5f228) |

F1-Q2 are the B200 runs of the crop-11 timing job E (examples/time_crop11.py; torch port of FFN,
`distil.TeacherNet`, `BatchTracer` for 32 at once); the checkpoints are the files job E used (paths
relative to the MIT run folder distil-20261031). The full sha256s are fixed in liconn_v2/unseen.py
and checked before each run. Run V is the crop-12 head-to-head's FFN (docs/headtohead_design.md).

## Pre-stated analysis

**Question 1: fast mode against float32 (a confirmation; the rule is fixed now).** F5 against F1,
both on the B200 torch stack, on all objects of the ten crops pooled (N objects), each object paired
with itself. With d = F5 minus F1 per object and EPS = 1e-9 (as in liconn_v2/manyseed.py):
- new merges, d(precision) < -0.10 - EPS: count <= 0.02 N + EPS;
- new early stops, d(recall) < -0.10 - EPS: count <= 0.02 N + EPS;
- mean d(recall) >= -0.01 - EPS; mean d(precision) >= -0.01 - EPS;
- all ten crops complete, every object traced, on pinned inputs.

This is the owner's 2% rule (docs/tolerance2_results.md) moved from 300 seeds to every object. It
was set after the fresh-seed result on crops 13-40; applied here before crops 41-50 are read, it is
a pre-registered test. Fast mode passes only if every line passes. What it confirms is the B200
fast mode of Figs 2c and 3c (32 objects at once, stream scaled by 1/16), not the V100 fast tracer
that the 300-seed tests judged.

Reported with it, not part of the pass/fail rule:
- the counts per crop;
- the counts for classified dendrites, classified axons and unclassified objects separately. On the
  fresh seeds all five new merges were in dendrites (docs/fresh_results.md), so a pooled pass over
  every object, where dendrites are a smaller share than among the seeds, can hide a higher rate in
  dendrites. A pass is stated as a pass pooled over large objects, with the dendrite rate beside it;
- objects on which F1 already merged (precision below 0.2), where a further merge can hardly lower
  precision by 10 points (manyseed.ALREADY_MERGED_PRECISION).

**Questions 2 and 3.** The unit is the crop (n = 10). For each crop, the unweighted mean per-object
F1 of each method; then the paired per-crop differences d_k (k = 41-50), with their mean and 95%
t-interval (t with 9 degrees of freedom). The t-interval decides every claim below. A percentile
bootstrap over crops (10,000 resamples, `numpy.random.default_rng(20261104)`) is reported as a check
and decides nothing. Intervals are per comparison, not adjusted for the five comparisons. Pooled
per-object means (every object weighted equally), merges and runaways are reported as well.
- FFN against SegNeuron (tuning gap: 0.07 F1 on crop 12): estimate and interval, from F5. The same
  comparison with run V (the crop-12 protocol: same V100, reference port, one object at a time) is
  reported beside it, with the per-object F1 agreement of V and F5. If the two give different claims
  (the interval excludes 0 for one and not the other), both are reported and neither is preferred.
  As on crop 12: the mean F1 of both methods without objects whose seed SegNeuron puts in predicted
  background, and the voxels claimed by two or more of FFN's independent traces (run V).
- Student, P8 and P4 against F5, with a non-inferiority margin fixed now of 0.02 F1. With [L, U] the
  interval of (method minus FFN):
  - L > -0.02 and U >= 0: "not worse than FFN by more than 0.02";
  - L > -0.02 and U < 0: "worse than FFN, by less than 0.02";
  - L <= -0.02 and U < 0: "worse than FFN";
  - L <= -0.02 and U >= 0: "inconclusive".
  (If L > 0, "better than FFN" is stated as well.)
- P8 against the student: a difference is claimed only if the interval excludes 0.
- Speed: crop time per method (B200, one run each), per object and per crop, and the speed-up over
  F1 from the ten crops' totals. Break-even object count of FFN (run V) against SegNeuron on the
  V100, both ways as in Fig. 4: end to end (primary, as in the text) and GPU only; per crop (median,
  range) and from the ten crops' totals.

Every number above is reported whatever it shows. No method, crop, object, threshold or rule is
dropped or changed after results are seen. A task that fails for an infrastructure reason (node
failure, preemption, file-system or out-of-memory error unrelated to a method's numbers) may be rerun
unchanged, and the failure is recorded. A failure caused by a method itself (non-finite outputs, a
time-out because traces run away) is a result for that method and is reported, not rerun.

Method failures, concretely. A run fails on a crop if it did not finish (non-finite encoder features
stop the student's run; a crash or time-out leaves the run or the result file missing) or if any
network output during the run was non-finite where it is counted: the float16 runs F5, S10, P2 and
Q2 (time_crop11's count; all four had none on crop 11, job E) and SegNeuron's affinities and
foreground. The float32 run F1 and run V do not count non-finite outputs (the tracers do not
instrument them), so for those two only a run that stops is a failure. A
failed run's crop is not scored for that method, and every comparison that needs the method is
reported as "not judged", with the crops and reasons; nothing is filled in. Question 1 fails if F5
fails on any crop (it requires all ten crops complete) and is not judged if F1 fails.

Provenance. Every result file records the design's sha256, the crop files' sha256s, and the
checkpoints' sha256s, and each B200 file its run's settings (network, number format, objects
at once, box_only, stream scale, TF32 off); the judge (examples/judge_unseen.py) refuses a file whose values differ from
those fixed here (settings: liconn_v2/unseen.py RUN_SETTINGS, the table above) and in
docs/unseen_crops.json, and refuses to run if the design changed after the
crops were fetched (the manifest records the design's sha256 at fetch time; a second fetch never
rewrites the manifest and refuses to run if the design changed). It also refuses any
file whose records are not every object of its header, in order. The class split uses
liconn_v2/proofread_classes.json, whose sha256 is fixed in liconn_v2/unseen.py. Time
limits are 2 h per task, as when a crop's runs shared one job; that is about 6x the longest
single task (B200: F1, about 20 min; V100: FFN or SegNeuron, about 20 min each). A completed run is final.

## Hardware and compute

- MIT B200, one job per crop and run (an array of 50; cluster/mit/unseen.sbatch): each of S10,
  F1, F5, P2 and Q2 on each crop is its own Slurm task writing its own result file, so a crash or
  time-out of one method cannot stop or delay another. About 35 min of runs per crop from crop-11
  times (F1 alone about 20 min), plus a few minutes per task to load the network; about 6 GPU-hours
  in all. Runs of a crop may land on different B200 nodes; the GPU and node of every task are
  recorded, and the speed comparisons are between B200s of the same model rather than one device.
- Wahab V100, one job per crop and method (two arrays of ten; cluster/wahab/unseen.sbatch with
  METHOD=ffn or METHOD=segneuron): FFN `fp16_head32` one object at a time, and SegNeuron (its
  environment exists only there), each its own job, so a crash or time-out of one cannot stop the
  other. Both use V100s of the same partition for the speed comparison, as in Fig. 4, but not
  necessarily the same node; the GPU of every job is recorded. FFN's accuracy in the decisive
  comparisons comes from the B200 run F5; run V's accuracy is reported beside it (question 2).

## Code (written and checked before anything runs)

- A fetch step for crops 41-50 only, pinning their sha256s.
- One run script per site that accepts only crops 41-50 and refuses any other, records the sha256
  of this plan, the checkpoints and the crop files, and writes one result file per crop and run
  (MIT: crop-K/RUN.json for each B200 run; Wahab: crop-K/ffn.json and crop-K/segneuron.json).
- A judge script that computes the pre-stated analysis from the result files alone, with tests.
- Smoke runs on crop 11 (validation) only, never on a crop of 41-50.

## Disclosed in advance

- FFN's network was trained by the volume's authors on proofread data from this volume, possibly
  including these regions.
- The student, P8 and P4 were trained on crops 0-10 with FFN as teacher and selected on crop 11.
  SegNeuron was fine-tuned on crops 0-11, with its checkpoint and threshold chosen on crop 12.
- Crops 13-40 were read before for 300 seeded float32 and `fp16_head32` FFN traces
  (docs/fresh_results.md), and the 2% rule was set after that result. That is why fast mode is
  tested on crops 41-50 and not there.
- One run per method; no training repeats, so training-to-training spread is not measured. Speed is
  one run per method per GPU.
- Ten crops resolve gaps of a few F1 points; smaller gaps are judged against the 0.02 margin.
- The crops are treated as independent, but they are not fully: several abut (41 and 49, 45 and 49,
  42 and 48 share a face), and a neuron can be a large object in two crops. The number of object IDs
  scored in more than one crop is reported. Objects within a crop are not independent either, which
  is why the crop is the unit for questions 2 and 3; question 1's pooled counts carry no correction.
- Crops 41-50 are the last ten positions of the fixed order. They are not a fresh random sample of
  the volume, but their order was set by the fixed random permutation, not by choice. Four of the ten
  lie on the outermost grid columns or row used (x0 = 100 or 3100, or y0 = 4083), against 16 of 28
  among crops 13-40; the labelled fraction (0.51-0.59) matches crops 13-40 (0.51-0.59).
