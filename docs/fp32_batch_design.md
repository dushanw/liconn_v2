# Float32 FFN traced many objects at once: crop 11 widths and the test crops

Owner request (2026-11-09): settle whether fast FFN's speed-up comes from batching or from reduced
precision. In the test (docs/unseen_design.md) float32 FFN ran one object at a time (F1) and fast FFN
32 at once in `fp16_head32` with the stream scaled by 1/16 (F5), so the 4.8x mixes the two. Float32
was never traced several objects at once. Owner approvals (2026-11-09): crop 11 and one B200 for a
width study; then float32 on the ten test crops. TF32 stays off (owner: no TF32 run).

This is an addition after the test results were known, like the SegNeuron B200 timing
(docs/unseen_b200_segneuron_design.md). It changes no method, checkpoint, rule or verdict of the test.

## Part 1: widths on crop 11 (one job, one B200)

The 512 crop-11 objects and seeds of the timing study (docs/distil_results.md), the torch port of
FFN (export/teacher.pt, sha256 123c8e92...52389), box-only copy, TF32 off, one warm-up per run, as
examples/time_crop11.py. Runs, in this order:

| run | format | objects at once | stream scale |
|---|---|---|---|
| F32x1 | float32 | 1 | 1 |
| F32x8 | float32 | 8 | 1 |
| F32x32 | float32 | 32 | 1 |
| F32x64 | float32 | 64 | 1 |
| H16x1 | `fp16_head32` | 1 | 1/16 |
| H16x8 | `fp16_head32` | 8 | 1/16 |
| H16x32 | `fp16_head32` | 32 | 1/16 |
| H16x64 | `fp16_head32` | 64 | 1/16 |

F32x1 is run F1 and H16x32 is run F5 of the timing study; repeating them puts all eight in one job.
Reported, with no pass mark:
- crop seconds, ms per step, steps, mean F1, merges and non-finite outputs per run (counted in
  `fp16_head32` only, warm-up included, as in the timing study; not counted in float32);
- identity (same steps, mask and history for every object) of each batched run against the width-1
  run of the same format; batching is expected to leave traces unchanged, but on the GPU batched
  convolutions may round differently, so identity is measured, not assumed;
- run fingerprints, compared with the timing study's F1 and F5 fingerprints (repeatability);
- the split of the speed-up, by crop time and by time per step (crop time also reflects changes in
  step count): precision at equal width (F32xW over H16xW) and batching within a format (Fx1 over
  FxW). Both orders are reported (batching in float32 then precision at 32; precision at width 1
  then batching in `fp16_head32`), since the two effects need not multiply independently.

What each comparison isolates:
- "Precision" is fast FFN's whole change of arithmetic: `fp16_head32` together with the stream
  scaled by 1/16. Its speed comes from the number format; its effect on traces is that of both.
- "Batching" from width 1 compares two tracers, `TorchTracer` one object at a time (timed per
  object) and `BatchTracer` (timed over the crop), as in the timing study's F3 and F4. Widths 8, 32
  and 64 use the same tracer.

## Part 2: float32 on the ten test crops

**Width fixed now: 32**, the width of every other tracer on the test crops (F5, S10, P2, Q2), so that
float32 against fast FFN differs only in number format and stream scale. Part 1 does not choose it;
if Part 1 shows another width to be faster in float32, that is reported from crop 11 only.

Run F32x32 on crops 41-50, one B200 per crop (Slurm array, one task per crop), with the same loading,
checks and crop time as the test's B200 runs (examples/run_unseen.py): teacher sha256, crop sha256s,
TF32 off, crop time = traces plus the image copy. Reported, with no pass mark:
- per crop and over ten crops: time, speed-up over F1 (batching in float32) and F32x32 over F5
  (precision at equal width);
- identity against the test's F1 traces (steps, mask and history sha256 per object, from the copied
  F1 result files, checked against their sha256 in docs/unseen_evidence.json), and separately the
  objects with the same mask; if every mask is identical, F32x32's accuracy is F1's;
- new merges, new early stops and runaways against F1 (all zero if the traces are identical), and
  objects with a merge and merged voxels as in Table 1 (then equal to F1's);
- SegNeuron's B200 break-even against F32x32, as in docs/unseen_b200_segneuron_design.md (end to end
  and without start-up), with the SegNeuron times of docs/unseen_b200_evidence.json.

A failed or stopped crop is reported as such and is not rerun silently: the per-crop results are
reported for the crops that finished, and the ten-crop totals, accuracy and break-evens only when all
ten finished. Part 2 measures one order of the split (batching in float32, then precision at width
32); Part 1 gives both orders on crop 11. Batched runs are timed after an untimed warm-up on W objects, and the
end of each crop has rounds with fewer than W open traces; both follow the timing study's method.

## Limits

- One job per setting; the timing study found repeats on other nodes within 3% (run times) and
  identical traces. On the test crops F32x32 runs in new jobs and is compared with F1 and F5 times
  from the test's jobs, on other B200 nodes.
- If batched float32 changes traces, the test-crop accuracy of F32x32 is reported against F1, not
  judged by the test's rules.
- A break-even far below 32 objects per crop cannot be reached by a tracer working 32 at once at
  its full-crop rate; break-evens are reported as for the other tracers.
- Crop 11 is a tuning crop. Its timing split is a description, not a test.
- The test-crop float32 batch run is planned after the test results; it is reported as an addition.
- Widths other than 1, 8, 32 and 64, and TF32, are not tried.
