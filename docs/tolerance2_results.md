# fp16_head32 under the owner's 2% rule (decided 2026-10-23)

**Owner decision (2026-10-23):** keep `fp16_head32` and judge with a 2% tolerance: at
most 6 of 300 new merges and at most 6 of 300 new early stops (a per-seed precision or
recall fall of more than 10 points), instead of 3 of 300. Everything else is unchanged:
mean recall and mean precision may each fall by at most 1 point; runs must be complete,
on pinned inputs, with all 300 seeds.

**Outcome: under the 2% rule, `fp16_head32` passes on both seed sets, and plain float16
still fails.**

**Important: this rule was set after the fresh-seed result** (docs/fresh_results.md,
where `fp16_head32` failed the original rule with 5 new merges). A pass under it is a
pass under a revised tolerance, not a pre-registered confirmation. The first 300 seeds
also took part in choosing `fp16_head32` (docs/head32_results.md).

No new runs: the recorded, pinned runs are re-judged by examples/judge_tolerance2.py
(liconn_v2/tolerance2.py). The original rule's code and all earlier evidence are
unchanged. tests/test_tolerance2.py recomputes the report and checks it against the
recorded one.

| seed set | variant | new merges | new early stops | original rule (3 of 300) | 2% rule (6 of 300) |
|---|---|---|---|---|---|
| first 300 | float16 | 3 | 10 | fails | **fails** |
| first 300 | `fp16_head32` | 1 | 1 | passes | **passes** |
| fresh 300 | `fp16_head32` | 5 | 3 | fails | **passes** |

Mean changes, all within the 1-point limits: first 300, float16 recall -0.11 and
precision -0.31 points, `fp16_head32` recall +0.27 and precision +0.13 points; fresh
300, `fp16_head32` recall -0.16 and precision -0.48 points.

## What passing under 2% means here

- Speed: 2.19x (first 300) and 2.21x (fresh) per step against float32 on V100s, and
  2.07x per whole trace on the fresh seeds, where merges add steps.
- Cost in accuracy, fresh seeds: 5 of 300 traces (1.7%) gain a new merge, all
  dendrites, each leaking into one neighbouring neuron. 3 of 300 (1.0%) stop early. 2
  merges are avoided and 4 traces extend. Mean IoU against the proofread segmentation
  is 71.45% against float32's 71.92%; merged voxels are 5.47 M against 2.93 M.
- **The 2% limit holds only for axons and dendrites pooled.** Among dendrites alone,
  the fresh new-merge rate is 5 of 114 (4.4%), against 0 of 72 in the first 300.
- One-sided 95% upper bounds on the fresh new-merge rate are 3.5% overall and 9.0% for
  dendrites, assuming independent seeds (optimistic).
- ERL (first 300 seeds; fresh ERL withheld by the sanity gate, docs/fresh_results.md):
  `fp16_head32` 6.99 um against float32's 7.18 um, and 3.22 against 3.93 um on
  dendrites. ERL here is an adapted, crop-bounded single-trace measure, comparable
  between networks in this project but not with published ERLs.

## Limits

The 2% rule was chosen after the result it lets pass. One volume, one depth range,
V100 only. Dendrites carry the merges, and the 95% bound on the dendrite new-merge
rate is about 9%.
