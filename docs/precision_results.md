# Lever B, part 2: lower precision. Results (runs 2026-10-21)

**Outcome: float16 passes the owner-confirmed 1% tolerance on the 12 benchmark seeds
and is 2.19x faster per step on a V100. bfloat16 fails.** The float16 pass is a
candidate, not an established speedup:
- every seed's mask differs visibly from the reference;
- a similar small numerical change (bfloat16) created a large new merge on one seed;
- held-out seeds have not been tested.

This is an implementer-run record of owner-approved runs on Wahab V100s:
- screen at batch 1: Slurm job 7369485;
- float16 benchmark: job 7369486;
- bfloat16 benchmark: job 7369487.

These are re-runs. The first runs (screen 7365849 at batch 8, benchmarks 7365855 and
7365856) are kept as `superseded-*`. The re-runs evaluate the screen at batch 1, as
in tracing, and count infinite canvas values. Every one of the 24 benchmark traces
reproduced the first run bitwise (canvas and visit order). The batch-1 screen gave
the same decision changes and non-finite counts as batch 8.

tests/test_precision_evidence.py checks the following against
docs/precision_evidence.json:
- the code that ran is the checked code;
- the outputs are the recorded ones;
- every screened trace matched the reference;
- the selection follows the rule fixed in advance;
- the comparisons and timings below follow from the outputs.

A first screen (job 7365841) failed while writing results, after all 12 traces had
matched the reference. A 16-bit variant produced non-finite outputs, which are not
valid JSON. The screen was changed to count non-finite outputs as changed decisions,
and was re-run.

## 1. Screen (1,054 recorded inputs, batch 1; Tesla V100-SXM2-16GB; float32 14.69 ms per evaluation)

| variant | decisions changed | non-finite outputs | time per evaluation | speedup |
|---|---|---|---|---|
| fp16 | 0.0061% | 0.0001% (in 5 inputs) | 3.96 ms | 3.71x |
| fp16_ln32 | 0.0061% | 0.0001% (in 5 inputs) | 4.05 ms | 3.63x |
| bf16 | 0.0215% | 0 | 10.02 ms | 1.47x |
| bf16_ln32 | 0.0215% | 0 | 10.03 ms | 1.47x |

For comparison, the best single block drop changed 1.18% of decisions
(docs/block_drop_results.md). The `_ln32` variants gave the same decisions as their
16-bit counterparts. A likely reason, not verified, is that the LayerNorm
implementation already computes its statistics in float32. Under the rule fixed in
advance, fp16 and the bfloat16 variant with fewer changed decisions (bf16, by name
order in a tie) went to the benchmark.

## 2. Benchmark against the reference (12 seeds, closed loop)

| variant | passes | mean recall | mean precision | merged voxels | total steps | time per step |
|---|---|---|---|---|---|---|
| reference (float32) | | | | | 8,407 | 19.67 ms |
| fp16 | **yes** | +1.16 points (10 seeds: +1.32) | +7.82 (+1.21) | -93.9% (-47.8%) | 6,117 (10 seeds: 5,734 vs 5,841) | **8.99 ms (2.19x)** |
| bf16 | no | +0.12 (+0.02) | +0.87 (-6.97) | +93.1% (+1,579%) | 9,685 (9,298) | 14.78 ms (1.33x) |

The 10 seeds exclude dendrite_1525 and axon_218. Positive recall and precision
changes are improvements.

fp16, per seed:
- no seed is outside the per-seed rules (largest recall loss: dendrite_56789, -0.72
  points);
- notable changes: axon_5506 +13.37 recall points (367 steps instead of 318);
  dendrite_61046 +12.98 precision points (742 steps instead of 931);
- dendrite_1525 again avoids its false merge (merges 1 -> 0; precision +81.72 points;
  376 steps instead of 2,558), as with postponing in lever A;
- every seed is flagged for review: mask IoU with the reference is 0.81-0.97
  (dendrite_1525: 0.18).

bf16 fails on dendrite_61046: it gains a new large merge (object 55261). The trace
grows from 931 to 4,357 steps, and precision falls by 69.13 points. dendrite_1525
also changes strongly: it loses its merge into 97123 but merges with another object
(IoU 0.18; precision +80.16 points; 375 steps). The other ten seeds change less
(IoU 0.91-0.98).

Speed: the network alone runs 3.7x faster in float16 on this GPU. Per step the
gain is 2.19x, because the rest of each FFN step (canvas updates and movement, in
Python) is unchanged. All 12 traces took 55 s instead of 165 s in total, also helped
by dendrite_1525's shorter trace. This is a hardware effect, from the V100's float16
tensor cores; the network's FLOPs are unchanged. Other GPUs and CPUs will differ.

## What this does and does not show

- On these 12 seeds, float16 evaluation of the authors' network meets every
  accuracy rule and halves the trace time on a V100.
- Small numerical changes alter the traced objects. Both 16-bit variants changed
  every seed's mask by several percent. Whether a merge forms or not can flip either
  way: float16 removed two merges, and bfloat16 added a large one. The float16 pass on
  12 seeds may therefore partly reflect which merges these seeds happen to have.
  **Before treating float16 as a speedup it must be confirmed on new held-out seeds**,
  with the same tolerance.
- Rare non-finite outputs occur in float16 (5 of 1,054 screened inputs). In the 24
  benchmark traces no canvas value is infinite. A NaN output cannot be told apart
  from a never-predicted voxel in the canvas, so NaN outputs are not ruled out.
- V100 only; A100 (native bfloat16, and float16 tensor cores) not tested.
