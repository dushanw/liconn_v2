# Held-out confirmation of float16: results (runs 2026-10-21)

**Outcome: float16 does not pass the owner-confirmed 1% tolerance on the held-out
seeds.** Two held-out dendrites form large false merges that float32 does not, and a
third loses 4.1 recall points. The benchmark pass (docs/precision_results.md) did
not generalise. float16 is therefore not an acceptable speedup under the current
tolerance.

This is an implementer-run record of owner-approved runs on Wahab V100s:
- held-out float32 reference: Slurm 7370167;
- float16: 7370168;
- both on crop B with 12 seeds, as in docs/heldout_design.md.

tests/test_heldout_evidence.py checks the following against
docs/heldout_evidence.json:
- the code that ran is the checked code;
- the outputs are the recorded ones;
- the comparison below follows from them.

The code ran from a staged copy of the workspace plus the held-out files; the file
hashes identify it.

## Comparison (float16 against held-out float32)

| seed | steps fp32 / fp16 | mask IoU | recall change | precision change |
|---|---|---|---|---|
| heldout_axon_151289 | 280 / 283 | 0.9307 | +0.80 | +0.38 |
| heldout_axon_229311 | 144 / 141 | 0.9398 | -0.92 | -0.37 |
| heldout_axon_197450 | 262 / 277 | 0.9447 | +0.02 | +0.20 |
| heldout_axon_176723 | 521 / 503 | 0.9380 | -0.24 | +0.29 |
| heldout_axon_14447 | 303 / 489 | 0.6410 | +27.87 | -0.02 |
| heldout_axon_242804 | 196 / 201 | 0.9399 | -0.35 | +0.01 |
| heldout_dendrite_283334 | 238 / 254 | 0.8939 | +1.26 | 0.00 |
| heldout_dendrite_211590 | 257 / 267 | 0.9419 | -0.61 | +0.01 |
| heldout_dendrite_217911 | 351 / 347 | 0.9314 | +0.53 | +0.10 |
| heldout_dendrite_283771 | 104 / 101 | 0.9084 | **-4.07** | 0.00 |
| heldout_dendrite_245099 | 727 / 2,866 | 0.2301 | **-4.99** | **-74.77** |
| heldout_dendrite_250842 | 79 / 2,266 | 0.0660 | **-2.25** | **-93.18** |

Recall and precision changes are in percentage points.

- Merges:
  - heldout_dendrite_245099 already overlaps proofread object 245232 by 2,571 voxels in
    float32 (counted as a merge). In float16 that overlap grows to 678,477 voxels, and
    a new merge into object 238468 (13,444 voxels) appears.
  - heldout_dendrite_250842 has no merge in float32. In float16 it merges into 245232
    (669,850 voxels) and 238468 (13,976 voxels).
- Means over all 12 seeds: recall +1.42 points, precision -13.95 points.
- No infinite canvas values in either run.
- Speed: 20.06 ms per step in float32, 9.07 ms in float16 (2.21x), consistent with
  the benchmark (2.19x).

## What this shows

- With FFN's single-object traces, a numerically tiny change in the network (float16
  changes about 0.006% of per-evaluation decisions) can decide whether a trace crosses
  into a neighbouring object. On the benchmark seeds float16 happened to remove
  merges; on the held-out seeds it created two large ones. bfloat16 had already shown
  the same on one benchmark seed.
- Under per-seed rules at 1%, any change to the network's arithmetic therefore risks
  failing on some seed. Speedups that keep the computation bitwise identical (for
  example, removing the roughly 60% of each step spent outside the network; see
  docs/beyond_fp16_ideas.md) avoid this risk.
- One held-out crop of the same volume, 12 seeds, V100.
