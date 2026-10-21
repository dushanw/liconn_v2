# Lever B, part 1: dropping residual blocks. Results (runs 2026-10-20)

**Outcome: both benchmarked block drops fail the owner-confirmed 1% tolerance by a
wide margin.** In the screen, no single block's removal left the network's decisions
nearly unchanged (lowest flip rate 1.18%). The two variants chosen by the
pre-registered rule then lost 3-11 recall points on average in full traces.

This is an implementer-run record of owner-approved runs on Wahab V100s:
- screen: Slurm job 7362833;
- drop block 12: job 7363566;
- drop blocks 12 and 17: job 7363567.

tests/test_block_drop_evidence.py checks the following against
docs/block_drop_evidence.json:
- the code that ran is the checked code;
- the outputs are the recorded ones;
- every screened trace matched the reference;
- the selection follows the pre-registered rule;
- the comparisons below follow from the outputs.

The Wahab clone's git label reads `ac5ad6a-dirty`; the file hashes identify the code.
One file changed after the runs: a usage comment in cluster/wahab/block_array.sbatch
named the wrong script. It was corrected, and the evidence test checks that every
non-comment line is still the one that ran.

## 1. Screen (open loop, 1,054 recorded network inputs from the 12 reference traces)

Every screened trace matched the reference's canvas and visit order bitwise.

Single-block drops: flip rate, i.e. the share of output voxels whose decision at
logit(0.6) changes. Each block costs 3.97 of 81.6 GFLOP per evaluation (4.9%).

| lowest | flip rate | lost / gained | highest | flip rate |
|---|---|---|---|---|
| block 12 | 1.18% | 0.70% / 0.48% | block 1 | 7.0% |
| block 17 | 1.21% | 0.55% / 0.66% | | |
| block 15 | 1.24% | 0.89% / 0.35% | | |
| block 16 | 1.36% | 0.81% / 0.55% | | |
| block 10 | 1.44% | 0.83% / 0.61% | | |

The sequential greedy search, where each added block is chosen given those already
dropped:

| blocks dropped | set | flip rate | GFLOP per evaluation |
|---|---|---|---|
| 1 | 12 | 1.18% | 77.6 |
| 2 | 12, 17 | 1.69% | 73.7 |
| 3 | 12, 15, 17 | 2.30% | 69.7 |
| 4 | 9, 12, 15, 17 | 2.93% | 65.7 |
| 6 | 8, 9, 12, 15, 16, 17 | 3.27% | 57.8 |
| 10 | 3, 5, 6, 8, 9, 11, 12, 15, 16, 17 | 8.22% | 41.9 |
| 20 | all | 37.9% | 2.1 |

No single block or set reaches the 0.1% or 0.5% limits. Under the selection rule,
fixed before the screen ran (including the sequential greedy amendment; see
docs/block_drop_design.md), the benchmark received only the best single drop
(block 12) and the largest set at or below 2% (blocks 12 and 17).

## 2. Benchmark against the reference (12 seeds, closed loop)

| variant | mean recall change | mean precision change | seeds outside the 1-point recall rule | GFLOP per correct voxel, all (10 seeds) | passes |
|---|---|---|---|---|---|
| drop 12 | -3.39 points | +8.61 | 9 of 12 | -34.0% (-11.3%) | no |
| drop 12, 17 | -10.52 points | +8.90 | 11 of 12 | -36.4% (-14.8%) | no |

The 10 seeds are all except dendrite_1525 and axon_218.

Cost broken down into evaluation cost and steps:

| variant | GFLOP per evaluation | steps, all 12 | steps, 10 seeds | GFLOP per correct voxel, 10 seeds |
|---|---|---|---|---|
| reference | 81.6 | 8,407 | 5,841 | 0.2708 |
| drop 12 | 77.6 (-4.9%) | 5,488 (-34.7%) | 5,125 (-12.3%) | 0.2403 (-11.3%) |
| drop 12, 17 | 73.7 (-9.7%) | 5,100 (-39.3%) | 4,797 (-17.9%) | 0.2308 (-14.8%) |

Most of the cost reduction comes from fewer steps (traces stopping earlier), not
from the cheaper evaluations.

Largest per-seed recall losses:
- drop 12: dendrite_56789 -24.15 points, dendrite_1525 -4.32, notebook -2.01;
- drop 12 and 17: axon_10801 -39.78, dendrite_56789 -34.81, dendrite_1525 -20.39,
  axon_3574 -9.26.

Mask IoU with the reference: 0.71-0.93 (drop 12) and 0.55-0.90 (drop 12 and 17), apart
from dendrite_1525, where the merge changes (0.16 and 0.12).

**What happens:** the pruned networks trace more cautiously. Traces stop earlier and
take fewer steps (8,407 -> 5,488 and 5,100). Merged voxels fall by 93-99%, and
precision rises (most dendrites lose their small merges). But recall falls on almost
every seed. The lower GFLOP per correct voxel comes from tracing less, not from
tracing the same objects more cheaply. Per evaluation, about 1-2% of decisions change.
Over full traces, single seeds lose up to 24 recall points (one block) and 40 points
(two blocks). That is consistent with small per-step changes accumulating over
hundreds of steps, though this was not measured step by step.

## What this does and does not show

- The two benchmarked variants, dropping block 12 and dropping blocks 12 and 17, fail
  the tolerance. The other 18 single drops disturbed the network more in the screen
  (flip rates 1.21-7.0%), but were not benchmarked in full traces. Their closed-loop
  accuracy is untested, and open-loop rank need not match closed-loop rank.
- Untested hypothesis, for judging other no-training approximations such as low-rank
  layers or lower precision: since about 1.2-1.7% of changed decisions per evaluation
  failed here, an approximation would probably need a much smaller disturbance.
  These experiments do not establish a threshold. Flip rate also does not describe
  where changes occur or how they affect movement decisions.
- Retraining or fine-tuning a pruned network (not allowed for now) might recover
  accuracy. That is untested.
- Single-object traces on one crop.
