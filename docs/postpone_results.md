# Postponing nearly enclosed steps: results (runs 2026-10-19)

**Outcome: no policy passes the owner-confirmed 1% tolerance.** The owner decided
on 2026-10-20 to record this as the outcome of lever A and to move to lever B
(cheaper network evaluations).

This is an implementer-run record of owner-approved runs on Wahab V100s, one 12-seed
job array per policy. The jobs:
- default (control): Slurm 7359122;
- skip_enclosed: 7359123;
- postpone: 7359124;
- postpone_drop_enclosed: 7359125.

tests/test_postpone_evidence.py checks the following against
docs/postpone_evidence.json:
- the code that ran is the checked code;
- the outputs are the recorded ones;
- every policy's bookkeeping adds up;
- the default control reproduces the reference bitwise;
- the comparisons below follow from the outputs.

The Wahab clone's git label reads `ac5ad6a-dirty` because the clone was last updated
at that commit and the files were synced directly. The file sha256 values identify
the code exactly.

## Results against the reference (gpu-baseline-a-20261019)

| policy | total steps | change | without dendrite_1525 and axon_218 | evals / 1k correct, all (without) | passes |
|---|---|---|---|---|---|
| default (control) | 8,407 | 0 | 5,841 (0) | 4.382 (3.318) | yes, bitwise identical |
| skip_enclosed | 7,682 | -8.6% | 5,468 (-6.4%) | 4.015 (3.115) | no |
| postpone | 6,174 | -26.6% | 5,787 (-0.9%) | 3.229 (3.302) | no |
| postpone_drop_enclosed | 5,541 | -34.1% | 5,191 (-11.1%) | 2.897 (2.960) | no |

Why each fails (all rules at 1%; changes in percentage points):
- skip_enclosed: dendrite_61046 loses 1.21 recall points, over the 1-point per-seed
  limit. Without the two sensitive seeds, total merged voxels also grow 2.3%, over
  the 1% limit. Over all 12 seeds the means are within limits: recall -0.12, precision
  -0.07, unlabeled +0.16, merged voxels +0.8%.
- postpone: notebook -2.38 and axon_10801 -1.51 recall points.
- postpone_drop_enclosed: notebook -2.37 and axon_10801 -1.46 recall points.

Per seed (mask IoU with the reference / recall change in points):

| seed | skip_enclosed | postpone | postpone_drop_enclosed |
|---|---|---|---|
| axon_3574 | 0.9973 / +0.01 | 0.9636 / -0.27 | 0.9627 / -0.26 |
| axon_13395 | 0.9867 / +0.04 | 0.9580 / -0.21 | 0.9563 / -0.09 |
| axon_10801 | 0.9709 / -0.13 | 0.9414 / **-1.51** | 0.9386 / **-1.46** |
| axon_5506 | 0.9937 / +0.01 | 0.9555 / +0.14 | 0.9552 / +0.14 |
| axon_218 | 1.0000 / 0 | 1.0000 / 0 | 1.0000 / 0 |
| axon_57259 | 0.9997 / 0 | 0.9729 / +0.38 | 0.9722 / +0.39 |
| dendrite_1525 | 0.9609 / +0.03 | 0.1771 / +0.44 | 0.1773 / +0.38 |
| dendrite_99141 | 0.9814 / +0.06 | 0.9661 / -0.08 | 0.9651 / -0.04 |
| dendrite_56789 | 0.9609 / -0.71 | 0.9556 / -0.29 | 0.9539 / -0.26 |
| dendrite_61046 | 0.9630 / **-1.21** | 0.9401 / -0.21 | 0.9399 / -0.17 |
| dendrite_93375 | 0.9572 / +0.40 | 0.9604 / -0.42 | 0.9579 / -0.36 |
| notebook | 0.9996 / 0 | 0.9317 / **-2.38** | 0.9316 / **-2.37** |

Seeds flagged for review (IoU < 0.99 or a step change > 1%): 9 of 12 for
skip_enclosed, and 11 of 12 for each postpone policy (all but axon_218).

## Where the postpone savings come from

- dendrite_1525: postponing avoids its false merge into object 97123 entirely. Merges
  drop from 1 to 0, precision rises by 82.0 points, and recall by 0.44 (postpone) and
  0.38 (postpone_drop_enclosed) points. Steps fall from 2,558 to 379 and 342. This
  one seed accounts for most of the overall step reduction. It is an accuracy change
  on one seed, not a general saving.
- On the other 10 seeds, total steps change by -54 (-0.9%) with postpone and by -650
  (-11.1%) with postpone_drop_enclosed, against 5,841 in the reference. skip_enclosed
  saves 373 steps (-6.4%) there and 725 (-8.6%) over all 12 seeds.

Policy bookkeeping, summed over the 12 seeds. Main queue: popped = dropped (done) +
dropped (invalid) + skipped + postponed + evaluated. Postponed queue: popped
(= postponed) = dropped (done) + dropped (invalid) + dropped (enclosed) + evaluated.

| policy | main popped | main dropped, done | main dropped, invalid | skipped enclosed | postponed | main evaluated | postponed dropped, done | postponed dropped, invalid | postponed dropped, enclosed | postponed evaluated (of which enclosed) |
|---|---|---|---|---|---|---|---|---|---|---|
| skip_enclosed | 37,651 | 28,519 | 854 | 596 | 0 | 7,682 | - | - | - | - |
| postpone | 30,409 | 21,363 | 767 | 0 | 3,357 | 4,922 | 2,095 | 10 | 0 | 1,252 (546) |
| postpone_drop_enclosed | 27,011 | 18,414 | 710 | 0 | 3,035 | 4,852 | 1,834 | 8 | 504 | 689 |

Evaluations (main + postponed) equal the total steps: 7,682, 6,174 and 5,541. About
60% of postponed candidates (62% and 60%) became redundant while waiting, because
another step evaluated their cell. Postponing works as intended, but changing the
order of steps changes the traced object enough to break the 1% recall rule on some
seeds.

## What this does and does not show

- On these 12 seeds, none of the three policies is an acceptable speedup under the
  confirmed tolerance. The gentlest, skip_enclosed, saves 8.6% of steps and fails on
  one seed by 0.21 recall points.
- Tuning a variant until it passes on these same 12 seeds would fit the benchmark.
  Any later attempt would need confirmation on new, held-out seeds.
- The avoided merge on dendrite_1525 suggests that evaluation order affects false
  merges. That was not investigated further.
- Single-object traces on one crop; dense segmentation is not covered.
