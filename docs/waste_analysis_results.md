# Step-waste analysis: results (run 2026-10-19)

Review history: controller check 20261019-135404-cdba2f09 (task 20261019-125317-f5fa29a5)
flagged two wrong interpretations in an earlier version of this file. They are corrected
below, and the owner approved a re-registration to re-check them.

This is an implementer-run record. It is not a controller-runner result. The controller
check runs only the notebook seed. The full run was approved by the owner as a CPU job.

- Command: `.venv/bin/python -I -u examples/run_waste_analysis.py --out ~/ResearchFlow/data/liconn-v2/runs/waste-20261019`
- Seeds: the 12 benchmark seeds without dendrite_1525 (owner decision): 6 axons,
  4 dendrites and the notebook seed. Wall time 2,435 s on an Apple M5 Max CPU.
- Source: liconn_v2/, examples/, tests/ and docs/ sha256 values were recorded before the
  run (`source_sha256.txt`) and matched unchanged afterwards.
- Outputs:
  - `waste_analysis.json`, sha256 5937c296f125f5e56d5e7971a872c8eeb4893d93353cc54676470975cd72eb57
  - `waste_steps.npz`, sha256 e0312eec65a719cbf2f849d909bd2eccfbe1bd4c936d8a81cfabfafddfc2945f
- Instrumentation checks passed for every seed. Step count = FFN iterations; positions
  = visit history; new voxels sum to predicted voxels; established owners sum to
  predicted voxels. The notebook seed matched the milestone-1 fingerprint bitwise.
  Step counts equal the benchmark baseline's for every seed.

## Per seed (shares of that seed's steps)

Primary classes as originally specified; additional classes from the Cursor method check.

| seed | steps | idle | low-yield | no final decision | mask-idle | mask-low-yield | enclosed |
|---|---|---|---|---|---|---|---|
| axon_3574 | 312 | 0 | 0.099 | 0.090 | 0 | 0.349 | 0.022 |
| axon_13395 | 532 | 0 | 0.160 | 0.248 | 0 | 0.338 | 0.058 |
| axon_10801 | 507 | 0 | 0.128 | 0.130 | 0 | 0.323 | 0.037 |
| axon_5506 | 318 | 0 | 0.104 | 0.104 | 0 | 0.374 | 0.013 |
| axon_218 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| axon_57259 | 159 | 0 | 0.126 | 0.075 | 0 | 0.321 | 0.006 |
| dendrite_99141 | 957 | 0 | 0.165 | 0.200 | 0 | 0.347 | 0.047 |
| dendrite_56789 | 934 | 0 | 0.157 | 0.275 | 0 | 0.330 | 0.069 |
| dendrite_61046 | 931 | 0 | 0.178 | 0.267 | 0 | 0.336 | 0.077 |
| dendrite_93375 | 830 | 0 | 0.186 | 0.231 | 0 | 0.341 | 0.092 |
| notebook | 359 | 0 | 0.081 | 0.058 | 0 | 0.334 | 0.006 |

## Pooled (shares of all steps)

| group | steps | idle | low-yield | no final decision | mask-low-yield | enclosed |
|---|---|---|---|---|---|---|
| all 11 | 5,847 | 0 | 0.152 | 0.202 | 0.338 | 0.055 |
| axons (6) | 1,836 | 0 | 0.127 | 0.148 | 0.339 | 0.034 |
| dendrites (4) | 3,652 | 0 | 0.171 | 0.243 | 0.338 | 0.070 |
| all without axon_218 | 5,839 | 0 | 0.152 | 0.202 | 0.339 | 0.055 |

no_established_decision is 0 for every seed: every step was the last one to establish
the decision of at least one voxel.

## Recomputed summaries (flips, timing)

`examples/run_waste_analysis.py --resummarize waste_steps.npz` applies the current
`waste.seed_summary` to the recorded per-step arrays, without re-tracing. The output is
`waste_analysis_resummarized.json`, sha256
c5ead49b6a6c78bdaf46d2f530e18ec0271ced4bc460c0c1578ea7b74e1e173d. Its step classes equal
the original run's for every seed. Low-yield is the specified definition; MLY is
mask-low-yield.

| seed | steps with flips (after step 0) | median flips | median gained / lost | low-yield Q1 / middle / Q4 | MLY Q1 / middle / Q4 |
|---|---|---|---|---|---|
| axon_3574 | 311 / 311 | 121.5 | 28.5 / 84 | 0.103 / 0.122 / 0.051 | 0.321 / 0.365 / 0.346 |
| axon_13395 | 531 / 531 | 111 | 32 / 80 | 0.098 / 0.192 / 0.158 | 0.241 / 0.376 / 0.361 |
| axon_10801 | 506 / 506 | 120 | 30 / 83 | 0.111 / 0.118 / 0.167 | 0.302 / 0.329 / 0.333 |
| axon_5506 | 317 / 317 | 107.5 | 31 / 72 | 0.063 / 0.119 / 0.114 | 0.278 / 0.406 / 0.405 |
| axon_218 | 7 / 7 | 518.5 | 125 / 398 | 0 / 0 / 0 | 0 / 0 / 0 |
| axon_57259 | 158 / 158 | 82 | 22 / 55 | 0.103 / 0.111 / 0.179 | 0.256 / 0.321 / 0.385 |
| dendrite_99141 | 956 / 956 | 106 | 27 / 74 | 0.209 / 0.154 / 0.142 | 0.368 / 0.344 / 0.331 |
| dendrite_56789 | 933 / 933 | 143 | 39 / 95 | 0.159 / 0.158 / 0.155 | 0.356 / 0.342 / 0.279 |
| dendrite_61046 | 930 / 930 | 140 | 36 / 97 | 0.177 / 0.193 / 0.151 | 0.315 / 0.330 / 0.371 |
| dendrite_93375 | 829 / 829 | 132 | 33 / 90 | 0.174 / 0.221 / 0.126 | 0.266 / 0.365 / 0.367 |
| notebook | 358 / 358 | 77 | 20 / 53 | 0.045 / 0.088 / 0.101 | 0.393 / 0.326 / 0.292 |

Q1 and Q4 are the first and last quarter of a trace's steps, and middle is the half
between them. Axons without axon_218 pooled: 1,828 steps, low-yield 0.128,
mask-low-yield 0.341, enclosed 0.034, no final decision 0.148.

## What the measurements show

- No step is idle, so a rule that skips only steps that change nothing would save
  nothing. Every step after a trace's first one (5,836 of 5,836) flips the decision of at
  least one previously predicted voxel. A first step has no previously predicted voxels,
  so it cannot flip any. Median flips per step are 77-143 by seed (518.5 for the 8-step
  axon_218). More flips go from object to background than the reverse (per-seed median lost
  53-97 versus gained 20-39, axon_218 excluded). The cause has not been analysed.
- Low-yield steps are 15% of steps under the specified definition and 34% under the
  mask-only definition. Except for axon_218 (8 steps, none low-yield), each seed's
  mask-low-yield share lies between 0.32 and 0.37.
- Low-yield steps occur in every part of the traces. Each seed's low-yield share is
  0.045-0.209 in the first quarter, 0.088-0.221 in the middle half and 0.051-0.179 in
  the last quarter (axon_218 excluded). There is no consistent concentration at the end.
  Several axons are lower in the first quarter than later, and several dendrites are
  higher early. Quarter averages cannot show shorter bursts within a quarter.
- Only `enclosed` steps (5.5%; 7.0% for dendrites) are a step class identifiable before
  spending the evaluation. The number of not-yet-predicted voxels in a box (new_voxels)
  is also known in advance, but on its own it is a quantity, not a class. The
  final-decision classes are known only at the end of a trace.
- One step in five (20%) leaves no final decision under the specified last-writer
  definition. Every step established at least one voxel decision that survived to the
  end (no_established_decision is 0 for all 5,847 steps). So for each of those 20%, at
  least one surviving decision was later rewritten by another step without a flip. The
  metric does not show what happened to the step's other decisions.

## What this does not show

- These shares are not savings. Skipping a step changes later steps, because continuous
  logits feed later evaluations and a step's candidates can keep cells reachable. Any
  savings must be measured by running a modified policy on the benchmark.
- The per-step flip counts do not separate genuine boundary refinement from flip-flopping
  that later reverses; that would need a separate measurement.
- Same single-object, one-crop, CPU limits as the benchmark.
