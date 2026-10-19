# Benchmark baseline: authors' FFN, 12 seeds (run 2026-10-18)

This is an implementer-run record. It is not a controller-runner result. The controller
check runs only the notebook seed (`--only-notebook-seed`). The full run was approved
by the owner as a separate CPU job.

- Command: `.venv/bin/python -I -u examples/run_benchmark.py --out ~/ResearchFlow/data/liconn-v2/runs/baseline-20261018`
- Hardware: Apple M5 Max CPU (JAX CPU backend), float32. Wall time 3,590 s.
- Source: liconn_v2/, examples/, tests/ and docs/ sha256 values were recorded before the
  run (`source_sha256.txt`) and matched unchanged afterwards. Inputs are pinned in
  `liconn_v2/data.py`.
- Outputs:
  - `benchmark.json`, sha256 629e35d39a6b18f83623eec6c681f5906e06bf27f1b60cdf57c098c138be5b44
  - `benchmark_masks.npz`, sha256 cf8f510b8cdbef621586c4e6b678c1807d951c4f35bc52efd7133f06d73a3591
- Seed rule: owner-approved on 2026-10-18 (12 seeds: 6 axons + all 5 qualifying dendrites
  + the notebook seed). Candidates: 43 axons, 5 dendrites.

| seed | proofread voxels | steps | recall | precision | unlabeled | merged voxels | merges (>=1k) | evals / 1k correct |
|---|---|---|---|---|---|---|---|---|
| axon_3574 | 85,876 | 312 | 0.908 | 0.998 | 0.127 | 145 | 0 | 4.00 |
| axon_13395 | 155,415 | 532 | 0.989 | 0.998 | 0.098 | 250 | 0 | 3.46 |
| axon_10801 | 136,337 | 507 | 0.994 | 0.999 | 0.126 | 106 | 0 | 3.74 |
| axon_5506 | 88,699 | 318 | 0.824 | 1.000 | 0.122 | 12 | 0 | 4.35 |
| axon_218 | 84,020 | 8 | 0.157 | 1.000 | 0.187 | 0 | 0 | 0.61 |
| axon_57259 | 52,593 | 159 | 0.841 | 0.999 | 0.103 | 24 | 0 | 3.60 |
| dendrite_1525 | 188,120 | 2,568 | 0.771 | 0.178 | 0.087 | 672,317 | 1 (97123: 670,790) | 17.70 |
| dendrite_99141 | 330,195 | 957 | 0.881 | 0.992 | 0.108 | 2,440 | 1 (57076: 2,389) | 3.29 |
| dendrite_56789 | 346,452 | 934 | 0.991 | 0.993 | 0.072 | 2,493 | 1 (100179: 2,191) | 2.72 |
| dendrite_61046 | 311,903 | 931 | 0.929 | 0.861 | 0.111 | 46,633 | 1 (8560: 46,123) | 3.21 |
| dendrite_93375 | 294,735 | 830 | 0.911 | 0.886 | 0.078 | 34,509 | 1 (14447: 34,419) | 3.09 |
| notebook | 85,197 | 359 | 0.976 | 0.999 | 0.122 | 63 | 0 | 4.32 |

Summary (means over seeds; cost as ratio of totals):

| group | mean recall | mean precision | mean unlabeled | merges | merged voxels | steps | GFLOP | evals / 1k correct |
|---|---|---|---|---|---|---|---|---|
| all (12) | 0.848 | 0.909 | 0.112 | 5 | 758,992 | 8,415 | 686,669 | 4.39 |
| axons (6) | 0.786 | 0.999 | 0.127 | 0 | 537 | 1,836 | 149,819 | 3.69 |
| dendrites (5) | 0.897 | 0.782 | 0.091 | 5 | 758,392 | 6,220 | 507,556 | 4.65 |

Observations (not conclusions about the paper's pipeline):
- These are single traces with no consensus, agglomeration or proofreading, so the
  merges and early stops are expected to differ from the paper's final segmentation.
- dendrite_1525 merges into proofread object 97123, an unclassified object of 718,972
  voxels in the crop. That one merge dominates total merged voxels and mean dendrite
  precision; summary statistics over 12 seeds are sensitive to it.
- axon_218 stops after 8 steps. Its seed lies on the 16-voxel margin (x = 16).
- The notebook seed reproduced milestone 1 (359 steps; mask 94,726 voxels).
- dendrite_1525 alone accounts for 31% of steps (2,568 / 8,415) and 89% of merged voxels.
  axon_218's seed is pinned at the margin and its trace can change abruptly. Judge
  variants per seed and with and without these two seeds (docs/benchmark_design.md).
- Summary rows in this file come from the run's summarize() output; GFLOP per correct
  voxel was added to summarize() after this run. It equals total GFLOP / total correct
  voxels: 0.358 overall.
