# GPU baseline on Wahab V100s (run 2026-10-19)

This is an implementer-run record, approved by the owner and run on ODU's Wahab
cluster. The controller checks run locally on CPU only. They do not re-run the GPU,
but tests/test_gpu_baseline_evidence.py checks this record against
docs/gpu_baseline_evidence.json:
- the code files used by the GPU jobs are byte-identical to the checked source;
- the run outputs in the data folder have the recorded sha256;
- both runs record V100, deterministic XLA ops, their Slurm job and cuDNN 9.6.0.74;
- run a and run b match seed by seed;
- the CPU-vs-GPU comparison below is recomputed from the stored outputs.

## What ran

- Method: unchanged, as in docs/benchmark_design.md: authors' checkpoint and notebook
  settings, 12 benchmark seeds, one trace per seed.
- Backend: `LICONN_BACKEND=gpu`, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, Tesla
  V100-SXM2-16GB, `timed-gpu` partition, one GPU per seed (Slurm job array).
- Environment: Python 3.12.8 (uv); jax/jaxlib 0.4.38; cuDNN 9.6.0.74 and CUDA 12.6
  libraries (requirements-cuda.txt); other packages as in requirements.txt. Each
  record's `platform.versions` lists all of them.
- Source: base commit ac5ad6a plus this milestone's uncommitted changes. All 24 job
  logs print `ac5ad6a-dirty`. The working tree
  was synced to Wahab and its sha256 manifest checked identical, ignoring sort order;
  a copy is in each run's `source_sha256.txt`. The files the jobs used are listed with
  their sha256 in docs/gpu_baseline_evidence.json. After the runs, only
  liconn_v2/compare.py (comparison edge cases, merge input checks, per-seed tolerance
  fields), examples/compare_runs.py, docs, tests and README changed. The merged files
  were produced by the earlier merge code; their hashes are checked.
- Two independent runs, each a separate 12-task job array, on 8 different V100 nodes
  in total:
  - run a: Slurm job 7358888;
  - run b: Slurm job 7358889.
- An earlier submission (jobs 7358860, 7358861) failed in every task before tracing,
  because `run_benchmark.py` did not create the output folder's parent. That was fixed
  (`mkdir(parents=True)`), and the logs are kept on Wahab under
  `/scratch/dwadduwa/liconn_runs/failed-20261019`.

Outputs (copied to `data/runs/` in the repository folder; data/ is git-ignored):

| file | sha256 |
|---|---|
| gpu-baseline-a-20261019/merged/benchmark.json | d640bf803f938ed0710dd5446be4d103f333b5839783ef0bd65eb82045aff704 |
| gpu-baseline-a-20261019/merged/benchmark_masks.npz | 2474393f02bfea381d1563138d5e97d7541c824323e3c4a0da060ff1596ac349 |
| gpu-baseline-b-20261019/merged/benchmark.json | 6608b804cd035c0c1b5ced6268294d62dd5c88e506a21fb09d3eff43513cd41c |
| gpu-baseline-b-20261019/merged/benchmark_masks.npz | 2474393f02bfea381d1563138d5e97d7541c824323e3c4a0da060ff1596ac349 |
| gpu-baseline-a-20261019/compare_cpu_vs_gpu.json | 190436f729766607f6b1ab7d65c647bae49d8514bd9144218aedf5ee2c0efc7c |
| gpu-baseline-b-20261019/compare_a_vs_b.json | 8420d40645f23279c5fb6435062a498d639c3a3204276eabac9afc519c0d7ed0 |

- The per-seed part outputs written by the 24 GPU jobs (`parts/<seed>/`) are hashed
  in docs/gpu_baseline_evidence.json.
- The merged files were rebuilt from those parts on 2026-10-19 with the current merge
  code, which records the input hashes. The first merged versions, kept in
  `merged-superseded/`, lacked them. Records, summaries and masks are identical, and
  the masks file is byte-identical.
- The two benchmark.json files differ only in run metadata (node names, job IDs,
  times).
- The comparison files were written by the current comparison tool, which also
  checks that both runs used the same inputs and seeds.

## Reproducibility on GPU (run a vs run b)

For all 12 seeds, the two runs produced the same canvas sha256, the same visit-order
sha256, the same step count and the same metrics. The mask files are byte-identical.
With deterministic XLA ops, GPU results were reproducible across separate jobs and
nodes in this test.

## GPU (run a) vs CPU baseline (docs/benchmark_baseline.md)

| seed | mask IoU | steps CPU / GPU | recall change | precision change | merged voxels change | merges CPU / GPU |
|---|---|---|---|---|---|---|
| axon_3574 | 1.000000 | 312 / 312 | 0 | 0 | 0 | 0 / 0 |
| axon_13395 | 1.000000 | 532 / 532 | 0 | 0 | 0 | 0 / 0 |
| axon_10801 | 1.000000 | 507 / 507 | 0 | 0 | 0 | 0 / 0 |
| axon_5506 | 1.000000 | 318 / 318 | 0 | 0 | 0 | 0 / 0 |
| axon_218 | 1.000000 | 8 / 8 | 0 | 0 | 0 | 0 / 0 |
| axon_57259 | 1.000000 | 159 / 159 | 0 | 0 | 0 | 0 / 0 |
| dendrite_1525 | **0.986744** | 2,568 / 2,558 | 0.0000 | +0.0008 | -3,673 | 1 / 1 |
| dendrite_99141 | 0.996254 | 957 / 959 | -0.0000 | -0.0000 | +4 | 1 / 1 |
| dendrite_56789 | 1.000000 | 934 / 934 | 0 | 0 | 0 | 1 / 1 |
| dendrite_61046 | 1.000000 | 931 / 931 | 0 | 0 | 0 | 1 / 1 |
| dendrite_93375 | 1.000000 | 830 / 830 | 0 | 0 | 0 | 1 / 1 |
| notebook | 1.000000 | 359 / 359 | 0 | 0 | 0 | 0 / 0 |

IoU 1.000000 means exactly 1: the masks are voxel-identical.

- Ten seeds have the same step count and voxel-identical masks. Their visit order
  cannot be compared bitwise, because the CPU baseline predates the history_sha256
  field.
- Two dendrites diverge slightly. dendrite_99141: 2 more steps, IoU 0.996.
  dendrite_1525: 10 fewer steps, IoU 0.987. Where and why their traces first diverge
  has not been measured. A plausible but unverified explanation is a value lying
  within rounding distance of the 0.6 move or segment threshold. The setup probe on 50
  independent boxes found no threshold crossings, but it did not cover traces. dendrite_1525 is flagged for owner
  review (IoU < 0.99). It is the seed with the large merge into object 97123, and the
  merge is present on both backends. Where its 1.3% mask difference lies, against the
  proofread labels:

  | voxels in only one mask | total | in dendrite 1525 | in merged object 97123 | unlabeled | other objects |
  |---|---|---|---|---|---|
  | CPU only | 8,308 | 1 | 4,245 | 3,557 | 505 |
  | GPU only | 3,603 | 1 | 1,006 | 2,525 | 71 |

  The differences are almost entirely outside the dendrite: the GPU trace reaches
  slightly less far into the falsely merged neighbour. That is why recall is
  unchanged and precision is slightly higher. For dendrite_99141, 646 CPU-only and
  589 GPU-only voxels differ, mostly unlabeled (593 and 548).
- Accuracy tolerance (owner-confirmed 2026-10-19, every rule at 1%): GPU vs CPU is
  within tolerance for all 12 seeds and without the two sensitive seeds. Mean recall
  change 0; mean precision +0.00007; no seed's recall dropped; no new merges; total
  merged voxels -0.48%.
- Cost: 8,407 vs 8,415 steps in total; 4.382 vs 4.386 evaluations per 1,000 correct
  voxels.

## The reference run (GPU run a), absolute values

Adopted by the owner on 2026-10-19 as the reference for speed variants.

| seed | proofread voxels | steps | recall | precision | unlabeled | merged voxels | merges (>= 1k voxels) | evals / 1k correct |
|---|---|---|---|---|---|---|---|---|
| axon_3574 | 85,876 | 312 | 0.9084 | 0.9981 | 0.1272 | 145 | none | 4.000 |
| axon_13395 | 155,415 | 532 | 0.9894 | 0.9984 | 0.0976 | 250 | none | 3.460 |
| axon_10801 | 136,337 | 507 | 0.9942 | 0.9992 | 0.1258 | 106 | none | 3.740 |
| axon_5506 | 88,699 | 318 | 0.8243 | 0.9998 | 0.1225 | 12 | none | 4.349 |
| axon_218 | 84,020 | 8 | 0.1565 | 1.0000 | 0.1867 | 0 | none | 0.608 |
| axon_57259 | 52,593 | 159 | 0.8409 | 0.9995 | 0.1030 | 24 | none | 3.595 |
| dendrite_1525 | 188,120 | 2,558 | 0.7713 | 0.1783 | 0.0859 | 668,644 | 97123: 667,551 | 17.630 |
| dendrite_99141 | 330,195 | 959 | 0.8812 | 0.9917 | 0.1082 | 2,444 | 57076: 2,389 | 3.296 |
| dendrite_56789 | 346,452 | 934 | 0.9909 | 0.9928 | 0.0724 | 2,493 | 100179: 2,191 | 2.721 |
| dendrite_61046 | 311,903 | 931 | 0.9289 | 0.8614 | 0.1109 | 46,633 | 8560: 46,123 | 3.214 |
| dendrite_93375 | 294,735 | 830 | 0.9106 | 0.8861 | 0.0778 | 34,509 | 14447: 34,419 | 3.093 |
| notebook | 85,197 | 359 | 0.9759 | 0.9992 | 0.1216 | 63 | none | 4.318 |

| group | mean recall | mean precision | mean unlabeled | merges | merged voxels | steps | evals / 1k correct |
|---|---|---|---|---|---|---|---|
| all 12 | 0.8477 | 0.9087 | 0.1116 | 5 | 755,323 | 8,407 | 4.382 |
| axons (6) | 0.7856 | 0.9992 | 0.1271 | 0 | 537 | 1,836 | 3.688 |
| dendrites (5) | 0.8966 | 0.7820 | 0.0911 | 5 | 754,723 | 6,212 | 4.645 |
| without dendrite_1525 and axon_218 (10) | 0.9245 | 0.9726 | | | 86,679 | 5,841 | 3.318 |

Known limits of the tolerance, as confirmed:
- It bounds the total of merged voxels, not each seed's. dendrite_1525 holds 89% of
  the total, so a 1% budget on the total (about 7,550 voxels) is large compared with
  the other seeds' merges. Per-seed merged-voxel changes are reported by
  examples/compare_runs.py but are not a pass/fail rule.
- Mask IoU against the reference is reported and flagged below 0.99, but is not a
  pass/fail rule.

## Speed (same method, different hardware)

| | total trace time, 12 seeds | per step |
|---|---|---|
| Apple M5 Max CPU (baseline-20261018) | 3,583 s | 425.8 ms |
| one V100 per seed (run a, summed over seeds) | 165 s | 19.7 ms |

This is a hardware speed-up of about 22x per step, not a method speed-up; the method's
cost per correct voxel is unchanged. Each array task took about 3-4 minutes of wall
time including start-up and seed selection, so the whole array finished in minutes.

## What this does and does not establish

- On these 12 seeds, the GPU runs the same method with results identical to the CPU in
  10 seeds and slightly different in two long dendrite traces, for an unmeasured
  reason.
- Adopted as the reference: on 2026-10-19 the owner adopted run a as the reference
  for later speed variants (`liconn_v2.compare.REFERENCE_RUN`). This was done knowing
  that dendrite_1525's IoU with the CPU is 0.987, below the 0.99 review threshold.
- A100 nodes, other GPU types and larger volumes are not covered.
