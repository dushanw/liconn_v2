# Benchmark on a Wahab GPU: design (milestone 4)

Goal: run the unchanged FFN method (authors' checkpoint and settings) on NVIDIA GPUs
on ODU's Wahab cluster. Record a GPU baseline for the 12 benchmark seeds, and compare
it seed by seed with the CPU baseline (docs/benchmark_baseline.md). Owner request,
2026-10-19. No speedup of the method is implemented.

## Backend selection

- `LICONN_BACKEND` environment variable: `cpu` (default) or `gpu`. liconn_v2/reference.py
  reads it at import and sets `JAX_PLATFORMS` (`cpu` or `cuda`) before JAX is imported.
  `Tracer` raises if the active JAX backend does not match. CPU remains the default, so
  milestone 1's bitwise CPU reference test is unchanged.
- TensorFlow stays hidden from the GPU (as in the notebook); it is only used to read the
  checkpoint.
- GPU reproducibility: jobs set `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. Running the
  notebook seed twice on the GPU must give the same canvas sha256 (checked in the run).

## Running on Wahab

- `requirements-cuda.txt`: jax[cuda12]==0.4.38 plus NVIDIA libraries pinned to releases
  contemporary with it (cuDNN 9.6.0.74, CUDA 12.6). The newest cuDNN, 9.27, failed on
  V100 with cuDNN status 5003 during setup.
- `cluster/wahab/README.md`: login (password + DUO from a terminal; ssh connection
  reuse), environment build (uv-managed Python 3.12.8), data fetch, code sync and job
  submission.
- `cluster/wahab/benchmark_array.sbatch`: a Slurm job array, one benchmark seed per
  task, on the `gpu` or `timed-gpu` partition with one GPU each. Each task calls
  `examples/run_benchmark.py --seed NAME --out RUN/NAME`.
- `examples/merge_benchmark.py`: combines per-seed outputs into one benchmark.json and
  benchmark_masks.npz with the same format and summary as a single run.

## Comparison with the CPU baseline

`examples/compare_runs.py BASE CANDIDATE` writes, per seed:
- differences in steps, recall, precision, unlabeled fraction, merged voxels and merges;
- voxel IoU between the two runs' object masks (from benchmark_masks.npz);
- the accuracy tolerance checks (docs/benchmark_design.md; owner-confirmed at 1% on
  2026-10-19 after this design was written), reported per seed and
  overall, with and without dendrite_1525 and axon_218. A comparison passes only if
  all seeds are present, both runs used the same inputs and seeds, and every rule
  holds (`passes_tolerance`).

Expected: GPU arithmetic differs from the CPU at rounding level. The setup probe found
a max logit difference of 3.5e-5 on 50 boxes and no threshold crossings. Traces may
still diverge where a value sits right at a decision threshold, so step counts and
masks may differ slightly. The milestone-1 bitwise reference stays a CPU-only test.

## Verification

- Unit tests (local, no GPU): backend selection and its errors; `--seed` selection,
  including unknown names; merge of per-seed outputs (summary equals a single-run
  summary); compare_runs metrics and IoU on synthetic runs.
- Existing tests, including the CPU bitwise reference, stay unchanged.
- The controller example runs the notebook seed via `--seed notebook` on CPU.
- The GPU baseline itself is an owner-approved implementer-run job on Wahab, recorded
  with its source (base commit and the sha256 of every file the jobs used, since the
  runs used uncommitted changes), job IDs, GPU type, package versions and output sha256 in
  docs/gpu_baseline.md.

## Limits

GPU results are for V100 (Wahab `gpu` partition); A100 is not covered. Wall times
depend on the shared cluster's load. Same single-object, one-crop limits as the
benchmark.
