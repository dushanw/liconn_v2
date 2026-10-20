# Postponing nearly enclosed steps: design (milestone 6)

Owner decision (2026-10-19): try option A from docs/predictability_results.md, a
movement policy that postpones low-value steps, and judge it on the benchmark.
This is the first change to the FFN method itself. It changes only the order and
selection of steps; the network, weights, thresholds and box size are unchanged.

## Background

Nearly enclosed steps are 27% of steps: fewer than 359 of the 33^3 box voxels are not
yet predicted. They contain every low-yield step, but 44% of them are still useful
(docs/predictability_results.md). Skipping them outright risks cutting off branches.
Postponing them lets other steps run first, which may make them redundant.

## Policies

Policies are installed through the existing read-only instrumentation hook of
`Tracer.trace`: after the Canvas is created and before `segment_at`, the hook
replaces `canvas.movement_policy`. reference.py, benchmark.py and run_benchmark.py
stay byte-identical, because the GPU-baseline and predictability evidence pin them.
Policy runs use examples/run_policy_benchmark.py (with `--policy` and `--seed`),
which writes the same benchmark.json and benchmark_masks.npz format.

All variants subclass FFN's `FaceMaxMovementPolicy` (ffn/inference/movement.py at
12d680e). They keep its proposals and its pop-time checks: a candidate is dropped if
its 8-voxel cell was already evaluated, or if `Canvas.is_valid_pos` fails (canvas
value below the move threshold, or too close to the edge). Each candidate that passes
those checks is classified by U, the number of not-yet-predicted voxels in its box,
counted from the canvas at that moment:

| policy | rule |
|---|---|
| `default` | unchanged FFN (the reference) |
| `skip_enclosed` | if U = 0, do not evaluate; mark its cell as done |
| `postpone` | if U < 359 and not postponed before, move it to a second queue that is used only when the main queue is empty; evaluate it then (re-checked when it comes up) |
| `postpone_drop_enclosed` | as `postpone`, but when a postponed candidate comes up and U = 0, do not evaluate it; mark its cell as done |

Marking the cell as done follows FFN's own convention: one evaluation per 8-voxel
cell. Without it, another queued candidate in the same cell would be evaluated
instead, and the saving would only move elsewhere. Each policy counts how many
candidates it skipped, postponed, and later evaluated or dropped; the counts are
recorded per seed.

## Evaluation

- All 12 benchmark seeds per policy, on the reference backend (Wahab V100,
  `LICONN_BACKEND=gpu`, deterministic XLA ops), one job array per policy.
- Each policy run is compared with the adopted reference (gpu-baseline-a-20261019)
  by examples/compare_runs.py:
  - `passes_tolerance`: every rule at 1%, all seeds present, same inputs;
  - per-seed mask IoU and deltas;
  - cost in evaluations per 1,000 correct voxels and total steps.
- A policy is a candidate speedup only if it passes the tolerance and lowers the
  evaluations per 1,000 correct voxels. The change in total steps is the method's
  saving; wall time is secondary.

## Verification

- Unit tests on a synthetic canvas: each policy's order, skip and drop behaviour; the
  done-cell marking; the statistics.
- The default policy is untouched. The existing bitwise reference tests must still
  pass. `--policy default` through the new script must match the reference per seed
  (canvas and visit-order sha256) on the GPU, as a control run.
- Policy statistics per seed, per queue: popped, dropped because the cell was done,
  dropped as invalid, skipped or dropped as enclosed, postponed, evaluated. They must
  add up: each queue's pops are accounted for, every postponed candidate is popped
  again, and evaluations equal the trace steps. A run with inconsistent counts fails.
  Known edge case: if FFN ends a trace because the seed voxel itself falls below the
  move threshold, the last popped candidate is counted as evaluated without running,
  and the check fails loudly rather than misreporting. It did not occur in the runs.
- Results report passes_tolerance together with per-seed mask IoU, flagged seeds, the
  absolute and relative change in total steps, and evaluations per 1,000 correct
  voxels, for all seeds and without dendrite_1525 and axon_218. dendrite_1525 was not
  part of the predictability study.
- The controller example runs the notebook seed on CPU with `postpone_drop_enclosed`.
- The GPU runs are owner-approved implementer runs on Wahab, recorded with source
  hashes, job IDs and output hashes, and checked by an evidence test like the earlier
  milestones.

## Limits

Single-object traces on one crop. A policy that passes on 12 seeds is not proven in
general. Dense segmentation (`segment_all`) and other crops are not covered.
