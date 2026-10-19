# FFN step-waste analysis: design (milestone 3)

Question: how many network evaluations in the authors' FFN traces change nothing?
Their share bounds what a smarter movement policy could save. It is only a rough bound,
because skipping a step would change later steps. This milestone measures; it does not
change the method.

Owner decisions (2026-10-19): run on the benchmark seeds without dendrite_1525 (its
single trace includes a large merge and dominates totals), i.e. 11 seeds. The
benchmark baseline itself is unchanged.

## Instrumentation

Each trace runs exactly as in the benchmark (authors' checkpoint and settings, fresh
Canvas per seed). Two wrappers are added on the Canvas instance. Both only read state
and record; they must not change the canvas, the queue or the visit order:
- around `update_at(pos)`: copy the 33^3 canvas box before and after the update;
- around `movement_policy.update(...)`: record which candidates were appended to the queue.

Verification that they change nothing: the instrumented notebook-seed trace must
reproduce milestone 1's committed fingerprint bitwise.

## Per-step quantities

FFN behaviour at the pinned commit, checked in the source (ffn/inference/inference.py,
movement.py):
- `segment_at` calls `update_at(pos)` and then `movement_policy.update(pred, pos)`.
- `update_at` always writes the whole 33^3 box. With the notebook options,
  `disco_seed_threshold` = 0.0, so a guard keeps previously-below-0.5 voxels from rising,
  but the box is still written.
- Before step 0, `init_seed` sets the seed voxel to logit(0.95).
- The queue `scored_coords` is a FIFO deque. Candidates are dropped at pop time if their
  8-voxel cell (`quantize_pos`) is in `done_rounded_coords`, or if the canvas value there
  is below the move threshold.
- `reset_state` resets the policy in place.

With t = np.float32(logit(0.6)) (as in benchmark.object_mask), box B, and canvas values
before (old) and after (new) the step. For step 0, the seed voxel's pre-set value counts
as NaN.
- new_fg / new_bg: voxels in B that were NaN before and are now >= t / < t.
  new_voxels = new_fg + new_bg. new_voxels > 0 is known before evaluating (from old alone).
- gained / lost: previously predicted voxels in B whose decision changed to >= t / < t.
  flips = gained + lost.
- new_cells: distinct quantized cells of the candidates this step appended to the queue
  that were not in `done_rounded_coords` (after this step's own cell was added) and not
  among cells still waiting in the queue before this step. Duplicates within the step
  count once.
- mean_abs_dlogit: mean |new - old| over previously predicted voxels in B. Even without
  flips, a step changes values that later steps read as input.
- final_owner_voxels (as originally specified): voxels whose final value was last
  written by this step, and whose decision that write established (first prediction or
  flip). Voxels later overwritten without a flip have no owner under this definition.
- established_owner_voxels (additional; Cursor method check): each voxel's owner is the
  last step that established its decision, so every predicted voxel has exactly one owner.

Step classes, as originally specified (primary results):
- idle: new_voxels == 0, flips == 0 and new_cells == 0;
- low-yield: not idle, new_voxels + flips < 359 (1% of the box) and new_cells == 0;
- no_final_decision: final_owner_voxels == 0.

Additional classes (Cursor method check; reported alongside, not as replacements):
- mask-idle: new_fg == 0, flips == 0 and new_cells == 0. The object mask and the queue's
  new territory are unchanged; new background predictions are allowed.
- mask-low-yield: not mask-idle, new_fg + flips < 359 and new_cells == 0.
- enclosed: new_voxels == 0, i.e. the box was already fully predicted. A policy could
  detect this before spending the evaluation.
- no_established_decision: established_owner_voxels == 0.

Invariants checked by tests: the steps' new_voxels (seed voxel counted at step 0) sum to
the number of predicted voxels; established_owner_voxels sum to the same number
(final_owner_voxels sum to at most that); and the
instrumented notebook trace matches milestone 1 bitwise.

Biases (Cursor method check, 2026-10-19):
- Mask-idle can overstate savings: skipped steps also skip continuous logit updates that
  later steps read, and a step's candidate can be the backup that later keeps a cell
  reachable.
- It can understate savings: steps whose new flips or new cells are later reversed or
  rejected count as useful.
- Only `enclosed` is observable before evaluation; the other classes are known only
  after spending the evaluation.

## Outputs

Per seed: number of steps; idle, mask-idle, low-yield and enclosed counts and shares; distributions
(quartiles) of new_fg, new_bg, flips, new_cells and mean_abs_dlogit; idle share in the first and last
quarter of the trace (to see whether waste concentrates late). Also a share of
steps that set no final decision (final_owner_voxels == 0).
Summary: totals and shares over all 11 seeds and per class, with and without axon_218.
Per-step records are written to outputs/waste_steps.npz for later policy simulations
(not a required controller artifact).

## Limits

- An idle step can still matter indirectly: it confirms values that later
  movement decisions read. The idle share is an estimate of the savings, not a guarantee.
- Same single-object, CPU, one-crop limits as the benchmark (docs/benchmark_design.md).
