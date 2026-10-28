# Why float16 stops traces early: design (milestone 11, fixed before any new run)

Owner decision (2026-10-21): investigate why float16 stops traces early
(docs/manyseed_results.md: 10 new early stops in 300 seeds against 4 extensions).
This is a diagnosis. It claims no speedup; any fix it suggests is judged later on the
300 seeds as its own milestone.

Also in scope: the open finding from the many-seed review. Re-running the fetch
command must not re-pin changed cached files.

## Background: how FFN decides to move

After each network evaluation, FaceMaxMovementPolicy (google/ffn
ffn/inference/movement.py) looks at the six faces 8 voxels from the box center
(17 x 17 voxels each). For each face it takes the highest logit. If that is at least
logit(0.6), it queues a move to that voxel's position, 8 voxels out. Moves from one
step are queued by score, highest first; the queue is first in, first out. So float16
can stop a trace in three ways:
- **threshold**: a face's highest logit falls below logit(0.6) in one run but not the
  other;
- **argmax**: the same face moves in both runs, but to a different position;
- **order**: the same moves are queued in a different order.

## Seeds (fixed now)

- **Early-stop seeds:** the 10 new early stops of docs/manyseed_results.md.
- **Controls:** for each early-stop seed, the seed without any event in the same
  crop and class whose float32 step count is closest (ties by name). If none
  exists, the same class in any crop. No seed is used twice. 10 controls.

## Runs (Wahab V100, deterministic XLA ops, same code path as the many-seed runs)

1. **Closed loop:** each of the 20 seeds traced in float32 and in float16, with a
   recorder on the movement policy. At every step it records the center and, for each
   face, the highest logit, its position and whether a move was queued. Before any
   analysis, each rerun must reproduce the recorded many-seed run bitwise (canvas and
   visit order hashes); otherwise the recorder changed the trace and the run is
   rejected.
2. **Open loop:** along each float32 trace, the network input of every step is
   evaluated again in float16 and in the candidate variants below, on the same input.

## Analyses (pre-registered)

**A. Where each early stop begins.** For each seed, the first step in float32 visit
order whose center was visited in both runs but whose queued moves differ. That step
is classified as threshold, argmax or order (see above). For a threshold step, the
margin is the distance of the float32 face logit from logit(0.6). Reported for the
early-stop seeds and the controls separately. The hypothesis "early stops begin at a
threshold crossing" is supported if at least 7 of the 10 early-stop seeds begin with
a threshold step.

**B. Bias.** On the open-loop inputs, the signed error of float16 against float32 for
the six face maxima, overall and for faces within 0.05 logits of logit(0.6). A
95% confidence interval comes from a bootstrap that resamples seeds, since steps
within a seed are not independent. The hypothesis "float16 is biased toward not
moving" is supported if the near-threshold interval lies entirely below 0. Also
reported: losses (float32 moves, float16 does not) against gains on the same inputs.

**C. Candidate fixes, screened on the open-loop inputs only.**
- `fp16_head32`: float16 network, but the last LayerNorm, ReLU and output
  convolution in float32;
- `fp16_res32`: float16 convolutions, but the residual sums kept in float32;
- `fp16_shift`: float16 with a constant logit offset equal to minus the mean
  near-threshold face error, measured on the 10 controls only and evaluated on all
  inputs. Screened only if B finds a bias.

For each, measured against float32 on the same inputs: the move-decision flip rate
(steps whose queued moves differ), threshold losses and gains, the voxel decision flip
rate, and the time per evaluation at batch 1.

**Selection for a later 300-seed milestone (rule fixed now):** the candidate with the
lowest move-decision flip rate that is at least 1.5x faster per evaluation than
float32. If none qualifies, none is proposed.

## Changes after the independent method check (2026-10-22, before any run)

The method check (Cursor) raised issues that change three definitions above. These
replace them; nothing had been run.

1. **A, primary: the frontier.** "First step whose queued moves differ" fires on
   harmless one-voxel argmax shifts anywhere, controls included, and can hide the step
   where the trace actually stops. The primary measure is now the frontier: on FFN's
   8-voxel lattice (quantize_pos, from the seed), the first float32-visited cell, in
   float32 visit order, that float16 never visits, although float16 did visit the cell
   of the float32 step that queued it. At float16's step in that parent cell, the face
   that queued the lost move is classified:
   - threshold: it does not move in float16;
   - argmax: it moves, but into a different cell;
   - dropped at pop: it moves into the same cell, but the move is never evaluated
     (FFN skips a popped move whose canvas value has fallen below logit(0.6)).
   For threshold, the margin is the float32 face score minus logit(0.6). Hypothesis A
   is supported if at least 7 of the 10 early-stop seeds have a threshold frontier.
   The first difference (original definition) is still reported, as secondary.
2. **B, primary on controls.** The early-stop seeds were selected because float16
   failed on them, which biases their errors negative even without a real bias. The
   bias test therefore uses the 10 controls: supported if their near-threshold
   interval lies entirely below 0. The early-stop seeds are reported separately.
   `fp16_shift` is screened only if B is supported, with offset minus the controls'
   mean near-threshold error.
3. **C, selection on early-stop seeds, against plain float16.** Candidates are
   compared on the 10 early-stop seeds (the shift offset was not fitted on them). The
   selected candidate must have a lower move-decision flip rate than plain float16
   there, and be at least 1.5x faster than float32. Screens over all 20 seeds are
   reported too. `fp16_shift` is jitted and timed like the other candidates.
4. `fp16_res32` keeps the residual stream in float32 up to the last LayerNorm.
5. The voxel flip rate uses the segment threshold (0.6, the same value as the move
   threshold).

## Fetch fix

`liconn_v2.manyseed.fetch_and_select` must, when docs/manyseed_crops.json exists,
verify the coarse level and every cached crop against it before doing anything else,
and never write different pins over existing ones. Tests use temporary files, with no
network.

## Verification

- Unit tests: control selection; the step classifier (threshold, argmax, order) on
  synthetic records; face maxima and argmax against FFN's get_scored_move_offsets;
  the bootstrap; the candidate variants (`fp16_head32` and `fp16_res32` equal the
  float32 network when every part is float32); the fetch fix.
- Controller example: the analyses recomputed from the recorded run outputs.
- GPU runs: owner-approved implementer runs, recorded with source hashes, job IDs
  and output sha256, checked by an evidence test.

## Limits

20 seeds (10 early stops, 10 controls) from one volume, V100 only. The analyses
describe where and why float16 diverges on these seeds; the candidate screen is open
loop. Neither establishes that a fix passes the 300-seed rule.
