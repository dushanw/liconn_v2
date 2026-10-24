# Fast tracer: results (runs 2026-10-24)

**Outcome: the fast tracer is bitwise identical on all 624 traces, in float32 and in
`fp16_head32`, and faster.** In the same job and on the same GPU it is 1.12x faster in
float32 and 1.43x faster in `fp16_head32`. `fp16_head32` with the fast tracer runs at
5.96 ms per step against 19.27 ms for the original float32 path in the same jobs:
about **3.2x** per step. Because every trace is identical, accuracy and the owner's 2%
verdict for `fp16_head32` (docs/tolerance2_results.md) carry over unchanged.

As the contract requires, the face maxima are computed on the GPU. Only six scores and
six positions come back per step; FFN's own queue and lattice code run on the host,
fed by `queue_moves`, which repeats FaceMaxMovementPolicy.update (docs/fasttrace_design.md).

This is an implementer-run record of owner-approved runs on Wahab V100s:
- float32 on the 300 fresh seeds: Slurm 7432091;
- `fp16_head32` on the 300 fresh seeds: 7432092;
- float32 on the 12 benchmark seeds, with the reference path in the same job: 7432093;
- `fp16_head32` on the 12 benchmark seeds, with the reference path in the same job:
  7432094.

The code that ran is the checked code; there are no post-run edits.
tests/test_fasttrace_evidence.py checks the following against
docs/fasttrace_evidence.json:
- the code that ran is the checked code;
- the outputs are the recorded ones;
- each trace's step count, canvas sha256 and history sha256 equal the recorded run's,
  compared by the test itself, not taken from the runner's flags;
- the summaries below follow from the outputs.

## Bitwise identity (the pass criterion)

Canvas sha256, visit history sha256 and step count:

| traces | compared with | identical |
|---|---|---|
| float32, 300 fresh seeds | recorded run fresh-fp32-20261023 | 300 / 300 |
| `fp16_head32`, 300 fresh seeds | recorded run fresh-fp16_head32-20261023 (the runs judged under the 2% rule) | 300 / 300 |
| float32, 12 benchmark seeds | recorded run gpu-baseline-a-20261019, and the reference path in the same job | 12 / 12 |
| `fp16_head32`, 12 benchmark seeds | the reference path in the same job (no recorded run exists) | 12 / 12 |

The last row rests on the runner's in-job comparison. That is the one comparison the
evidence test cannot repeat independently, because no recorded run exists for it.

On CPU, the authors' notebook seed is identical to the pinned reference fingerprint
in every field (examples/check_fasttrace.py, tests/test_fasttrace.py).

## Speed (V100)

| | reference path | fast tracer | gain |
|---|---|---|---|
| float32, 12 benchmark seeds, same job | 19.27 ms/step | 17.16 ms/step | 1.12x |
| `fp16_head32`, 12 benchmark seeds, same job | 8.55 ms/step | 5.96 ms/step | 1.43x |
| `fp16_head32` fast against the original float32 (the float32 job's reference path) | 19.27 ms/step | 5.96 ms/step | 3.23x |
| 300 fresh seeds (separate jobs; approximate) | float32 19.51 ms/step (recorded run fresh-fp32-20261023) | `fp16_head32` fast 6.41 ms/step | 3.04x |

**Timing windows.** The two paths do not time exactly the same work:
- The fast tracer's clock starts after the image has been copied to the GPU and the
  canvas created. Both are waited for (`block_until_ready`) before the clock starts, so
  this setup is excluded. The image is copied for every trace.
- The fast tracer's time includes copying the final canvas back. On the first seed of
  a job it also includes compiling its small GPU functions.
- The reference path's clock (liconn_v2/reference.py) also starts after its setup:
  FFN's Canvas, which holds the image and allocates the canvas arrays, is built before
  the clock starts. Its time covers only `segment_at`, not copying the result out.

So both paths exclude per-trace setup. Only the fast tracer's time includes the final
copy and the first seed's compilation. These were not measured separately; if anything
they count against the fast tracer. The per-step figures therefore compare tracing
work, not the whole cost of a trace including setup.

## How the GPU face maxima were made identical (and fast)

Three versions were run. The two superseded ones are kept in the data folder
(`runs/superseded-*-2026102[34]`) and are not part of the evidence.

1. **Face maxima on the host, from FFN's own code (runs 2026-10-23, Slurm
   7431344-7431347).** All 624 traces were identical; `fp16_head32` ran at 6.36
   ms/step. The task requires GPU face maxima, so this was not accepted.
2. **GPU face maxima with plain `jnp.argmax` (Slurm 7431514-7431517).** Not identical.
   FFN finds the maximum with NumPy, where the first NaN counts as the maximum. On the
   V100, `jnp.argmax` over several NaNs does not reliably return the first one; a probe
   found 634 of 1,000 such faces placed differently. NaN faces occur when a prediction
   is NaN in float16. `fp16_head32` mismatched on 3 of 12 benchmark seeds and on 4 of
   the first 168 fresh seeds before the runs were stopped. float32 matched on all 12
   benchmark seeds and the first 88 fresh seeds.

   The fix locates NaNs explicitly: if a face has a NaN, the argmax of the NaN mask
   gives the first one, and argmax of a boolean array breaks ties by the lowest index.
   This change keeps FFN's rule that a NaN score passes the `score < threshold` test,
   so a NaN face still becomes a move.
3. **Explicit NaN handling, six separate reductions (Slurm 7431535-7431538).**
   Identical on all 12 + 12 benchmark seeds and the first 139 + 260 fresh seeds
   (stopped then), but slower than the host version: 6.83 ms/step for `fp16_head32`.
   Each of the six faces ran its own small reductions.
4. **Final: one stacked (6, 289) reduction (this record).** Identical, 5.96 ms/step.
   Before the final runs, a V100 probe compared the stacked function with NumPy on
   2,000 random six-face sets with NaNs and ties: 0 mismatches (Slurm 7432098,
   exploratory; `runs/probe-argmax-20261024` in the data folder).

## What this shows

- The bookkeeping around the network, including the face maxima, can run on the GPU
  without changing a single number of FFN's output, for both number formats.
- Exactness on a GPU needed explicit care for NaN ordering. Without it, `fp16_head32`
  traces changed on 3 of 12 benchmark seeds and 4 of 168 fresh seeds.
- With `fp16_head32`, about 4 ms of the 5.96 ms step is the network (float16 profile in
  docs/beyond_fp16_ideas.md; the head32 layout was not profiled separately). The
  remaining ~2 ms is host work and small transfers. Further gains need a faster
  network, or overlapping host work across traces.
- Bitwise identity is shown empirically (624 traces). The runs used deterministic XLA
  ops; GPU kernel determinism across jobs and GPU models is not guaranteed in general.

## Limits

Single-object traces with the authors' settings, not FFN's multi-object segmentation
mode. V100 only; other GPUs will differ. The fresh-seed speed comparison is across
separate jobs. The float32 gain (1.12x) is below the 1.3x ceiling estimated for
removing all overhead (docs/beyond_fp16_ideas.md).
