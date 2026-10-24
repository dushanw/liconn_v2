# Fast tracer: removing the overhead around the network (milestone 16, design)

Owner decision (2026-10-23): remove the per-step overhead around the network while
keeping every trace bitwise identical, for float32 and for `fp16_head32`.

## Why

On a V100 a float16 step takes about 10 ms, of which about 4 ms is the network
(docs/beyond_fp16_ideas.md profile). The rest is FFN's bookkeeping: input assembly on
the CPU, a hand-off to an executor thread, copies between CPU and GPU, the canvas
update and the face maxima. A direct network call (no executor thread) was already
bitwise identical and 9% faster in float32.

## What the fast tracer does

Per trace, the normalised image and the canvas (float32 logits, NaN where unpredicted)
are kept on the GPU. Each step:
1. Host: pop the next position with FFN's exact queue rules (FaceMaxMovementPolicy and
   Canvas.segment_at): first in, first out; skip a position whose 8-voxel lattice cell
   is done; skip it if the canvas value there is below the move threshold, or if it is
   too close to the volume border; stop if the canvas value at the seed falls below the
   move threshold. The canvas values needed for these checks are read from the GPU.
2. GPU, one jitted function: cut the 33^3 canvas box (NaN replaced by the pad value,
   which FFN converts to logit(0.5) = 0) and the image box, and concatenate them as the
   network input.
3. GPU: the network, called directly with the same jitted network function the
   reference path uses for that format.
4. GPU, one jitted function: FFN's seed-consistency step (disco_seed_threshold 0),
   the write of the box into the canvas, and the six face maxima of FFN's
   get_scored_move_offsets: for each face 8 voxels from the center (17 x 17 voxels), the
   highest logit and its position, by argmax in C order. As in NumPy, the first maximum
   wins and a NaN counts as the maximum. On the V100, `jnp.argmax` does not reliably
   return the first of several NaNs, so NaNs are located explicitly (argmax of the NaN
   mask); the six faces are reduced together as one (6, 289) array
   (docs/fasttrace_results.md).
5. Host: only the six maxima and positions come back. queue_moves does exactly what
   FaceMaxMovementPolicy.update does with them: mark the lattice cell done; keep a face
   unless `score < threshold`, so a NaN score is kept as in FFN; de-duplicate equal
   (score, position) pairs; sort descending; queue. The position is recorded in the
   visit history.

The constants come from FFN's own conversion: Canvas.__init__'s logit conversion of
init_activation, pad_value, move_threshold and segment_threshold, applied to the authors'
options. A test checks them against a real Canvas.

## Pass criterion (fixed now): bitwise identity

For every seed, the fast tracer must reproduce the recorded run exactly: the canvas
sha256, the visit history sha256 and the number of steps.
- Float32: the 12 benchmark seeds (gpu-baseline-a-20261019) and the 300 fresh seeds
  (fresh-fp32-20261023).
- `fp16_head32`: the 300 fresh seeds (fresh-fp16_head32-20261023).
- CPU (controller example and tests): the authors' notebook seed in float32 against the
  pinned CPU reference fingerprint (liconn_v2/reference.py).

Any seed that differs fails the milestone for that format. Differences are reported,
not tuned away. Because traces are identical, accuracy and the owner's 2% verdict for
`fp16_head32` carry over unchanged.

## Speed (reported)

In the same Slurm job and on the same GPU, the reference path and the fast path trace
the 12 benchmark seeds in both formats. Reported: time per step and the speed-up over
the reference path. Also reported: time per step on the 300 fresh seeds for each
format, against the recorded runs (separate jobs, approximate).

## Verification

- Unit tests (CPU):
  - the device face maxima with queue_moves against FFN's FaceMaxMovementPolicy.update on
    the full prediction (random arrays, ties, repeated edge maxima, NaN maxima, nothing
    above threshold), and the validity checks against a real Canvas;
  - the on-device box cut and seed-consistency update against FFN's NumPy code on random
    arrays, bitwise;
  - the full fast trace against the reference, bitwise. As implemented this uses the
    authors' real network on the notebook seed (CPU) rather than a small synthetic
    network, which tests the same path with the real model.
- Controller example: the notebook seed on CPU, bitwise against the pinned reference.
- GPU runs: owner-approved implementer runs on Wahab, recorded with source hashes, job
  IDs and output sha256, checked by an evidence test.

## Limits

V100 only for speed; other GPUs will differ. The fast tracer handles single-object
traces with the authors' settings, not FFN's full segmentation mode (multiple objects,
restrictors, resume). If the GPU compiler fuses the network with the surrounding steps
differently, outputs could differ. The design keeps the network as its own jitted call
to avoid this, and the identity check would catch it.
