# Lever B, part 2: lower-precision arithmetic (milestone 8)

Owner decision (2026-10-21): try option (a), lower-precision evaluation of the
authors' network, with no training. Weights, network structure, movement policy,
thresholds and box size are unchanged; only the number format used inside the
network changes.

## Variants

The authors' network is computed in 16-bit floating point instead of 32-bit. The
parameters stay stored in float32 and are cast at use. The input and the final
output stay float32.

| variant | convolutions | LayerNorm |
|---|---|---|
| fp16 | float16 | float16 |
| fp16_ln32 | float16 | float32 |
| bf16 | bfloat16 | bfloat16 |
| bf16_ln32 | bfloat16 | float32 |

liconn_v2/precision.py defines the network once more, with the checkpoint's layer
names and a dtype per layer type. In float32 it must equal the authors' network
bitwise (tested). pruned.py, reference.py, benchmark.py and run_benchmark.py stay
byte-identical, because earlier evidence pins them. A variant is installed by
replacing the tracer's compiled network function, as in the block-drop runs.

What it buys: FLOPs stay the same. 16-bit arithmetic halves the memory traffic and
runs on the tensor units of most current accelerators. The V100 has float16 tensor
cores but no native bfloat16, so bfloat16 may not be faster there. Speed is therefore
reported as measured time per evaluation and per step on the same GPU type as the
reference (V100), with the hardware named. It is not a hardware-independent number.

## Step 1: screen

The same inputs and measures as the block-drop screen. Network inputs of every 8th
step are recorded from the 12 reference traces, and each trace must match the
reference's canvas and visit-order sha256. For each variant:
- flip rate at logit(0.6), lost and gained;
- mean |logit difference| and mean signed shift;
- median time per evaluation at batch 1 on the V100, as FFN calls the network,
  against float32 on the same node.

## Step 2: benchmark (rule fixed before the screen)

The fp16 variant and the bfloat16 variant with the lower flip rate are each run on
the 12 benchmark seeds on Wahab V100s. Each is compared with the reference by
examples/compare_runs.py: passes_tolerance (every rule at 1%), per-seed IoU and
recall change, and steps. Per-step trace time is also reported against the
reference's 19.7 ms per step.

## Verification

- Unit tests: the float32 configuration equals the authors' network bitwise; the
  16-bit variants run, return float32 output, and differ from float32 by a small
  amount on random input; variant names are validated.
- The controller example runs the screen on CPU for the notebook seed with a small
  sample. CPU timings are not the result.
- The GPU screen and benchmark runs are owner-approved implementer runs on Wahab,
  recorded with source hashes, job IDs and output sha256, and checked by an evidence
  test.

## Limits

V100 only (A100 not tested). Single-object traces on one crop. A variant passing on
12 seeds would still need held-out confirmation.
