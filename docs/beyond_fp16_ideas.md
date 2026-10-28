# Going beyond float16: ideas (brainstorm, 2026-10-21)

Owner request: brainstorm how to make FFN inference faster than the float16 result.
These are ideas with rough expectations, not measured results.

**Update after held-out confirmation (docs/heldout_results.md):** float16 failed on
held-out seeds. Two dendrites formed large false merges that float32 does not.
Changes to the network's arithmetic (section C) therefore carry a real risk of
failing the per-seed 1% rules, however small the per-evaluation change. Section A,
which keeps the computation bitwise identical, is now the clear first priority. Each would be
screened and benchmarked against the reference under the 1% tolerance, and then
confirmed on held-out seeds, like float16.

## Where the time goes now (V100, float16)

- Per FFN step: about 9.0 ms (8.99 ms measured over 6,117 benchmark steps).
- The network evaluation itself is about 4.0 ms (3.96 ms per call at batch 1 in the
  screen).
- So roughly 5 ms per step, more than half, is now spent outside the network:
  - copying the image and seed boxes to the GPU and the result back (two 33^3 boxes in,
    one out);
  - canvas updates in NumPy (overwrite of a 33^3 region, the disconnectedness guard);
  - the movement policy (face maxima, queue, quantization) in Python;
  - thread hand-offs in FFN's executor (client/server queues).

That is the main lesson. After float16, the network is no longer the main cost.
Making the network faster again can save at most about 4 ms of the 9 ms. A 2x
reduction of the rest would save about 2.5 ms.

## A. Remove the non-network overhead (no change to the method's decisions)

These keep every computation the same in principle, so they are the safest. Some
may even be bitwise identical to float16.

1. **Keep the canvas on the GPU.** Store the 500^3 canvas and image as device arrays.
   Do the box crop, the network call and the canvas update in one compiled function,
   and copy back only the 6 face maxima the movement policy needs. This removes most
   host-device copies and NumPy work. Untested estimate: it might bring the step closer to the
   network time, but that has to be
   measured. It needs care to reproduce FFN's exact update rules, including the
   disconnectedness guard.
2. **Drop the executor threads for single-object tracing.** FFN's client/server
   executor exists to batch many canvases. With one canvas, each step pays thread
   synchronisation for nothing. A direct call path is simple, and the output is
   unchanged.
   **Probed (exploratory, Slurm 7370208, float32 network, 12 benchmark seeds):** a
   client that calls the compiled network directly reproduced every canvas and visit
   order bitwise. It took 17.88 ms per step against the reference's 19.67 ms, about 9%
   faster. Data and script: `runs/probe-direct-call-20261021/` in the data folder (the
   LICONN cache; here the repository's data/ folder).
3. **Profile (done, exploratory).** One profiled float16 trace of the notebook seed on
   a V100 (Slurm 7370178, after a warm-up trace; 360 steps, 10.2 ms per step). Data are
   in `runs/profile-fp16-20261021/` in the data folder: profile.json, the log and the
   script. The
   split from FFN's counters and cProfile is approximate:

   | part of a step | ms per step |
   |---|---|
   | network computation | ~4.0 |
   | assembling the input on the GPU (concatenate, dtype conversion) | ~1.7 |
   | canvas update in NumPy (`update_at` beyond `predict`) | ~2.0 |
   | executor thread hand-offs and queues | ~1.5 |
   | copying the result back | ~0.7 |
   | movement policy | ~0.3 |

   About 60% of a step is outside the network computation. That supports trying A1
   and A2 first. Separately, each trace pays about 0.15 s once to clear the 500^3
   canvas arrays.

## B. Run more work per network call (same decisions, better hardware use)

4. **Trace several objects at once.** A batch of 1 underuses the GPU. Tracing many
   seeds or objects in parallel, with one network call per batch of FOVs (FFN already
   supports multi-client batching in dense segmentation), raises throughput per GPU
   without changing any single trace. It does not make one object faster; it makes
   segmenting a volume faster. The owner's goal ("the method, on any infrastructure")
   favours per-object speed, so this is secondary, but it is the standard way FFN is
   deployed.
5. **Speculatively evaluate queued candidates.** Several queued positions could be
   evaluated in one batched call before they are needed, keeping the results that are
   still valid when popped. This changes timing only if done exactly. The bookkeeping is
   complex and the gain uncertain.

## C. Make the network cheaper (changes decisions; needs the screen and the tolerance)

6. **Lower precision beyond float16.**
   - int8 with per-channel scales calibrated on recorded inputs (post-training
     quantization, no retraining). On GPUs with int8 tensor cores, int8 throughput is
     typically higher than float16; not measured here. Hypothesis: the disturbance
     may be larger than float16's 0.006% decision changes, but a calibrated int8 network
     can have different error patterns. A screen would show whether it stays near
     float16 or approaches the
     block-drop level (1.2%, which failed).
   - FP8 on newer GPUs (H100), similar trade-off.
7. **Mixed precision by layer.** Keep the few most sensitive layers in float16 and
   push the rest lower. The screen can measure per-layer sensitivity the way it ranked
   blocks.
8. **Compiler/runtime tuning.** XLA autotuning flags, cuDNN algorithm choice, or
   exporting the network to TensorRT. This is the same arithmetic with faster kernels;
   it may change results at rounding level, like float16. Expected gain on the 4 ms
   network part: unmeasured; any change of kernels may also change results at
   rounding level.

## D. With training (if the owner allows it; see docs/training_feasibility.md)

9. **Distil a smaller or shallower network** (fewer blocks, or 16-24 channels instead
   of 32) from the authors' network, on teacher-labelled inputs, including the
   student's own trajectories. Network cost scales with depth and roughly with the
   square of width, so a 2-4x cheaper network is the target. Combined with float16 and
   (A), this is where the largest gains are, but also the largest risk to the 1%
   rule.
10. **Quantization-aware fine-tuning** to make int8 safe, if (6) is close but fails.

## What removing overhead alone can reach

In float32 on the V100, one network evaluation in the screen took 14.69 ms, measured
as a Python call that includes moving the input and output. A reference step took
19.67 ms. Removing only the work around that evaluation, with the evaluation itself
unchanged, can therefore save at most about 1.3x. So far only the direct call (A2) has
been shown to keep results bitwise identical; a GPU-resident canvas (A1) has not
been built or tested.

This is not a limit on all result-preserving speedups. Faster execution of the same
float32 network could also preserve results: memory layout, fused or CUDA-graph
execution, or batching several boxes per call, if the GPU's arithmetic order stays
unchanged. Whether a given option keeps results bitwise identical must be tested,
as the direct call was.

Changes to the network's arithmetic or structure (float16, int8, smaller networks)
are where larger gains are expected. float16 failed on held-out seeds through
new false merges. The same risk may apply to other such changes (a hypothesis, not
tested) under the
per-seed 1% rules. Whether to accept that risk, for example with a tolerance judged
on more seeds, is an owner decision.

## Suggested order

1. Profile the 5 ms of non-network time (cheap, no risk).
2. (A1/A2) GPU-resident canvas and a direct network call. The direct call (A2) is
   the only change shown so far to keep results bitwise identical (9% faster in
   float32). Removing all overhead around the unchanged evaluation can give at most
   about 1.3x on the V100.
3. (C6) int8 screen on recorded inputs (about one GPU hour). Untested hypothesis:
   after the float16 held-out result, int8 may also risk merge flips. But per-evaluation
   errors and closed-loop merge outcomes need not scale simply with precision, and a
   calibrated int8 network can produce different error patterns, so only a screen,
   benchmark and held-out run can tell.
4. (D9) distillation pilot, only if the owner allows training.
