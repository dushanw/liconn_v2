# Why float16 stops traces early: results (runs 2026-10-22)

**Outcome: neither pre-registered hypothesis is supported by these runs. That does
not rule them out.** Only 2 of the 10 early stops had a threshold frontier, and those
compared different box centers (A). The bias test (B) had too few near-threshold faces
to detect a bias. What the runs do show: float16's trace leaves float32's in all 20
traces, early stops and controls alike, and the first differences are mostly moves to
a different voxel of the same face (argmax). The candidate screen (C) selects `fp16_head32` (float16 network, float32
output head). It cuts the steps whose queued moves differ from 84% to 16% on the
early-stop seeds, at the same speed (3.4x per evaluation). That screen is open loop
on 20 seeds; no speedup is claimed, and `fp16_head32` would need the 300-seed rule.

This is an implementer-run record of owner-approved runs on Wahab V100s:
- float32 with open-loop evaluation of the candidates: Slurm 7392292;
- float16: Slurm 7392293.

A first pair of jobs (7392246, 7392247) picked the float16 run's controls by float16
step counts instead of float32's (a bug in examples/run_earlystop.py). It was fixed
and both runs were repeated; the first pair is kept as `superseded-earlystop-*` and
not used.

tests/test_earlystop_evidence.py checks the following against
docs/earlystop_evidence.json:
- the code that ran is the checked code, and later edits touched only declared
  analysis functions;
- the outputs are the recorded ones;
- every rerun reproduced its many-seed trace bitwise;
- the open-loop emulation of FFN's move inputs matched FFN at every step;
- the numbers below follow from the outputs.

## Seeds

The 10 early stops of docs/manyseed_results.md and 10 controls picked by the
pre-registered rule (same class, same crop where possible, closest float32 step
count). Both runs traced the same 20 seeds and reproduced the many-seed traces bitwise.

## A. Where float16's trace stops (primary: the frontier)

| frontier kind | early stops | controls |
|---|---|---|
| threshold | 2 | 2 |
| argmax | 7 | 7 |
| queued, not evaluated | 1 | 1 |

- Supported only if at least 7 of the 10 early stops were "threshold": 2 were. **Not
  supported.**
- "Queued, not evaluated": float16 queued a move into the lost cell but never
  evaluated a box there. The recorder does not see why. (The design called this
  "dropped at pop", after FFN's rule of skipping popped moves whose canvas value has
  fallen below 0.6, but that reason was not verified.)
- The frontier is the first region float32 reaches and float16 does not. It is not
  shown to be the region whose loss makes the recall drop: controls have frontiers
  too, without losing recall.
- The early stops and the controls look alike. Float16 visits different cells than
  float32 in every one of the 20 traces, controls included.
- In 7 of the 10 early stops, float16's step in the parent cell was not at the same
  center as float32's. So even the two "threshold" frontiers compare different
  positions, not one face falling just below 0.6. Their float32 face scores were 2.2
  and 2.5 logits above the threshold.
- Secondary (the original "first difference"): argmax in 9 early stops and 10
  controls, and order in 1 early stop. Never threshold.

## B. Bias of float16 on the same inputs (primary: the controls)

- Faces within 0.05 logits of logit(0.6) are rare: 7 in the controls and 6 in the
  early stops. Controls' mean error is -0.0003 logits; 95% interval -0.0028 to
  +0.0037. **Not supported.** Too few near-threshold faces exist for this test to say
  much, so `fp16_shift` was not screened.
- Over all control faces, float16 is very slightly lower (mean -0.0003 logits;
  interval -0.00033 to -0.00030). That is small compared with how far moving faces sit
  above the threshold (median 2.5 logits).
- Float16 overflows on a few inputs. 6 of 27,528 faces are NaN in `fp16` (and in
  `fp16_head32`), all on faces where float32 moves. FFN never moves through a NaN
  face, so on those inputs float16 would not make that move. These are all in one control seed
  (m2_dendrite_221993). `fp16_res32`, with float32 residual sums, has none.

## C. Candidate fixes, open loop

| variant | speed per evaluation vs float32 | steps whose queued moves differ (early stops) | (all 20 seeds) |
|---|---|---|---|
| fp16 | 3.45x | 84.2% | 84.4% |
| **fp16_head32** | 3.43x | **16.0%** | 16.0% |
| fp16_res32 | 3.32x | 83.4% | 84.0% |

By the pre-registered rule (at least 1.5x faster than float32, fewer differing steps
than plain float16, then the fewest), **`fp16_head32` is selected** for a possible
300-seed milestone.

## Exploratory (not pre-registered, added after the runs)

- 91% of moving float32 faces have a highest logit above 2.9, on the network's
  saturated plateau (median 2.94).
- On the same inputs, float16 moves the same face to a different voxel 8,229 times
  on the plateau and 63 times below it. `fp16_head32` does so 649 and 22 times,
  `fp16_res32` 8,154 and 67 times.
- A possible explanation, not tested here: many voxels on a face share nearly the same
  saturated logit. Float16 output (about 3 significant digits) would round several of
  them to the same value, so argmax could pick a different one, while a float32 output
  head would keep them distinct. Each shift moves the next box by a voxel or more, and
  the traces could then drift apart, some ending earlier.

## What this shows

- These runs describe how float16 diverges; they do not establish what causes the
  early stops. Float16 drifts from float32 in every trace, mostly through argmax
  shifts on the saturated plateau. A plausible hypothesis, not tested: early stops are
  where such drifted trajectories happen to end sooner. Threshold crossings and a bias
  against moving are not ruled out; they were rare or undetectable here.
- On these 20 seeds' inputs, most argmax shifts disappear when only the output head
  runs in float32, at almost no cost in speed. Separately, float32 residual sums
  removed float16's rare overflow there, which a float32 head alone did not.

## Limits

20 seeds of one volume, V100 only. The candidate screen is open loop: it compares
decisions on float32's inputs, not whole traces. Whether `fp16_head32` passes the
300-seed rule is not known. The plateau explanation is a hypothesis.
