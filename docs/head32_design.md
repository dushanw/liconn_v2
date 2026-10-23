# fp16_head32 on the 300-seed rule: design (milestone 12, fixed before any new run)

Owner decision (2026-10-22): judge `fp16_head32`, the candidate selected by the
pre-registered rule of docs/earlystop_results.md, under the owner-confirmed many-seed
rule (docs/manyseed_design.md).

## Candidate

`fp16_head32` exactly as screened in milestone 11 (liconn_v2/earlystop.py, layout
`fp16_head32`): the authors' network with its input layers and 20 residual blocks in
float16, and its head (last LayerNorm, ReLU and 1x1x1 output convolution) in float32.
Parameters stay float32 and are cast at use; input and output are float32.

## Runs

- Candidate: the 300 recorded seeds (docs/manyseed_seeds.json) on the 13 pinned crops,
  Wahab V100s, deterministic XLA ops, one Slurm task per crop. This is the same code
  path as the many-seed runs, with the network installed from earlystop.apply_fn.
- Reference: the existing float32 many-seed run (manyseed-fp32-20261021, pinned by
  docs/manyseed_evidence.json). It is not re-run.

## Verdict (unchanged, fixed 2026-10-21)

examples/judge_manyseed.py against the float32 reference:
- at most 3 of 300 new merges and at most 3 of 300 new early stops (a per-seed
  precision or recall fall of more than 10 points);
- mean recall and mean precision each fall by at most 1 point;
- complete, pinned inputs, all 300 seeds.

Reported alongside: the same numbers for plain float16 (docs/manyseed_results.md), and
the time per step of `fp16_head32` against the float32 reference run. Those were
separate jobs on V100 nodes, so the speed comparison is approximate.

## Fixes first (open findings of the milestone 11 review)

1. examples/analyze_earlystop.py stops with an error, and writes no conclusions or
   selection, when any rerun fails bitwise reproduction or the movement emulation.
2. The open-loop voxel flip count treats a non-finite candidate output as a changed
   decision, as precision.screen does, with a regression test (NaN against a float32
   background voxel).

## Not included

A variant combining the float32 head with float32 residual sums (which also removed
float16's rare overflow in milestone 11) was not pre-registered. It is left out unless
the owner adds it.

## Verification

- Unit tests: the candidate runs through the many-seed runner (variant names are
  checked); the two fixes.
- Controller example: the verdict recomputed from the recorded runs.
- GPU runs: owner-approved implementer runs, recorded with source hashes, job IDs and
  output sha256, checked by an evidence test.

## Limits

The same 300 seeds of one volume as the float16 verdict, V100 only. A pass would mean
`fp16_head32` meets the many-seed rule on these seeds, not on other data or hardware.
