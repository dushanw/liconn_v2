# Factorised FFN pilot, part 1: port and student (results, 2026-10-31)

Design: docs/factorised_design.md. Code: liconn_v2/factorised.py, tests/test_factorised.py,
examples/run_factorised_port.py (writes `factorised_port.json`). The controller verifies the
unit tests, which repeat each check below; the example's JSON is a fuller local record (run
2026-10-31 on the owner's laptop), not verified by the controller. CPU, float32, PyTorch 2.7.1 and
the pinned JAX 0.4.38. No training and no GPU.

## Checks (pre-registered tolerance: maximum absolute logit difference < 1e-3)

| check | inputs | max abs logit diff | threshold flips (0.6) | result |
|---|---|---|---|---|
| PyTorch port vs the authors' JAX network | 15 inputs recorded every 24th step of the notebook's reference trace (359 steps; trace identical to the recorded reference fingerprint) | 4.1e-5 | 0 of 539,055 | pass |
| same | 16 notebook-crop boxes with masks built from the proofread labels | 2.5e-5 | 0 of 574,992 | pass |
| same | 2 random inputs (unit test) | < 1e-3 | — | pass |
| undilated student at initialisation vs teacher with all-unknown mask | the 16 boxes | 0.0 | 0 | pass |
| tiled dense encoding vs one pass; box crop vs encoder on box with halo | random volume, shallow encoder (unit tests) | < 1e-3 | — | pass |
| teacher FLOP per step | — | 81.6 GFLOP | — | reproduces the manuscript value |

For scale: the mask changes the teacher's output by up to 5.9 logits on these boxes (mean 1.2;
125,614 of 574,992 voxels cross the threshold), so the port differences above are about five
orders of magnitude smaller than the effect the decoder must learn.

## Model sizes and cost

- Parameters: teacher 1,138,081; student encoder 915,424, decoder 250,369.
- Per 33^3 box: teacher 81.6 GFLOP, decoder 17.9 GFLOP (4.5x less per step).
- Crop 12 at FFN's recorded 137,723 steps (PFLOP):

| encoder tiling | teacher | encoder (with halo) | decoder | student | ratio |
|---|---|---|---|---|---|
| one pass over the padded crop | 11.24 | 0.33 | 2.47 | 2.81 | 4.0x |
| 250^3 tiles | 11.24 | 0.47 | 2.47 | 2.94 | 3.8x |
| 125^3 tiles | 11.24 | 0.84 | 2.47 | 3.31 | 3.4x |

These are FLOP counts, not times (LayerNorm, ReLU, kernel launches and host bookkeeping are not
counted).

## What this does and does not show

- Shows: the PyTorch port reproduces the authors' network to about 4e-5 logits on real
  mid-trace inputs, so it can serve as the teacher on MIT B200s (GPU numerics to be checked
  there); the student is a faithful slice of the teacher at initialisation; dense encoding
  is tiled correctly.
- Does not show: that the student can learn to trace, how fast it will be on a GPU, or anything
  about accuracy. The dilated pilot student does not equal the teacher at initialisation.
- The independent method check (Cursor) found that the first encoder cost ignored the tiling
  halo (corrected above) and listed risks for part 2 (mask reach at step 0, dilation across thin
  structures, box-edge targets, mask-stem start, launch overheads); these are in the design.

## Notes on the task record

- The task contract's limits still say the checks use constructed masks, "not recorded trace
  inputs"; that sentence predates the recorded-trace test, which is now part of the suite
  (`test_port_matches_authors_on_recorded_trace_inputs`).
- Memory (method check): one 500^3 crop of float32 encoder features is 16 GB, and one 568^3
  activation is 23.5 GB, so a one-pass dense encoding needs a large-memory GPU (fine on a B200,
  not on a 16 GB V100); a V100 needs tiles. Caching float32 features for crops 0-11 would take
  about 192 GB (half in float16).
