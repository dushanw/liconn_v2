# Held-out confirmation of float16 (milestone 9)

Owner decision (2026-10-21): confirm the float16 result (docs/precision_results.md)
on held-out seeds before calling it a speedup.

## Held-out data (fixed before any held-out trace was run)

- Crop B of ExPID82_1: x, y, z offset 2200, 2600, 111 (18 x 18 x 24 nm level), 500^3.
  It does not overlap the benchmark crop (x 1100-1600, y 1083-1583). The depth range
  is the same, where labels are dense: 57.9% of crop B's voxels are labeled (1,514
  objects at 144 x 144 x 192 nm), as in the benchmark crop (57.9%, 1,385).
- The same public volumes, cached separately (`heldout-crop-b/` in the data folder)
  and pinned by sha256 in liconn_v2/heldout.py.
- Seeds use the benchmark's rule: proofread axons and dendrites with at most 350k
  voxels in the crop and at least 20k inside the 16-voxel margin, deepest interior
  point, rng seed 0, up to 6 per class. Every benchmark seed object is excluded, and
  there is no notebook seed. In crop B: 30 axon and 10 dendrite candidates; 6 + 6
  seeds.

## Runs and pass rule (fixed in advance)

- Held-out reference: the authors' network in float32 on Wahab V100s
  (`LICONN_BACKEND=gpu`, deterministic XLA ops).
- Candidate: the same network in float16 (liconn_v2/precision.py, variant fp16).
- Pass: examples/compare_runs.py reports passes_tolerance for float16 against the
  held-out float32 run, under the owner-confirmed tolerance (every rule at 1%; all
  seeds present; same inputs). Mask IoU, steps and per-step time are reported too.
- The 1% tolerance excludes dendrite_1525 and axon_218 from the sensitive-seed group
  by name. Those seeds are not in crop B, so the "without sensitive seeds" group
  equals all seeds.

## Verification

- Unit tests: pinned crop hashes; the seed rule (deterministic, inside the margin,
  correct classes, no benchmark seed object, size rules).
- The controller example runs one held-out seed on CPU in float32.
- The GPU runs are owner-approved implementer runs on Wahab, recorded with source
  hashes, job IDs and output sha256, and checked by an evidence test.

## Limits

One additional crop of the same volume, 12 seeds, V100 only. Passing would be
evidence that float16 generalises within this dataset, not across datasets or
hardware.
