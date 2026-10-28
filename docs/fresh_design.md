# fp16_head32 on 300 fresh seeds, with ERL: design (milestone 13, fixed before any new run)

Owner decisions (2026-10-23): confirm `fp16_head32` on fresh seeds before calling it a
speedup (docs/head32_results.md was measured on seeds that took part in choosing it),
and report a skeleton-based expected run length (ERL) as an additional metric.

## Crops and seeds (fixed now)

- Continue the pre-registered crop order of docs/manyseed_design.md (qualifying grid
  crops, `default_rng(0)` order) after the 13 crops already used: crop 13 onwards, one
  crop at a time, until 300 fresh seeds.
- Same seed rule: proofread axons and dendrites by the benchmark's size rules, up to 15
  dendrites then axons to 30 per crop, `default_rng(0)` per crop. Every object already
  seeded anywhere (benchmark, crop B, the 300 many-seed seeds) is excluded. The last crop
  is cut, in selection order, to exactly 300.
- Pinned by sha256 in docs/fresh_crops.json and docs/fresh_seeds.json.
- Owner decision (2026-10-23): the first fresh crops gave 14 and 16 seeds, below the
  estimate behind the original 7.5 GB, so the owner approved up to 12 GB for the fresh
  crops. The download stops before a crop that would take the fresh crops past 12 GB;
  if that happens before 300 seeds, nothing is pinned and the shortfall goes to the
  owner. The 12 GB cap stopped the download at 224 seeds from 19 crops (11.9 GB);
  the owner then raised the cap to 16 GB (2026-10-23). At 16 GB the download stopped at
  270 seeds from 25 crops (15.6 GB): yields fall because the classified axons and
  dendrites (855 in the volume) are being used up. The owner raised the cap to 20 GB.
  If the pre-registered crop order runs out before 300 seeds, nothing is pinned and the
  shortfall goes to the owner.

## Runs

- Reference: the authors' network in float32. Candidate: `fp16_head32`, unchanged
  (liconn_v2/earlystop.py). Both on Wahab V100s, deterministic XLA ops, one Slurm task
  per crop, the same runner as the many-seed runs.

## Verdict (decisive, unchanged rule)

The owner-confirmed many-seed rule (docs/manyseed_design.md): at most 3 of 300 new
merges and at most 3 of 300 new early stops (a per-seed precision or recall fall of
more than 10 points); mean recall and mean precision each fall by at most 1 point;
complete, pinned inputs, all 300 seeds. No other candidate is tried on these seeds. A
pass is the confirmation; a fail is reported as it is.

## Reported alongside (not pass rules)

- Absolute recall, precision, IoU and Dice against the proofread segmentation;
  counts of traces with recall below 50% (large splits), with any merge (1,000 voxels
  or more of another object) and with a large merge (10% of the trace or more).
- **ERL, adapted to single-seed traces:**
  - Skeleton: each seed's proofread object within its crop, skeletonised with
    kimimaro 5.8.5 (TEASAR), anisotropy 18 x 18 x 24 nm, settings fixed now:
    `scale=1.5`, `const=300` nm, `pdrf_scale=100000`, `pdrf_exponent=4`,
    `dust_threshold=1000` voxels, `fix_branching=True`, `fix_borders=True`. If kimimaro
    returns several pieces for one object, their edges are pooled.
  - Runs: the connected pieces of the skeleton whose edges have both ends inside the
    trace mask.
  - Seed ERL = sum of squared run lengths / skeleton length, set to 0 if the trace
    merges (1,000 voxels or more of another object, as in liconn_v2/benchmark.py).
  - Overall ERL is length-weighted over seeds (sum of squared runs / sum of skeleton
    lengths), reported for all seeds and by class, with and without the merge penalty.
  - Computed for the fresh seeds (float32, `fp16_head32`) and for the 300 many-seed
    seeds (float32, float16, `fp16_head32`).
  - Sanity gate: for at least 95% of objects, the skeleton length must be at most 3
    times the diagonal of the object's bounding box, checked separately for each seed
    set. If the gate fails, that set's ERL is not reported, only the failure. An earlier
    exploratory attempt with a generic skeletoniser produced skeletons about 50 times
    too long, which is why the gate exists.
  - An amendment to 5 times, made by the implementer after looking at 8 pilot
    skeletons, was flagged by the review as unapproved. The owner decided on
    2026-10-23 to keep the pre-registered 3 times, and the amendment was withdrawn.
- Time per step for both runs, in the same job allocation per crop where the queue
  allows; otherwise across jobs, as approximate.

## Environment

kimimaro 5.8.5 (and its dependencies) installed into the project's .venv from PyPI,
pinned. It is needed only on the Mac for ERL; the GPU runs do not use it.

## Verification

- Unit tests:
  - the crop and seed extension (order continued, exclusions, exact 300);
  - ERL on synthetic shapes: a straight tube's skeleton length is within 10% of the
    tube length; runs and squared-length sums on hand-built graphs; merge zeroing;
    length weighting;
  - the sanity gate.
- Controller example: the verdict and ERL recomputed from the recorded runs.
- GPU runs: owner-approved implementer runs, recorded with source hashes, job IDs and
  output sha256, checked by an evidence test.

## Changes after the independent method check (before any new run)

- The verdict pairs runs by position; fresh crops are numbered from 13, so each seed's
  crop is mapped to its run pair's position (examples/judge_fresh.py, with a test).
- Objects are skeletonised in their bounding box padded by 2 voxels, so that kimimaro's
  border handling sees only true crop borders.
- An empty skeleton fails the sanity gate; an empty trace is not a large merge; "any
  merge" uses the project's merge definition; ERL by class is also reported without
  the merge penalty.
- Tests with an asymmetric object check the x, y, z and z, y, x axis orders.

## Limits

The fresh seeds are partly what was left: only 855 classified axons and dendrites exist
in the volume, and objects already seeded, including large ones spanning several
crops, were excluded. So the fresh set may lean toward smaller or more local objects.


One volume, one depth range, V100 only. ERL is an adapted, crop-bounded measure for
single traces, comparable between networks here but not with published ERLs, which
use whole segmentations of large volumes. The proofread segmentation was made from the
authors' FFN output, so it is not independent of FFN.
