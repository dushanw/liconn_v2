# Many-seed tolerance: design (milestone 10, fixed before any new data was read)

Owner decisions (2026-10-21):
- Judge speed variants over many more seeds, with an allowance for rare merge flips,
  instead of per-seed rules at 1% (docs/benchmark_design.md).
- Numbers: 300 fresh seeds; a flip is a per-seed precision or recall change of more
  than 10 points; at most 3 of 300 new merges and at most 3 of 300 new early stops.
- Approved: about 11 GB of public crop data. Not wanted: a CPU noise-floor run.

This document fixes how crops, seeds and verdicts are chosen before any new crop or
trace is seen.

## Crops

- Volume: ExPID82_1, the image at 18 x 18 x 24 nm (scale 1, 3825 x 4870 x 795 voxels)
  and the proofread segmentation 231030_agg_240123 on the same grid, as for crops A
  (benchmark) and B (held-out).
- Candidate positions: 500^3 crops at x0 = 100 + 500 i, y0 = 83 + 500 j, z0 = 111
  (the grid through crop A; z 111-611 is where labels are dense), fully inside the
  volume. A candidate that overlaps crop A or crop B is excluded.
- Coverage: the labeled fraction of the crop, measured on the coarse proofread level
  (288 x 288 x 384 nm, scale index 4; the level read for docs/training_feasibility.md),
  using every coarse cell the crop touches. A candidate needs at least 50%.
- Choice: sort qualifying candidates by (x0, y0), permute them with
  `numpy.random.default_rng(0)`, and take the first 10 (more in the same order if
  needed; see Seeds).
- Storage: raw as uint8; proofread labels as uint32 after checking that every ID is
  below 2^32, to halve disk use (about 0.6 GB per crop, about 6 GB for 10 crops).
  Both are pinned by sha256 in docs/manyseed_crops.json when first fetched.

## Seeds

- Candidates per crop: the benchmark rule (liconn_v2/benchmark.py): proofread axons
  and dendrites with at most 350k voxels in the crop and at least 20k inside the
  16-voxel margin; seed at the deepest interior point.
- Excluded: every object already seeded in the benchmark, in crop B, or in an
  earlier new crop (an object can span crops).
- Per crop, in crop order: `default_rng(0)`; up to 15 dendrites, then axons to fill
  30 seeds. Dendrites are scarcer than axons (crop A had 5 qualifying, crop B 10), so
  the 300 seeds will hold fewer than 150 dendrites. All merge flips seen so far were
  in dendrites, so results are also reported for dendrites alone.
- If the 10 crops give fewer than 300 seeds, the shortfall is reported to the owner
  before any further crop is fetched (the approved download covers 10 crops).
- Owner decision (2026-10-21, after the 10 crops gave 252 seeds): continue down the
  same rng(0) crop order, one crop at a time, until there are at least 300 seeds; the
  last crop's seeds are kept in their selection order (dendrites first, then axons)
  up to a total of exactly 300. Stored data stays within the approved 11 GB.
- Names: `m<crop index>_<class>_<object id>`.

## Runs

- Reference: the authors' network in float32 on Wahab V100s (`LICONN_BACKEND=gpu`,
  deterministic XLA ops), the same code path as the held-out reference.
- First candidate: float16 (liconn_v2/precision.py, variant fp16).
- One Slurm array task per crop and variant (30 seeds per task).

## Verdict (fixed in advance)

Per seed, against the reference, in percentage points:

| event | definition |
|---|---|
| new merge | precision falls by more than 10 points |
| new early stop | recall falls by more than 10 points |
| avoided merge | precision rises by more than 10 points |
| extension | recall rises by more than 10 points |

A variant passes if all hold:
1. new merges: at most 3 of 300;
2. new early stops: at most 3 of 300;
3. mean recall over all seeds falls by at most 1 point;
4. mean precision over all seeds falls by at most 1 point;
5. completeness: every selected seed is present in both runs, with the same inputs
   and seed coordinates;
6. the verdict covers all 300 seeds (the limits are counts out of 300, so a smaller
   set cannot pass; the development verdict therefore never passes).

Avoided merges and extensions are reported but never offset new merges or early
stops. Limits are compared with a float slack of 1e-9, as in liconn_v2/compare.py, so
that a change of exactly 10 points (or a mean change of exactly 1 point) is not
pushed over the limit by rounding. Runs must carry input hashes equal to the pinned
crop and checkpoint hashes.

Reported alongside (information, not pass rules), added after the independent method
check and before any new trace was run:
- one-sided 95% Clopper-Pearson upper bounds on the new merge and new early stop
  rates, and on the dendrite new merge rate; these assume independent seeds, which
  seeds within a crop are not, so they are optimistic;
- dendrite-only counts and mean changes (72 of the 300 seeds are dendrites);
- events per crop;
- the separate distributions of recall and precision changes below 10 points;
- seeds whose reference precision is below 20%: the precision rule can barely see a
  further merge there;
- empty candidate traces, if any.

Known limit of the definition (raised by the method check; for the owner): a new
merge is seen only through precision. A further merge on a seed whose reference
already merged, or a small merge onto a very large object (about 25k voxels onto
250k), can change precision by less than 10 points and is then not counted.

Why 10 points: in the 24 float16 comparisons so far (benchmark and crop B), every
per-seed recall or precision change is either below 5 points or above 12 points.

The same verdict is also computed, for information only, on the 24 development
seeds (benchmark crop A and crop B), which have been examined too closely to count.

## Verification

- Unit tests: the crop rule (grid, overlap exclusion, coverage, deterministic order)
  on synthetic coverage maps; the seed rule; the verdict on synthetic records,
  including the 10-point boundaries, avoided merges not offsetting new merges, and
  incomplete runs failing.
- Controller example: the verdict on the 24 development seeds from the recorded runs.
- GPU runs: owner-approved implementer runs on Wahab, recorded with source hashes,
  job IDs and output sha256, checked by an evidence test.

## Limits

Ten further crops of one volume (ExPID82_1), one depth range, one GPU type (V100).
Seeds within a crop are not fully independent, so the confidence bounds are
approximate. A pass would mean the variant's merge flips are rare in this dataset,
not that they never happen.
