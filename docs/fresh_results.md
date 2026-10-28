# fp16_head32 on 300 fresh seeds, with ERL: results (runs 2026-10-23)

**Outcome: `fp16_head32` fails the owner-confirmed many-seed rule on 300 fresh seeds.**
It makes 5 new merges (limit 3), all in dendrites. Early stops (3) and the mean recall
and precision rules pass. Its earlier pass (docs/head32_results.md) was on seeds that
took part in choosing it and did not hold on fresh data. Per step it is 2.21x faster
than float32; whole traces are 2.07x faster because the merges add steps.

This is an implementer-run record of owner-approved runs on Wahab V100s:
- float32 reference: Slurm 7430481;
- `fp16_head32`: Slurm 7430482;
- one task per fresh crop (crops 13 to 40, 300 seeds).

tests/test_fresh_evidence.py checks the following against docs/fresh_evidence.json:
- the code that ran is the checked code;
- the outputs are the recorded ones;
- every seed ran in its format on a V100;
- the verdict, metrics and ERL below follow from the outputs.

## Fresh seeds

The pre-registered crop order continued from crop 13. Yields fell because the volume
has only 855 classified axons and dendrites and earlier seeds were excluded. The owner
raised the storage cap from 12 to 16 to 20 GB (docs/fresh_design.md). 300 seeds came
from 28 crops (13 to 40, 16 GB). 114 of them are dendrites (38%, against 24% in the
first 300).

## Verdict (decisive; against float32 on the same fresh seeds)

| rule | `fp16_head32` | |
|---|---|---|
| new merges (precision falls by more than 10 points), at most 3 | **5** | **fail** |
| new early stops (recall falls by more than 10 points), at most 3 | 3 | pass |
| mean recall falls by at most 1 point | -0.16 points | pass |
| mean precision falls by at most 1 point | -0.48 points | pass |
| complete, pinned inputs, all 300 seeds | yes | pass |

- The new merges are m16_dendrite_2739 (precision -86.4 points), m18_dendrite_278141
  (-54.0), m20_dendrite_393653 (-27.6), m27_dendrite_355966 (-40.7) and
  m35_dendrite_120894 (-75.3). None of them merges in float32. In `fp16_head32` each
  leaks into one neighbouring object (52,000 to 629,000 voxels), and the trace grows,
  for example from 168 to 1,717 steps.
- New early stops: m17_axon_288397 (recall -46.2), m28_axon_1254 (-10.4),
  m36_axon_91844 (-40.1).
- Avoided merges: m20_dendrite_357563 and m26_axon_364565 (the latter's float32
  trace already merged). Extensions: 4. Neither offsets new events.
- Dendrites: 5 new merges in 114 (4.4%). One-sided 95% upper bounds, assuming
  independent seeds (optimistic): new merges 3.5%, new early stops 2.6%, dendrite new
  merges 9.0%.

## Accuracy against the proofread segmentation (fresh seeds)

| | float32 | `fp16_head32` |
|---|---|---|
| mean recall | 84.15% | 83.99% |
| mean precision | 98.14% | 97.65% |
| mean IoU | 71.92% | 71.45% |
| mean Dice | 82.12% | 81.75% |
| traces with recall below 50% | 22 | 23 |
| traces with any merge | 28 | 31 |
| traces with a large merge (10% of the trace or more) | 8 | 11 |
| merged voxels | 2.93 M | 5.47 M |
| steps | 111,209 | 118,757 |
| time per step | 19.51 ms | 8.83 ms |
| total trace time | 2,170 s | 1,048 s |

## Accuracy against the proofread segmentation (first 300 seeds)

The same metrics for the 300 many-seed seeds (docs/manyseed_results.md,
docs/head32_results.md), from the recorded runs:

| | float32 | float16 | `fp16_head32` |
|---|---|---|---|
| mean recall | 84.72% | 84.61% | 84.99% |
| mean precision | 98.38% | 98.07% | 98.51% |
| mean IoU | 73.28% | 72.84% | 73.61% |
| mean Dice | 83.06% | 82.58% | 83.31% |
| traces with recall below 50% | 23 | 25 | 23 |
| traces with any merge | 23 | 25 | 30 |
| traces with a large merge (10% of the trace or more) | 6 | 7 | 5 |
| merged voxels | 1.95 M | 2.52 M | 1.46 M |
| steps | 93,863 | 95,927 | 92,792 |
| time per step | 19.58 ms | 8.88 ms | 8.95 ms |

## ERL (adapted, single traces; reported, not a pass rule)

The pre-registered sanity gate, kept by the owner (docs/fresh_design.md): at least 95%
of a seed set's skeletons within 3 bounding-box diagonals.

| seed set | skeletons within 3 diagonals | gate | ERL |
|---|---|---|---|
| fresh 300 | 274 (91.3%); median 1.49, maximum 4.66 | fails | **withheld** |
| first 300 | 292 (97.3%); median 1.43, maximum 4.10 | passes | reported below |

| length-weighted ERL, first 300 seeds, um | float32 | float16 | `fp16_head32` |
|---|---|---|---|
| all | 7.18 | 7.30 | 6.99 |
| axons | 8.41 | 8.60 | 8.42 |
| dendrites | 3.93 | 3.87 | 3.22 |
| all, no merge penalty | 7.85 | 8.11 | 7.87 |
| axons, no merge penalty | 9.03 | 9.45 | 9.09 |
| dendrites, no merge penalty | 4.76 | 4.59 | 4.65 |

ERL is bounded by the 500^3 crops (about 9 x 9 x 12 um) and computed per single trace,
so it compares networks here, not with published ERLs. On the first 300 seeds,
`fp16_head32` is close to float32 overall and lower on dendrites. An amendment of the
gate to 5 diagonals, made without the owner's approval, was withdrawn; under it the
fresh-seed ERL would have been reported.

## What this shows

- The float32 output head fixed most of float16's early stops, on the seeds where they
  were found, but it does not make float16 meet the owner's rule on fresh data: rare
  dendrite merges remain, at 5 in 300 against a limit of 3.
- The fresh seeds hold more dendrites (114 against 72), and all 5 new merges are in
  dendrites, against none in the first 300. The two sets also differ in objects, crops
  and selection history, and the first set took part in choosing the candidate, so
  these runs do not show what explains the different verdicts.
- These results do not show why float16 still merges dendrites. That would need the
  milestone-11 style diagnosis on these merges.

## Limits

One volume, one depth range, V100 only. The fresh seeds are partly the classified
objects left after earlier seed sets, so they may lean toward smaller or more local
objects. ERL is an adapted, crop-bounded single-trace measure. The proofread
segmentation is not independent of FFN.
