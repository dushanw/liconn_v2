# fp16_head32 on the 300-seed rule: results (runs 2026-10-22)

**Outcome: `fp16_head32` passes the owner-confirmed many-seed rule on the 300 seeds,
at 2.19x float32's speed per step.** It makes 1 new merge and 1 new early stop (limit
3 each). Mean recall and mean precision rise slightly. Plain float16 failed the same
rule with 10 new early stops.

**Caveat: these are not fresh seeds for this candidate.** `fp16_head32` was chosen
(docs/earlystop_results.md) on 20 of these 300 seeds: float16's 10 early stops and
10 controls. The choice used an open-loop measure, changed move decisions, not the
traces judged here. Still, all of float16's early stops are among those 20 seeds by
construction, so the drop from 10 early stops to 1 is measured where the candidate
was picked. On the other 280 seeds, neither variant has an early stop, and
`fp16_head32` has 1 new merge against float16's 3. A confirmation on fresh seeds is
needed before calling this a speedup.

This is an implementer-run record of an owner-approved run on Wahab V100s:
`fp16_head32`, Slurm 7404325, one task per crop (13 crops, 300 seeds). The float32
reference is the existing pinned run (Slurm 7389883, docs/manyseed_results.md); it was
not re-run.

tests/test_head32_evidence.py checks the following against docs/head32_evidence.json:
- the code that ran is the checked code;
- the outputs are the recorded ones;
- every seed ran in this format on a V100;
- the verdict and the comparison below follow from the outputs.

## Verdict (against float32)

| rule | `fp16` (docs/manyseed_results.md) | `fp16_head32` | |
|---|---|---|---|
| new merges, at most 3 | 3 | 1 | pass |
| new early stops, at most 3 | 10 | 1 | pass |
| mean recall falls by at most 1 point | -0.11 points | +0.27 points | pass |
| mean precision falls by at most 1 point | -0.31 points | +0.13 points | pass |
| complete, pinned inputs, all 300 seeds | yes | yes | pass |

Reported for information:
- Avoided merges: 2. Extensions: 4. Neither offsets new events.
- One-sided 95% upper bounds, assuming independent seeds (optimistic): new merge rate
  1.6%, new early stop rate 1.6%, dendrite new merge rate 4.1% (0 of 72).
- Dendrites: no new events, 1 avoided merge; mean recall -0.16 and mean precision
  +0.62 points.
- Seeds with events:
  - m1_axon_132476: new merge (precision -33.3 points);
  - m10_axon_145819: new early stop (recall -26.7 points). It is one of float16's 10,
    and the only one still stopping early.
  - Avoided merges: m2_dendrite_17382 (+71.8), m8_axon_78192 (+24.0).
  - Extensions: m0_axon_30306, m0_axon_3428, m4_axon_59400, m5_axon_228843.
- Median absolute change below 10 points: 0.26 recall and 0.04 precision points
  (float16: 0.48 and 0.10).

## Split by the seeds used to choose the candidate

| seeds | variant | new merges | new early stops | avoided merges | extensions |
|---|---|---|---|---|---|
| the 20 selection seeds | fp16 | 0 | 10 | 0 | 0 |
| | fp16_head32 | 0 | 1 | 0 | 0 |
| the other 280 | fp16 | 3 | 0 | 2 | 4 |
| | fp16_head32 | 1 | 0 | 2 | 4 |

## Speed

- 8.95 ms per step for `fp16_head32` against 19.58 ms for the float32 reference
  (2.19x). Plain float16: 8.88 ms.
- Total steps: 92,792, against 93,863 for float32 (-1.1%). Total trace time: 830 s
  against 1,838 s (2.21x).
- The reference ran in a different job, possibly on different V100 nodes, so this
  comparison is approximate.

## What this shows

- With only its output head in float32, the float16 network meets the owner's
  many-seed rule on these seeds at about the same speed as plain float16. It also
  removes 9 of float16's 10 early stops on full traces. The change from milestone 11
  is consistent with the argmax-tie hypothesis there, but does not prove it.
- Because those 10 early stops are among the seeds the candidate was chosen on, the
  pass is not yet evidence on fresh data.

## Limits

The same 300 seeds of one volume as the float16 verdict, V100 only, with the
selection overlap described above. The float32 reference is reused from an earlier
job. A pass here does not cover other data or hardware.
