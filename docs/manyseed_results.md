# Many-seed verdict for float16: results (runs 2026-10-21)

**Outcome: float16 fails the owner-confirmed many-seed rule (docs/manyseed_design.md)
because of early stops, not merges.** It makes 10 new early stops in 300 fresh seeds
(limit 3). New merges are at the limit (3 of 300), and the mean recall and precision
changes are small.

This is an implementer-run record of owner-approved runs on Wahab V100s:
- float32 reference: Slurm 7389883;
- float16: Slurm 7389884;
- one task per crop (13 crops, 300 seeds).

A first submission (7389879, 7389880) was cancelled before it wrote any result, so
that the runs would use the code as corrected after the independent method check.
Its logs are kept on Wahab as `cancelled-manyseed-*`.

tests/test_manyseed_evidence.py checks the following against
docs/manyseed_evidence.json:
- the code that ran is the checked code;
- the outputs are the recorded ones;
- every seed ran in the right format on the right GPU;
- the verdict below follows from the outputs.

Two edits came after the runs:
- After review, verdict() in liconn_v2/manyseed.py (and its caller,
  examples/judge_manyseed.py) also checks every record against the registered seed
  list and its crop.
- docs/manyseed_seeds.json was rewritten one seed per line, with identical content,
  so that the review packet fits the workflow's size limit.

The GPU runs call none of the changed functions. The exact files that ran are kept
with the runs. The evidence test checks that only the declared definitions changed,
that the seed list's content is unchanged, and recomputes the verdict with the
current code.

## Crops and seeds

- The first 10 crops of the pre-registered order gave 252 seeds. By owner decision,
  3 further crops in the same order were added, the last one cut, to reach exactly
  300 seeds (docs/manyseed_crops.json, docs/manyseed_seeds.json).
- 13 crops in all, 7.6 GB stored, within the approved 11 GB.
- 72 of the 300 seeds are dendrites (24%), because few dendrites meet the size rules.

## Verdict (float16 against float32)

| rule | result | |
|---|---|---|
| new merges (precision falls by more than 10 points): at most 3 | 3 | pass |
| new early stops (recall falls by more than 10 points): at most 3 | **10** | **fail** |
| mean recall falls by at most 1 point | -0.11 points | pass |
| mean precision falls by at most 1 point | -0.31 points | pass |
| complete, pinned inputs, all 300 seeds | yes | pass |

Reported for information:
- Avoided merges: 2. Extensions (recall rises by more than 10 points): 4. Neither
  offsets new events.
- One-sided 95% upper bounds, assuming independent seeds (optimistic, since seeds
  within a crop are not independent): new merge rate 2.6%; new early stop rate 5.6%;
  dendrite new merge rate 6.4% (1 of 72).
- Dendrites: 1 new merge and 3 new early stops; mean recall -0.74 points and mean
  precision +0.61 points.
- Most seeds barely change: the median absolute change below 10 points is 0.5 recall
  points and 0.1 precision points (95th percentiles 2.8 and 0.8 points).
- One seed's reference trace already merged (m2_axon_178794, precision below 20%);
  no candidate trace was empty.
- Speed: 19.58 ms per step in float32 and 8.88 ms in float16 (2.2x). Total steps:
  93,863 and 95,927 (+2.2%).

Seeds with events (changes in percentage points):

| seed | recall | precision | event |
|---|---|---|---|
| m1_axon_175567 | -55.5 | 0.0 | early stop |
| m1_axon_173408 | -12.2 | -0.4 | early stop |
| m2_dendrite_104095 | -10.5 | 0.0 | early stop |
| m2_axon_92043 | -11.4 | +0.2 | early stop |
| m3_dendrite_142095 | -15.1 | +0.2 | early stop |
| m5_dendrite_226719 | -19.8 | +0.1 | early stop |
| m5_axon_232100 | -19.5 | 0.0 | early stop |
| m6_axon_216332 | -19.8 | +1.6 | early stop |
| m7_axon_320743 | -15.3 | +0.8 | early stop |
| m10_axon_145819 | -29.0 | -2.2 | early stop |
| m6_axon_175989 | +29.0 | -79.4 | merge (and extension) |
| m9_dendrite_231945 | -1.6 | -27.8 | merge |
| m9_axon_148932 | 0.0 | -83.3 | merge |
| m7_dendrite_374312 | +0.3 | +73.6 | avoided merge |
| m8_axon_78192 | +0.4 | +24.7 | avoided merge |
| m4_axon_59400 | +17.5 | -0.1 | extension |
| m5_axon_228843 | +32.3 | 0.0 | extension |
| m6_axon_172905 | +62.2 | +0.8 | extension |

## What this shows

- On 300 fresh seeds, float16 mostly changes where traces stop. Early stops outnumber
  extensions 10 to 4, and 7 of the 10 early stops are axons. Merges are rarer: 3 new
  and 2 avoided.
- Early stops were invisible to the 24 development seeds (none there). They appear
  only with many seeds.
- Why float16 stops traces early is not established by these runs. A plausible
  hypothesis, untested: decisions near FFN's 0.6 move threshold flip at the faces
  where the trace would continue. Investigating this is the owner's chosen next step.

## Limits

13 crops of one volume (ExPID82_1), z 111-611, V100 only. The confidence bounds treat
seeds as independent, which they are not within a crop. A new merge is seen only
through precision, so a further merge on an already-merged seed, or a small merge onto
a very large object, could go uncounted (docs/manyseed_design.md).
