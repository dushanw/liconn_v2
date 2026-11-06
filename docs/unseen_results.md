# Unseen crops 41-50: results (runs 2026-11-04/05)

Design: docs/unseen_design.md (owner-approved 2026-11-04, unchanged since the crops were fetched;
its sha256 is in docs/unseen_crops.json and every result file). Evidence: docs/unseen_evidence.json
(checked by tests/test_unseen_evidence.py, which re-runs the judge on the result files). Judge:
examples/judge_unseen.py. Every question and measure the design states is reported below, as its
rules give it; nothing was re-judged under other rules.

## Outcome

- **No method failed on any crop.** All 50 B200 runs (MIT job 27976430) and all 20 V100 jobs
  (Wahab jobs 7557287, FFN, and 7557288, SegNeuron) finished, every object of every crop was
  traced (4,579 objects in all). None of the runs that count non-finite outputs had one: the B200
  float16 runs F5, S10, P2 and Q2 (network outputs) and SegNeuron (affinities and foreground).
  The float32 run F1 and run V (float16 on the V100) do not count them, so for those two the only
  failure check was that every run finished, as the design states.
- **Question 1: FFN's fast mode passes the 2% rule against float32.** F5 against F1 over all 4,579
  objects: 51 new merges and 76 new early stops, each against a limit of 91.58 (2% of N); mean
  recall -0.0007 and mean precision +0.0010, against a limit of -0.01 each. It also avoided 54
  merges and extended 65 traces. The pass is pooled over large objects; dendrites, where all five
  fresh-seed merges were (docs/fresh_results.md), had 1 new merge among 73 classified dendrites.
- **Question 2: FFN is more accurate than the fine-tuned SegNeuron on every crop.** Mean per-crop
  F1 difference (F5 minus SegNeuron) +0.0580, 95% t-interval [0.0509, 0.0651], FFN higher on 10 of
  10 crops. Run V (the crop-12 protocol on the V100) gives +0.0576 [0.0505, 0.0646]; the two claims
  agree. SegNeuron is the faster method at the size of these crops: its break-even count is 181
  objects per crop end to end and 89 GPU only, and every crop has 401-532 objects.
- **Question 3: the student, P8 and P4 are each "worse than FFN"** under the margin of 0.02 F1:
  student -0.0280 [-0.0348, -0.0212], P8 -0.0260 [-0.0321, -0.0198], P4 -0.0489 [-0.0579,
  -0.0399], each lower than FFN on 10 of 10 crops. P8 and the student cannot be told apart
  (+0.0020 [-0.0017, 0.0057]; P8 higher on 5 crops, the student on 5). They are 7.3x (student),
  8.4x (P8) and 13.4x (P4) faster than float32 FFN on the B200, against 4.8x for FFN's fast mode.

## Question 1: fast mode against float32 (F5 - F1, objects pooled)

Rule (design, fixed before the crops were read): with d = F5 minus F1 per object, a new merge is
d(precision) < -0.10, a new early stop d(recall) < -0.10; each count at most 0.02 N; mean
d(recall) and mean d(precision) at least -0.01; all crops complete.

| measure | value | limit | passes |
|---|---|---|---|
| objects (N) | 4,579 | all ten crops complete | yes |
| new merges | 51 | 91.58 | yes |
| new early stops | 76 | 91.58 | yes |
| mean recall change | -0.0007 | -0.01 | yes |
| mean precision change | +0.0010 | -0.01 | yes |

Avoided merges 54, extensions 65. Among the 125 objects on which F1 already merged (precision
below 0.2), 1 had a new merge.

By class (reported, not part of the rule; classes label only some objects):

| class | objects | new merges | new early stops | avoided merges | extensions |
|---|---|---|---|---|---|
| dendrites | 73 | 1 | 0 | 1 | 0 |
| axons | 216 | 0 | 7 | 0 | 3 |
| unclassified | 4,290 | 50 | 69 | 53 | 62 |

Per crop:

| crop | objects | new merges | new early stops | avoided merges | extensions |
|---|---|---|---|---|---|
| 41 | 401 | 7 | 5 | 3 | 3 |
| 42 | 532 | 2 | 6 | 8 | 7 |
| 43 | 451 | 6 | 11 | 8 | 9 |
| 44 | 433 | 6 | 3 | 5 | 4 |
| 45 | 464 | 6 | 10 | 3 | 9 |
| 46 | 446 | 2 | 4 | 2 | 8 |
| 47 | 415 | 4 | 14 | 5 | 9 |
| 48 | 496 | 3 | 4 | 4 | 3 |
| 49 | 443 | 8 | 9 | 10 | 6 |
| 50 | 498 | 7 | 10 | 6 | 7 |

This confirms the B200 fast mode (32 objects at once, stream x 1/16), not the V100 fast tracer that
the 300-seed tests judged. Axons, a small share of the objects, had 7 early stops among 216
(3.2%) and a mean recall change of -0.0071; the rule does not apply per class.

## Questions 2 and 3: per-crop mean F1

Unweighted mean per-object F1 per crop; the crop is the unit (n = 10). Intervals are 95% t (9
degrees of freedom), per comparison, not adjusted for the five comparisons; the bootstrap over
crops (10,000 resamples, seed 20261104) is a check and decides nothing.

| method | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| FFN float32 (F1) | 0.877 | 0.880 | 0.906 | 0.877 | 0.869 | 0.902 | 0.878 | 0.907 | 0.870 | 0.878 |
| FFN fast (F5) | 0.878 | 0.882 | 0.907 | 0.877 | 0.863 | 0.903 | 0.881 | 0.908 | 0.874 | 0.877 |
| FFN on the V100 (V) | 0.876 | 0.880 | 0.907 | 0.878 | 0.868 | 0.903 | 0.873 | 0.907 | 0.872 | 0.880 |
| student (S10) | 0.850 | 0.854 | 0.868 | 0.864 | 0.853 | 0.877 | 0.845 | 0.875 | 0.837 | 0.849 |
| P8 (P2) | 0.846 | 0.862 | 0.878 | 0.862 | 0.852 | 0.875 | 0.849 | 0.871 | 0.840 | 0.856 |
| P4 (Q2) | 0.819 | 0.847 | 0.842 | 0.845 | 0.825 | 0.844 | 0.825 | 0.843 | 0.836 | 0.835 |
| SegNeuron | 0.801 | 0.832 | 0.844 | 0.832 | 0.811 | 0.836 | 0.820 | 0.842 | 0.825 | 0.827 |

| comparison | mean difference | t-interval | bootstrap | crops higher (first / second) | claim |
|---|---|---|---|---|---|
| F5 - SegNeuron | +0.0580 | [0.0509, 0.0651] | [0.0525, 0.0640] | 10 / 0 | FFN more accurate |
| V - SegNeuron | +0.0576 | [0.0505, 0.0646] | [0.0521, 0.0633] | 10 / 0 | FFN more accurate |
| student - F5 | -0.0280 | [-0.0348, -0.0212] | [-0.0331, -0.0220] | 0 / 10 | worse than FFN |
| P8 - F5 | -0.0260 | [-0.0321, -0.0198] | [-0.0308, -0.0208] | 0 / 10 | worse than FFN |
| P4 - F5 | -0.0489 | [-0.0579, -0.0399] | [-0.0563, -0.0416] | 0 / 10 | worse than FFN |
| P8 - student | +0.0020 | [-0.0017, 0.0057] | [-0.0009, 0.0051] | 5 / 5 | no difference claimed |

The paired per-crop differences d_k behind them:

| comparison | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| F5 - SegNeuron | +0.0768 | +0.0497 | +0.0626 | +0.0458 | +0.0526 | +0.0661 | +0.0617 | +0.0654 | +0.0489 | +0.0505 |
| V - SegNeuron | +0.0754 | +0.0477 | +0.0634 | +0.0465 | +0.0574 | +0.0668 | +0.0536 | +0.0651 | +0.0468 | +0.0529 |
| student - F5 | -0.0283 | -0.0282 | -0.0386 | -0.0139 | -0.0100 | -0.0260 | -0.0361 | -0.0329 | -0.0371 | -0.0286 |
| P8 - F5 | -0.0321 | -0.0198 | -0.0285 | -0.0152 | -0.0113 | -0.0280 | -0.0327 | -0.0366 | -0.0338 | -0.0216 |
| P4 - F5 | -0.0588 | -0.0356 | -0.0643 | -0.0327 | -0.0380 | -0.0585 | -0.0565 | -0.0643 | -0.0383 | -0.0421 |
| P8 - student | -0.0038 | +0.0083 | +0.0101 | -0.0012 | -0.0013 | -0.0020 | +0.0034 | -0.0038 | +0.0033 | +0.0070 |

"Worse than FFN" is the design's label for an interval below 0 whose lower end is at or below
-0.02. The student's and P8's upper ends (-0.0212, -0.0198) lie close to the margin; neither
interval lies wholly above -0.02, so neither is "worse by less than 0.02".

FFN against SegNeuron object by object (4,579 objects): FFN higher on 3,429 (F5) and 3,426 (V).
SegNeuron's seed fell in its predicted background for 29 objects; without them, mean F1 is 0.886
(F5) and 0.886 (V) against SegNeuron's 0.833. V and F5 agree on average (mean F1 difference
-0.0004) but differ by more than 0.1 on 182 objects (largest 0.886): they are two implementations
(the JAX reference port with the fast tracer, one object at a time, against the torch port with 32
at once), so they do not trace identically. FFN's independent traces (run V) cover 520.5 M voxels,
7.3% of them claimed by two or more traces (at most 9 on one voxel); SegNeuron's segmentation
claims each voxel once.

## Pooled over objects, merges and runaways

| method | recall | precision | F1 | merges (partners >= 1,000 voxels) | objects with a merge | merged voxels |
|---|---|---|---|---|---|---|
| F1 | 0.869 | 0.953 | 0.885 | 2,564 | 631 | 146.5 M |
| F5 | 0.868 | 0.954 | 0.885 | 2,514 | 627 | 156.9 M |
| V | 0.869 | 0.952 | 0.885 | 2,539 | 628 | 146.8 M |
| S10 | 0.852 | 0.933 | 0.857 | 11,012 | 911 | 309.3 M |
| P2 | 0.839 | 0.947 | 0.859 | 6,393 | 731 | 195.5 M |
| Q2 | 0.806 | 0.942 | 0.837 | 4,011 | 873 | 122.3 M |
| SegNeuron | 0.795 | 0.935 | 0.828 | 2,886 | 816 | 107.7 M |

Merges are summed over traces, which may overlap.

Runaway traces (at least 3x the reference's steps on the same object; descriptive). The
reference stopped at its first step on 2 objects, 206992 (crop 42) and 820108 (crop 49), the same
two for F1 and F5; 3x of one step is not a meaningful bound. Both are counted as runaways of S10,
P2 and Q2 against either reference (so 2 of each count below), and neither for F5 against F1.

| method | against F1 | against F5 |
|---|---|---|
| F5 | 38 | - |
| S10 | 171 | 171 |
| P2 | 131 | 127 |
| Q2 | 112 | 113 |

## Speed

B200 (MIT), one run per method per crop, each run its own job, TF32 off. Crop time is the traces
plus the student's encoder or FFN's image transfer; loading and object selection are outside it.

| run | total over 10 crops | mean per crop | per object | vs F1 |
|---|---|---|---|---|
| F1 | 12,984 s | 1,298 s | 2.84 s | 1.0x |
| F5 | 2,688 s | 269 s | 0.587 s | 4.8x |
| S10 | 1,780 s | 178 s | 0.389 s | 7.3x |
| P2 | 1,551 s | 155 s | 0.339 s | 8.4x |
| Q2 | 971 s | 97 s | 0.212 s | 13.4x |

Crop time per crop (seconds):

| run | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 1,379 | 1,558 | 1,148 | 1,146 | 1,183 | 1,240 | 1,215 | 1,410 | 1,386 | 1,319 |
| F5 | 389 | 296 | 214 | 226 | 249 | 261 | 251 | 274 | 268 | 261 |
| S10 | 194 | 195 | 167 | 146 | 152 | 194 | 177 | 197 | 202 | 156 |
| P2 | 165 | 136 | 127 | 112 | 127 | 186 | 156 | 189 | 200 | 153 |
| Q2 | 69 | 85 | 85 | 74 | 104 | 102 | 109 | 117 | 131 | 94 |

V100 (Wahab), FFN (run V) against SegNeuron, as in Fig. 4. Over the ten crops FFN's wall time was
13,531 s (traces 10,957 s); SegNeuron's 5,351 s end to end (inference 2,136 s). Break-even object
count (objects per crop above which SegNeuron is faster):

| way | per crop: median (range) | from the ten crops' totals |
|---|---|---|
| end to end (primary) | 179 (157-206) | 181 |
| GPU only | 88 (76-102) | 89 |

## Caveats and disclosures

- One run per method; no training repeats. Speed is one run per method per crop, and runs of a crop
  ran on different B200 nodes (four in all) and different V100 nodes (seven in all, every one a
  Tesla V100-SXM2-16GB, driver 580.178.04; docs/unseen_evidence.json lists the node of each job).
- The crops are not fully independent: 41 and 49, 45 and 49, and 42 and 48 abut, and 418 object IDs
  were scored in more than one crop (a neuron can be large in two crops). Question 1's pooled counts
  carry no correction for this or for objects within a crop.
- Crops 41-50 are the last ten positions of the fixed crop order, not a fresh random sample of the
  volume; their order was set by the fixed random permutation, not by choice. Four of the ten lie on
  the outermost grid columns or row used (x0 = 100 or 3100, or y0 = 4083), against 16 of 28 among
  crops 13-40; their labelled fraction (0.51-0.59) matches crops 13-40.
- FFN's network was trained by the volume's authors, possibly on these regions; the smaller FFNs
  were trained on crops 0-10 with FFN as teacher; SegNeuron was fine-tuned on crops 0-11. All
  crops are from one volume at one depth.
- The 2% rule was set after the fresh-seed result on crops 13-40; it was fixed before crops 41-50
  were read, so this is its first pre-registered test.
- Compute was 16.7 B200 GPU-hours (the summed elapsed time of the 50 MIT jobs, from Slurm's
  accounting), not the 6 the design estimated: each B200 job spent about 11 minutes choosing objects
  on the CPU. The owner approved about 15 GPU-hours before submission. The
  choice of objects is outside every timer.
- The code that ran is identified by one list of 166 file sha256s, the same for all 50 MIT tasks
  and the Wahab run (on Wahab, synced into the repository clone, so its git commit is not the
  code's). Smoke runs on crop 11 (MIT 27934261, Wahab 7556576 and 7556577) passed before the test.
- CHECKED status of this write-up is evidence that the numbers match the result files, not a
  review of the science.
