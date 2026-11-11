# SegNeuron on the B200, crops 41-50: results of the timing addendum

Design: docs/unseen_b200_segneuron_design.md (sha256 `3211f066a15563cb...`). Evidence:
docs/unseen_b200_evidence.json, written by examples/judge_unseen_b200.py (unchanged since its check) from the
result files of MIT job 28228697 in the data cache (`runs/unseen-b200-segneuron-20261106`) and the test's
V100 SegNeuron run (`runs/unseen-20261104/v100`, files pinned by docs/unseen_evidence.json). Evidence test:
tests/test_unseen_b200_evidence.py. This page is written by examples/write_unseen_b200_results.py, and the test
requires it to be exactly that script's output.

This addendum was planned after the unseen-crop test's results were known (docs/unseen_results.md);
its rule and measures were fixed before SegNeuron ran on the B200. It times SegNeuron; it does not
re-judge accuracy, and the test's accuracy claims (from the V100 run) are unchanged.

## Runs

- Smoke run on crop 11 (job 28225144_11, 20 objects scored): completed. It checked only that the job runs and is not a result.
- 10 crops (job 28228697, one B200 per crop, 8 CPUs): every crop finished; failures: 0; non-finite SegNeuron outputs: none on any crop.
- Every crop's records passed the judge's checks: NVIDIA B200, 590.48.01 (GPU and driver), the same staged source for all crops with every file of the test's code unchanged, the inference versions numpy==2.5.4, tifffile==2026.10.15, torch==2.7.1+cu128, the design's sha256, the test's checkpoint, threshold, crop files and objects.
- node5700-c1: crops 41, 44, 46, 48, 50.
- node5800-c1: crops 42, 43, 45, 47, 49.

## Crop time against each B200 run of the test

SegNeuron's crop time end to end (primary) is the wall time of `inference.py` plus post-processing, as on
the V100; without start-up it is `inference.py`'s own loop time plus post-processing. The FFN runs'
crop times are the test's (docs/unseen_evidence.json). Break-even: objects per crop at which the two cost
the same, per crop (median and range) and from the totals over all crops.

End to end (primary):

| run | statement | crops cheaper | break-even per crop, median (range) | break-even from totals |
|---|---|---|---|---|
| F5 | SegNeuron was cheaper on 7 of 10 crops | 7 of 10 | 398 (222-541) | 389 |
| F1 | SegNeuron was cheaper than float32 FFN one object at a time on every test crop | 10 of 10 | 78 (63-112) | 80 |
| S10 | SegNeuron was cheaper on 2 of 10 crops | 2 of 10 | 583 (417-871) | 587 |
| P2 | SegNeuron was cheaper on 2 of 10 crops | 2 of 10 | 661 (434-1048) | 674 |
| Q2 | P4 was cheaper than SegNeuron on every test crop | 0 of 10 | 1081 (777-1583) | 1075 |

Without start-up:

| run | statement | crops cheaper | break-even per crop, median (range) | break-even from totals |
|---|---|---|---|---|
| F5 | SegNeuron was cheaper on 7 of 10 crops | 7 of 10 | 382 (192-531) | 375 |
| F1 | SegNeuron was cheaper than float32 FFN one object at a time on every test crop | 10 of 10 | 75 (54-110) | 78 |
| S10 | SegNeuron was cheaper on 3 of 10 crops | 3 of 10 | 565 (385-855) | 567 |
| P2 | SegNeuron was cheaper on 2 of 10 crops | 2 of 10 | 648 (426-1028) | 651 |
| Q2 | P4 was cheaper than SegNeuron on every test crop | 0 of 10 | 1047 (758-1552) | 1038 |

Crops on which SegNeuron was cheaper, end to end:

- batched fast FFN (F5): 41, 42, 44, 46, 48, 49, 50.
- float32 FFN one object at a time (F1): 41, 42, 43, 44, 45, 46, 47, 48, 49, 50.
- the student (S10): 46, 48.
- P8 (P2): 46, 48.
- P4 (Q2): none.

Totals over all crops, end to end: SegNeuron 2,281 s; F1 12,984 s; F5 2,688 s; S10 1,780 s; P2 1,551 s; Q2 971 s.

## Per crop

| crop | objects | node | SegNeuron B200, end to end (s) | inference.py wall | loop | post-processing | F5 crop time | break-even vs F5 |
|---|---|---|---|---|---|---|---|---|
| 41 | 401 | node5700-c1 | 215 | 82 | 52 | 134 | 389 | 222 |
| 42 | 532 | node5800-c1 | 254 | 56 | 51 | 198 | 296 | 455 |
| 43 | 451 | node5800-c1 | 256 | 55 | 51 | 201 | 214 | 541 |
| 44 | 433 | node5700-c1 | 194 | 56 | 47 | 138 | 226 | 372 |
| 45 | 464 | node5800-c1 | 286 | 58 | 53 | 228 | 249 | 532 |
| 46 | 446 | node5700-c1 | 182 | 51 | 47 | 131 | 261 | 310 |
| 47 | 415 | node5800-c1 | 270 | 58 | 53 | 212 | 251 | 448 |
| 48 | 496 | node5700-c1 | 184 | 51 | 47 | 132 | 274 | 332 |
| 49 | 443 | node5800-c1 | 256 | 59 | 50 | 197 | 268 | 424 |
| 50 | 498 | node5700-c1 | 185 | 51 | 47 | 134 | 261 | 352 |

## SegNeuron on the B200 against the V100 (totals over all crops)

| | end to end | inference.py wall | loop | post-processing |
|---|---|---|---|---|
| B200 | 2,281 s | 577 s | 499 s | 1,705 s |
| V100 (the test's run) | 5,351 s | 2,345 s | 2,136 s | 3,006 s |

Per crop:

| crop | loop, B200 | loop, V100 | post-processing, B200 | post-processing, V100 |
|---|---|---|---|---|
| 41 | 52 s | 217 s | 134 s | 307 s |
| 42 | 51 s | 210 s | 198 s | 295 s |
| 43 | 51 s | 213 s | 201 s | 297 s |
| 44 | 47 s | 213 s | 138 s | 303 s |
| 45 | 53 s | 216 s | 228 s | 306 s |
| 46 | 47 s | 215 s | 131 s | 303 s |
| 47 | 53 s | 212 s | 212 s | 298 s |
| 48 | 47 s | 215 s | 132 s | 299 s |
| 49 | 50 s | 212 s | 197 s | 301 s |
| 50 | 47 s | 212 s | 134 s | 296 s |

Post-processing runs on the host CPUs of each cluster, so the post-processing rows compare CPUs, not GPUs.

## Agreement with the V100 run

Per-object F1, B200 minus V100, over all 4,579 objects: mean +0.000033, largest absolute 0.198, 2 objects differing by more than 0.1. Mean F1 per crop:

| crop | B200 | V100 | objects differing by > 0.1 |
|---|---|---|---|
| 41 | 0.8010 | 0.8010 | 0 |
| 42 | 0.8325 | 0.8325 | 0 |
| 43 | 0.8440 | 0.8440 | 0 |
| 44 | 0.8316 | 0.8316 | 0 |
| 45 | 0.8108 | 0.8108 | 0 |
| 46 | 0.8364 | 0.8364 | 0 |
| 47 | 0.8197 | 0.8197 | 0 |
| 48 | 0.8421 | 0.8423 | 1 |
| 49 | 0.8251 | 0.8249 | 0 |
| 50 | 0.8273 | 0.8269 | 1 |

The test's accuracy claims stand as they were; the agreement only shows that the B200 run is the same method.

## Disclosures

- **Post-processing time differed between the 2 nodes.** node5700-c1: 131-138 s; node5800-c1: 197-228 s (the GPU loop: 47-53 s on every crop). Against batched fast FFN, SegNeuron was cheaper on 5 of 5 crops on node5700-c1; 2 of 5 crops on node5800-c1. Each crop ran once on one node, so node and crop cannot be separated here. On the V100 the same crops' post-processing took 295-307 s, which is consistent with a node effect but does not show how these crops would run on the other B200 node. Whether the count of 7 depends on node assignment was not measured.
- Crops 41-50 had been read by every method of the test before this addendum ran.
- Planned after the test's results, including the V100 times and the B200 FFN times, were known; one run
  per method and crop; FFN's B200 times come from the test's own jobs, SegNeuron's from these later jobs,
  possibly on other nodes.
- Neither method is optimised for the B200: `inference.py` runs one 20 x 128 x 128 patch at a time and
  post-processing is single-process CPU code. SegNeuron's end-to-end time includes process start-up,
  model loading and reading and writing its affinities (1,500,000,128 bytes per crop) on MIT
  pool storage; FFN's crop time leaves out loading the network and follows a warm-up.
- For the batched runs (F5, S10, P2, Q2) crop time divided by objects is the average cost at the crop's
  own object count (401-532), so a break-even far from that range is an extrapolation.
- The B200 run's inference environment has torch==2.7.1+cu128; the V100 run's has torch==2.7.1+cu126 (the same version, another CUDA build).
