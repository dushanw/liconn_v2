# Float32 FFN traced many objects at once: results

Design: [fp32_batch_design.md](fp32_batch_design.md). Evidence: [fp32_batch_evidence.json](fp32_batch_evidence.json),
written by `examples/judge_fp32_batch.py` from the result files in the data cache
(`runs/fp32batch-20261109`) and checked by [test_fp32_batch_evidence](../tests/test_fp32_batch_evidence.py).
This study was added after the test on crops 41-50 and has no pass mark.

## Why

In the test, float32 FFN traced one object at a time, while fast FFN (`fp16_head32`, residual stream
x 1/16) traced 32 at once. The 4.8x between them therefore mixed two changes: batching and reduced
precision. This study batches float32 FFN as well, so the two can be separated.

## Crop 11: eight settings

FFN (the torch port) traced the 512 crop-11 objects in float32 and in fast FFN's format, 1, 8, 32 and 64
objects at once. Each setting ran in its own process. The first seven ran in job 28611579 on node5601;
H16x64 ran in job 28643365 on node5801, because node5601 was then reserved.

| Setting | Crop seconds | ms per step | Steps | Mean F1 | Merges | Traces identical to width 1 |
|---|---:|---:|---:|---:|---:|---|
| Float32 x1 (F32x1) | 1,186.7 | 7.50 | 158,257 | 0.898 | 112 | (same traces as the timing study's F1) |
| Float32 x8 (F32x8) | 799.5 | 5.05 | 158,257 | 0.898 | 112 | 512 of 512 |
| Float32 x32 (F32x32) | 771.4 | 4.87 | 158,257 | 0.898 | 112 | 512 of 512 |
| Float32 x64 (F32x64) | 762.1 | 4.82 | 158,257 | 0.898 | 112 | 512 of 512 |
| Fast FFN x1 (H16x1) | 576.0 | 3.62 | 159,191 | 0.902 | 106 | |
| Fast FFN x8 (H16x8) | 242.3 | 1.52 | 159,191 | 0.902 | 106 | 512 of 512 |
| Fast FFN x32 (H16x32) | 230.0 | 1.44 | 159,191 | 0.902 | 106 | 512 of 512 (same traces as the timing study's F5) |
| Fast FFN x64 (H16x64) | 227.9 | 1.43 | 159,191 | 0.902 | 106 | 512 of 512 |

Batching never changed a trace: at every width, in both formats, all 512 traces had the same steps,
mask and step history as one object at a time. The float32 run fingerprint equals the timing study's
F1 (`4e60f463...`) and the fast-FFN one its F5 (`cb4e7072...`). Fast FFN had no non-finite outputs.

**The split.** Fast FFN x32 traced crop 11 5.16x faster than float32 one at a time. The two orders give:

| Order | First change | Second change |
|---|---|---|
| Batching first | float32 x1 -> x32: 1.54x | float32 x32 -> fast FFN x32: 3.35x |
| Precision first | float32 x1 -> fast FFN x1: 2.06x | fast FFN x1 -> x32: 2.51x |

Per step the figures are almost the same (5.19x in all; 1.54x then 3.38x, or 2.07x then 2.51x). The two
changes interact: batching helps fast FFN (2.51x) much more than float32 (1.54x). A likely reason, not
measured here, is that a fast-FFN step at width 1 spends much of its time on the host, while a float32
step is dominated by the network. Either way, batching float32 alone gives about 1.5x, and most of the
gain comes from precision.

Width 8 gives most of the batching gain in both formats (float32 1.48x, fast FFN 2.38x); widths 32 and
64 add 4% to 6% more. Width 64 was the fastest measured in float32 (762.1 s against 771.4 s at width 32).
Width 32 was fixed in the design before these runs, as the width of every other tracer on the test crops.

## Test crops 41-50: float32, 32 at once

Float32 FFN traced the 4,579 test objects 32 at once (F32x32), one B200 per crop, and was compared with
the test's own F1 (float32, one at a time) and F5 (fast FFN, 32 at once) results.

| Tracer | Total seconds | Seconds per object |
|---|---:|---:|
| Float32, one at a time (F1) | 12,984 | 2.84 |
| Float32, 32 at once (F32x32) | 8,739 | 1.91 |
| Fast FFN, 32 at once (F5) | 2,688 | 0.59 |

All 4,579 F32x32 traces had the same steps, mask and history as F1's, so its Table 1 row is F1's apart
from time: pooled F1 0.885, recall 0.869, precision 0.953; 631 objects with a merge, 146.5 million merged
voxels; no new merges, no new early stops and no runaways against F1.

Batching float32 gave 1.49x (1.38x to 1.52x per crop) and fast FFN's format at the same width a further 3.25x (2.58x to 3.60x), together the test's 4.83x. SegNeuron was cheaper than F32x32
on all ten crops; its break-even was 117 objects per crop (median; 86 to 169) end to end and 113
(74 to 165) without its start-up. From the ten-crop totals, the break-even is 120 objects per crop end
to end and 115 without start-up. These break-evens extend F32x32's time per object, measured on crops
of 401-532 objects, to far fewer objects; no run measured batched throughput at such small counts.

Per crop (seconds; ms per step and speed-ups for F32x32):

| Crop | F32x32 node | F1 | F32x32 | F5 | F32x32 ms per step | Batching (F1 / F32x32) | Precision (F32x32 / F5) | F32x32 mean F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 41 | node5501 | 1,379.2 | 1,002.5 | 388.7 | 5.46 | 1.38 | 2.58 | 0.877 |
| 42 | node5501 | 1,558.2 | 1,027.7 | 296.2 | 4.96 | 1.52 | 3.47 | 0.880 |
| 43 | node5800 | 1,147.6 | 768.9 | 213.6 | 5.02 | 1.49 | 3.60 | 0.906 |
| 44 | node5800 | 1,145.7 | 756.7 | 225.7 | 4.95 | 1.51 | 3.35 | 0.877 |
| 45 | node5701 | 1,182.8 | 786.5 | 249.2 | 4.99 | 1.50 | 3.16 | 0.869 |
| 46 | node5601 | 1,240.1 | 816.3 | 261.1 | 5.00 | 1.52 | 3.13 | 0.902 |
| 47 | node5701 | 1,215.4 | 830.5 | 250.7 | 5.12 | 1.46 | 3.31 | 0.878 |
| 48 | node5701 | 1,409.7 | 954.3 | 274.2 | 5.08 | 1.48 | 3.48 | 0.907 |
| 49 | node5701 | 1,386.4 | 915.2 | 267.7 | 4.99 | 1.51 | 3.42 | 0.870 |
| 50 | node5701 | 1,319.2 | 880.2 | 261.4 | 5.06 | 1.50 | 3.37 | 0.878 |


## How the crop-11 numbers were obtained, and what was replaced

The design ran the eight settings in one process (job 28561105, node5501). That job ran out of GPU
memory at H16x32: canvases of finished traces were not freed between settings. Its fast-FFN x1 time was
also much slower (5.05 ms per step, against 3.62 above). On the same 20 smoke-test objects, fast FFN
one at a time took 3.83 ms per step on node5601 (`widths-smoke20`) and 4.92 ms in a fresh process on
node5501 (`widths-each-smoke20`), while node5501 was shared with other jobs. This suggests a node effect
rather than the memory problem, but single runs on two nodes, at different times and with host load not
measured, cannot settle the cause. The first job's float32
rows were within 3% of those above.

The owner approved rerunning all eight settings, each in its own process, on one node other than
node5501; that run is the table above. The first job's numbers are kept in the evidence (`first_job`).
H16x32 was also run again on node5801 next to H16x64: 1.45 ms per step against 1.44 on node5601, with
the same traces (`other_node`).

## Limits

- One run per setting, on shared B200 nodes. A record of the nodes' load failed (the compute nodes
  refused the login); only the cores booked by other jobs were seen (112 to 130 on node5601).
- H16x64 comes from a different node than the other seven settings, so every comparison that uses it
  is across nodes. On node5801 alone, H16x32 -> H16x64 is 1.02x.
- The test-crop rows come from separate jobs on several nodes. Crops 41 and 42 ran on node5501. Crop
  41 had the slowest F32x32 step (5.46 ms, against 4.95 to 5.12 on the other crops) and the lowest
  batching gain (1.38x), so it may be affected by that node's load.
- Crop 11 is a tuning crop. F32x32 on the test crops was added after the test; its traces equal F1's,
  so it adds timing only.
