# FFN against fine-tuned SegNeuron, object by object on crop 12: results

Design: `docs/headtohead_design.md`. Evidence: `docs/headtohead_evidence.json`, checked by
`tests/test_headtohead_evidence.py` against the run files in the data cache
(`runs/headtohead-20261029/`).

## Summary

- **FFN is more accurate on these 457 objects.** Mean F1 is 0.891 for FFN (recall 0.859,
  precision 0.970) and 0.823 for fine-tuned SegNeuron (0.801 / 0.926). FFN scores higher on 340
  objects and SegNeuron on 117. The median per-object difference is small (+0.015 for FFN), but
  the tail is long: at the 90th percentile FFN is ahead by 0.31. 319 objects reach F1 0.9 with
  FFN and 249 with SegNeuron. Without the 3 objects whose seed SegNeuron puts in background, the
  means are 0.890 and 0.829.
- **SegNeuron merges more.** SegNeuron's segments hold 249 pieces of other objects (at least
  1,000 voxels each) on 80 of the objects, 15.3 M voxels in those pieces. FFN's masks hold 93
  such pieces on 65 objects, 4.0 M voxels.
- **SegNeuron is about 2.2 times faster for these 457 objects, and its output covers the whole
  crop.**
  On one V100, FFN traced the 457 objects in 1,154 s of wall time (2.5 s per object), plus 79 s
  of one-time setup. SegNeuron took 534 s for the whole crop: 251 s for the authors' inference
  process (214 s of network time) and 283 s of CPU post-processing (272 s of it watershed).
  The same run also assigns segments to the rest of the crop, where 1,084 more labelled objects
  lie that FFN did not trace. How well those are segmented was not scored.
- **Break-even:** at FFN's mean time per object, FFN's total equals SegNeuron's whole crop at
  about 212 objects end to end. Counting FFN's one-time 79 s setup, it is about 180 objects. On
  network (GPU) time alone it is about 107 objects. These are averages: for a particular set of
  objects the crossing point depends on how many steps those objects take.
- **Both runs reproduced their recorded results exactly.** FFN's crop-12 trace of object 346093
  matched the recorded run in steps, canvas and history. SegNeuron's outputs and per-object
  scores were identical to the fine-tuning run (largest change 0).
- **Crop 12 favours both methods**: it chose SegNeuron's checkpoint and threshold, and FFN's
  network may have been trained on this volume. These are not test results, and the test crops
  13-40 were not read.

## What was run

| step | job | output (data cache `runs/headtohead-20261029/`) |
|---|---|---|
| FFN `fp16_head32` fast tracer on the 457 seeds; the authors' `inference.py` with the fine-tuned checkpoint; our pipeline at threshold 0.6; scoring and comparison | Slurm 7483667, `timed-gpu`, one Tesla V100-SXM2-16GB and 8 CPUs on node d3-w4140b-03, 1 h 44 min | `ffn_crop12.json`, `segneuron_crop12.json`, `headtohead_crop12.json`, `infer/inference.json`, `7483667.out`, `gpu.txt`, `env-infer.txt` |

- The job ran from the staged source `/scratch/dwadduwa/liconn_v2_headtohead`. Its manifest
  (`source_sha256.txt`) matches the repository's files that the run imports (checked by the
  evidence test).
- The job checked the fine-tuned checkpoint's sha256 and SegNeuron's pinned commit before any
  work.
- The job used 1 h 44 min of its 2 h limit. Most of the time not shown below was untimed
  scoring: `benchmark.score` per FFN seed and the object accounting.

## Accuracy (457 objects, the same seeds)

| | FFN `fp16_head32` | SegNeuron fine-tuned |
|---|---|---|
| mean F1 (recall / precision) | **0.891** (0.859 / 0.970) | 0.823 (0.801 / 0.926) |
| objects with the higher F1 | 340 | 117 |
| objects with F1 >= 0.9 | 319 | 249 |
| mean F1 without SegNeuron's 3 seeds in background | 0.890 | 0.829 |
| merges >= 1,000 voxels (objects with one; voxels in those pieces) | 93 (65; 4.0 M) | 249 (80; 15.3 M) |
| all voxels of other objects in the segment, any size | 4.1 M | 15.4 M |

Per-object F1 difference (FFN minus SegNeuron):

| min | 10% | 25% | median | 75% | 90% | max | mean |
|---|---|---|---|---|---|---|---|
| -0.775 | -0.054 | -0.001 | +0.015 | +0.076 | +0.308 | +0.991 | +0.067 |

The methods are close on most objects. The difference comes from a minority where SegNeuron loses
much more (splits or merges) than FFN.

FFN's traces are independent of one another. 1.21 M voxels lie in two or more of the 457 masks
(at most 4). That is 2.8% of the 42.7 M voxels the masks cover. SegNeuron cannot assign a voxel
twice. How much this overlap raises FFN's scores was not measured.

## Speed (one V100, one job)

| | FFN, 457 objects | SegNeuron, whole crop |
|---|---|---|
| network / tracing | 911 s (tracer clock; 137,723 steps at 6.61 ms) | 214 s (`inference_seconds`) |
| end to end | 1,154 s (per-seed wall: image copy, trace, mask) | 534 s (inference process 251 s + CPU post 283 s) |
| per object | 2.53 s mean, 2.28 s median | 1.17 s (crop total / 457) |
| one-time setup, not counted above | 79 s (model load, compile, one warm-up trace) | inside the inference process (251 s vs 214 s) |
| break-even, objects (at FFN's mean per object) | 212 end to end; 180 with FFN's setup; 107 GPU only | |

SegNeuron's CPU post-processing was 272 s watershed (with the layout), 10 s agglomeration and
1 s for the segmentation at 0.6. It ran single-process on the job's CPUs. It is over half of
SegNeuron's time and the clearest place to speed SegNeuron up; this was not tried. On the FFN
side, about 0.53 s per seed of the wall time (21%) is outside the tracer's clock, mostly copying
the 500^3 image to the GPU for every seed. Keeping the image on the GPU would cut that, but it is
a change to FFN's tracer and was not made.

## Reading

- **Accuracy:** FFN is ahead on crop 12 by 0.07 mean F1. Its precision is higher, and so is its
  recall. SegNeuron's extra merges and splits are concentrated in a minority of objects.
- **Speed:** on average SegNeuron is cheaper once more than about 180-210 large objects are wanted
  from a crop like this one, and one run covers the whole crop. FFN is cheaper for a few
  targeted neurons; for a particular set, the crossing point depends on those objects' steps.
- **Bias:** neither number is a test result. Both lean optimistic on crop 12, and not by the
  same amount: crop 12 chose SegNeuron's checkpoint and threshold, while FFN's settings are the
  authors' own. The test crops would give the fair accuracy comparison. The speed comparison does
  not depend on the crop as much: FFN's time grows with the steps per object, SegNeuron's with
  the volume.

## Disclosed

- FFN's network was trained by the volume's authors on proofread data from this volume,
  possibly including crop 12.
- FFN's traces may overlap; SegNeuron is a partition (above).
- Precision ignores label-0 voxels for both. FFN's masks are 15% label 0 on average (unlabelled
  neurites and background), which precision does not count.
- SegNeuron's per-object time is its crop time divided by the 457 scored objects. The run
  also produces segments for the rest of the crop (1,541 labelled objects in all), so if those
  other objects were wanted and segmented usefully, its cost per wanted object would be lower.
  Only the 457 were scored; even among them, 3 seeds fell in predicted background.
- One job, one V100, 8 CPU cores, no repeats; other hardware will differ. SegNeuron's
  post-processing is single-process Python and FFN's bookkeeping runs on the GPU; the timings
  compare these implementations as they are.
- FFN has no "lost to background / other segments" split, because a single trace has no other
  segments.
