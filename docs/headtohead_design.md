# FFN against fine-tuned SegNeuron, object by object on crop 12: design

Owner decisions on 2026-10-29, after the fine-tuning result (`docs/segneuron_finetune_results.md`):
- compare FFN with the fine-tuned SegNeuron one on one, on accuracy and on inference speed;
- use crop 12 only, and leave the test crops 13-40 unread;
- use FFN in `fp16_head32` with the fast tracer only. This is our fastest FFN; it is bitwise
  identical to the reference path in that number format (`docs/fasttrace_results.md`) and was
  accepted under the owner's 2% rule (`docs/tolerance2_results.md`).

The question: on the same objects and the same GPU, how accurate is each method, and how long
does each take?

## Objects and scoring

- **Objects:** the 457 large objects of crop 12 (`vitseg.tuning_objects`). Each object has one
  seed at `benchmark.seed_point`, which lies inside FFN's field-of-view margin. These are the
  objects and seeds that scored the ViT, zero-shot SegNeuron and fine-tuned SegNeuron.
- **FFN:** the authors' network and settings, traced from each object's seed on the 500^3 crop.
  Each seed is traced on its own, as in every earlier FFN run here. The mask is
  `benchmark.object_mask` of the canvas, scored with `benchmark.score`.
- **SegNeuron:** the fine-tuned checkpoint (`train/best.ckpt` of run `segneuron-ft-20261028`,
  sha256 `e726bb43...`) runs through the authors' `inference.py`. Our pipeline then runs at the
  threshold 0.6 chosen on crop 12 in the fine-tuning run; the threshold is fixed and not chosen
  again. Each object is scored by `vitseg.seed_records` (the segment that contains its seed),
  which gives the same numbers as `benchmark.score` on that segment's mask, plus the accounting
  of `segneuron.object_accounting`.
- **Reported:**
  - the per-object mean recall, precision and F1 for each method;
  - paired per-object F1: the number of objects where each method wins, and the median difference;
  - the mean and the quantiles (0, 10, 25, 50, 75, 90, 100%) of the per-object F1 difference;
  - the mean F1 of both methods without the objects whose seed SegNeuron puts in predicted
    background. Such an object scores F1 0 even if the neuron is segmented one voxel away;
  - merges for both methods (at least 1,000 voxels of another object, as in `benchmark.score`)
    and the merged voxels;
  - the overlap of FFN's independent traces: voxels that lie in two or more of the 457 masks.

## Speed (one job, one V100)

Both methods run in the same Slurm job on the same node, FFN first.

| | timed | not timed (reported separately) |
|---|---|---|
| FFN | per seed: `fast.trace` (the tracer's own clock, which excludes per-trace setup) and the wall time of copying the image to the GPU, tracing, and taking the mask | loading the model and compiling, and one untimed warm-up trace (the first object) that compiles the tracer's GPU functions; scoring |
| SegNeuron | the wall time of the authors' `inference.py` process, which includes loading the model and data. Its own `inference_seconds` is reported as well. Then our pipeline on the CPU cores of the same job: the layout and watershed, agglomeration, and the segmentation at 0.6, each timed separately; the threshold is not searched | scoring |

- FFN's cost grows with the number of objects traced. SegNeuron's cost is fixed per volume and
  labels every voxel of the crop, not only the 457 objects. Both are reported:
  - the total time for the 457 objects;
  - the time per object (for SegNeuron, the crop's total divided by 457).
- The break-even number of objects is reported two ways:
  - **end to end:** SegNeuron's total divided by FFN's mean wall time per object;
  - **GPU only:** `inference_seconds` divided by FFN's mean trace-clock time.
- FFN's fast tracer copies the 500^3 image to the GPU for every seed. Its wall time includes
  that copy and its trace clock does not, so both are reported. Keeping the image on the GPU
  between seeds would be a change to FFN's tracer and is not made here.
- The CPU post-processing is part of SegNeuron's cost, because the method has no output without
  it. It runs single-process on the job's CPUs.

## Checks built into the run

- FFN reproduces the recorded crop-12 trace. Object 346093 was traced in `fp16_head32` in run
  `manyseed-fp16_head32-20261022`; its canvas sha256, history sha256 and step count must match.
- SegNeuron reproduces the recorded fine-tuning result. The new GPU inference may differ in the
  last bits (cuDNN). The per-object F1 at threshold 0.6 is compared with
  `segneuron_ft_crop12.json`, and the largest difference is reported, with the largest change in
  the affinities and the foreground against the recorded outputs. One flipped merge at a hard
  threshold can change one object's F1 a lot even when the outputs barely differ.

## Disclosed in advance

- Crop 12 chose SegNeuron's checkpoint and threshold, so its accuracy here is optimistic. FFN's
  settings are the authors' own, not tuned here. FFN's network was trained by the volume's
  authors on proofread data from this volume, possibly including crop 12.
- FFN's traces are independent of one another and may overlap. SegNeuron must assign each voxel
  to one segment. For a single object, this favours FFN.
- The 457 objects are the large ones (benchmark size rules). SegNeuron's run also produces
  segments for the smaller and border objects in the crop, which FFN would need a trace each for.
  Only the 457 are scored, so how well the others are segmented is not known. Dividing
  SegNeuron's crop time by 457 is the cost per scored object; it would overstate the cost per
  wanted object only if those other objects were also wanted and usefully segmented. The
  break-even count compares against the 457 only.
- SegNeuron's post-processing runs on CPU in Python; FFN's per-step bookkeeping runs on the GPU
  (`docs/fasttrace_results.md`). The timings compare these implementations as they are.
- FFN has no "lost to background / other segments" split: a single trace has no other segments.
- Precision ignores label-0 voxels (`benchmark.score`), for both methods.
- The timings are for one V100 and one CPU allocation, in one job. Other hardware will differ.
  No repeat was made to measure the spread.

## Code

- `liconn_v2/headtohead.py`: pairing and summaries (numpy only, unit-tested).
- `examples/run_headtohead.py`: modes `ffn`, `segneuron` and `compare`.
- `cluster/wahab/headtohead.sbatch`: one GPU job that runs all three modes.
- `tests/test_headtohead.py`; afterwards results, evidence and an evidence test.
