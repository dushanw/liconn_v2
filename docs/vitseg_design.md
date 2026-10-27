# Single-pass ViT segmentation pilot (milestone 18, design)

> **Outcome (2026-10-26): stopped at the tuning crop by owner decision.** Training run 2 stopped
> itself at step 67,344; the step-66,000 checkpoint reaches mean F1 0.531 over crop 12's 457
> tuning objects, the crop that also chose its checkpoint and threshold. (FFN's many-seed mean is
> about 0.89, but over 300 seeds in crops 0-12, so the two are context, not a matched
> comparison.) No ViT was run or scored on the test crops 13-40. Results, breakdown and how to
> resume: `docs/vitseg_results.md`.

Owner decision (2026-10-24): try a ViT-like model that segments a whole block in one
pass, as an alternative to FFN's iterative tracing:
- dense output for all objects at once;
- trained on the proofread labels;
- judged against the proofread ground truth, side by side with FFN on the same seeds;
- about 2-4 V100-days.

Revision 2 (2026-10-24, before any training) follows the Gemini method check (task
20261024-100302-2eb37c6e); see "Changes in revision 2" at the end.

Revision 3 (2026-10-24, task 20261024-214940-32b9d3db, which calls it "revision 2" of
the pilot) follows training run 1, which diverged, and the owner's decisions on
background; see "Revision 3" at the end. Where it differs from the sections above, it
replaces them.

This is a different method, not a faster FFN. The FFN-agreement rules
(docs/manyseed_design.md, docs/tolerance2_results.md) do not apply. The pass rule
below was confirmed by the owner before any training.

## Why

With `fp16_head32` and the fast tracer, FFN spends about 2.5 s per object on a V100.
On the fresh seeds that is 761 s for 300 objects. A timing probe (exploratory, Slurm
7433377; data/runs/probe-vit-20261024) measured a UNETR-style ViT at 42 ms per 128^3
block in float16, about 20 ms per megavoxel. One 500^3 crop would then cost a few
seconds of network time for every object in it. Whether the CPU post-processing and
the accuracy allow this is what the pilot measures.

## Data (already held locally and on Wahab; no new data)

500^3 crops of ExPID82_1 at 18 x 18 x 24 nm: raw uint8 and the proofread labels
231030_agg_240123.

| role | crops | use |
|---|---|---|
| training | many-seed crops 0-11 (docs/manyseed_crops.json) | gradient updates |
| tuning | many-seed crop 12 | post-processing thresholds only |
| test | fresh crops 13-40 (docs/fresh_crops.json), 300 registered seeds (docs/fresh_seeds.json) | the verdict |
| never used | crop A (benchmark), crop B (held-out) | |

The image is normalised as FFN's input: (raw - 128) / 33.

**Leakage, disclosed rather than removed:**
- Crops lie on one grid, so a neuron can cross from a training crop into a test crop.
  The results report the verdict quantities separately for test seeds whose proofread
  object also appears in a training crop or crop 12. Leakage through neighbouring
  objects is not measured. The pre-registered verdict uses all 300 seeds.
- The authors' FFN (Model 3) was trained on proofread data from this volume. It may
  have seen the test crops, which would favour FFN.

## Model

A hybrid UNETR/TransUNet-style network (implemented in JAX/Flax, liconn_v2/vitseg.py).

- **Input:** one 128^3 block (z, y, x), 1 channel.
- **Convolutional encoder,** three levels:
  - 16 channels at full resolution;
  - 32 channels at 1/2 resolution (stride-2 convolution);
  - 64 channels at 1/4 resolution.

  Each level has two 3^3 convolutions with ReLU, and each level's output is kept as a
  skip connection.
- **Tokens:** the 1/4 level is cut into 2^3 patches, which is 8^3 voxels per token:
  4,096 tokens of width 384, plus a learned position embedding.
- **ViT:** 8 transformer blocks with pre-norm, global self-attention (6 heads) and an
  MLP (4x). Every token attends to the whole block: this is the single pass.
- **Decoder:** transposed convolutions upsample 1/8 -> 1/4 -> 1/2 -> full. At each level
  it concatenates the encoder skip, then applies two 3^3 convolutions (64, 32, 16
  channels).
- **Output:** a 1^3 output convolution to 6 channels, the affinities below (logits).

Float16 compute with float32 parameters. The skips give the decoder full- and
half-resolution context for thin neurites, which the 8^3 tokens alone cannot resolve.

**Affinities.** For each voxel v and offset o in (z, y, x):
- offsets: (1,0,0), (0,1,0), (0,0,1), (3,0,0), (0,4,0), (0,0,4);
- target 1 if v and v - o have the same nonzero label;
- target 0 if their labels differ and at least one is nonzero;
- masked when both are unlabelled (label 0), or when v - o is outside the block.

The three long-range channels are auxiliary training targets: they encourage context.
Segmentation uses only the three nearest-neighbour channels.

Labelled-unlabelled pairs count as boundaries. Proofread objects are complete
segments, so an unlabelled neighbour is not part of that object; this assumption is
stated in the limits.

## Training (fixed before training)

- Random 128^3 blocks from crops 0-11, batch 2.
- Augmentation:
  - flips in x, y and z;
  - x-y transposition (the voxels are anisotropic in z, so z is never transposed);
  - intensity scale x[0.9, 1.1] and shift +[-0.1, 0.1] after normalisation.
- Loss: binary cross-entropy on the unmasked affinities. Within each channel and batch,
  positive and negative targets are weighted to equal total weight.
- Optimiser: AdamW, peak learning rate 3e-4, weight decay 0.05, linear warm-up over
  1,000 steps, then cosine decay to zero.
- Static loss scaling (x1,024). A step with a non-finite gradient is skipped and
  counted; if more than 1% of steps are skipped, the run is reported as unstable.
- Steps: a 15-minute timing job measures the step time. The step count is set so
  training takes about 36 hours on one V100 (at most 1.5 V100-days), and recorded
  before the training job starts.
- Checkpoint selection: every 2,000 steps, the loss is measured on 8 fixed,
  unaugmented 128^3 blocks of crop 12, which is never trained on. The checkpoint with
  the lowest such loss is the model that is evaluated. The last checkpoint is also
  kept and reported.
- Random seed 0. Training is one Slurm job on Wahab's `gpu` partition (14-day limit).

## Inference and segmentation (CPU post-processing, deterministic)

1. **Tiling.** 128^3 blocks with stride 96. Each block contributes its central 96^3,
   plus the outer margin at the crop's faces. Every voxel is predicted exactly once;
   the last block in each axis is shifted inward to end at the face. Affinities =
   sigmoid(logits).
2. **Fragments.**
   - Boundary map: b(v) = 1 - the mean affinity of the up to six nearest-neighbour
     edges incident to v, in both directions.
   - Foreground: b < 0.5, the decision point of the class-balanced affinities.
   - Markers: connected local maxima (within a 5^3 neighbourhood) of the foreground's
     anisotropic distance transform. Every foreground blob, however thin, has at least
     one marker.
   - Watershed of b from the markers (scikit-image) labels every voxel. The result is
     deliberately over-segmented.
3. **Agglomeration** (mean-affinity, as in waterz):
   - Each pair of touching fragments has an edge. Its score is the mean
     nearest-neighbour affinity over the voxel pairs across their boundary in which
     both voxels are foreground.
   - Contacts through flooded background do not count, so fragments that touch only
     through background never merge.
   - Repeatedly merge the highest-scoring pair while its score is at least t; edges
     are re-scored by pooling the sums and counts.
4. **Threshold.** t is chosen on crop 12 only, from {0.05, 0.10, ..., 0.95}.
   - Objective: the mean F1 (benchmark.score precision and recall) over crop 12's
     objects that pass the test seeds' size rule (benchmark.candidates: at least 20,000
     voxels inside the 16-voxel margin and at most 350,000 in the crop), of any class.
     Each is scored from its own seed point (benchmark.seed_point).
   - Ties go to the higher t, which merges less.
   - The foreground level (0.5) and peak neighbourhood (5) are fixed, not tuned.

**Oracle check on crop 12** (before training; exploratory; scripts and outputs in
data/runs/vitseg-oracle-20261024).
- Setup: the post-processing was run with oracle affinities, i.e. the training targets,
  with 0.5 where both voxels are unlabelled. Scored over the 457 tuning objects.
- Edges over all boundary pairs (revision 2): mean recall 0.812, precision 0.998 at
  t = 0.55. Below t = 0.5, contacts through background merged everything (precision
  0.001).
- Both-foreground edges (adopted): recall 0.874, precision 0.991 at t = 0.85, and
  nearly insensitive to t (precision 0.97-0.99 over t = 0.05-0.95).
- Ceiling: the seed's 6-connected piece of each object holds 0.887 of the object on
  average, because objects leave and re-enter the crop. No 6-connected method can
  exceed this. FFN exceeded it on 10 of 37 seeds checked (crops 13-15), because it can
  bridge small gaps. That is an advantage for FFN in this comparison.
- CPU: watershed 74 s, edges 5 s, agglomeration 2 s per crop; peak memory 35 GB.

The step count, 170,000, was fixed from the timing job before training: 0.742 s per
step on a V100 over steps 101-400 (Slurm 7433735; one step of 400 skipped), about 35
hours.

## Evaluation and pass rule (owner-confirmed 2026-10-24, before any training)

**Scoring.** For each of the 300 fresh seeds, the ViT object is the segment containing
the seed voxel in that crop's segmentation. It is scored by benchmark.score against the
proofread labels, exactly as FFN's traces were.

**Reference.** The recorded `fp16_head32` FFN runs on the same seeds:
data/runs/fresh-fp16_head32-20261023, the runs judged under the 2% rule.

**Per-seed events** (manyseed.events, flip = more than 10 points):
- new merge: precision falls by more than 10 points against FFN;
- avoided merge: precision rises by more than 10 points;
- new early stop: recall falls by more than 10 points;
- extension: recall rises by more than 10 points.

Net new merges = new merges - avoided merges. Net new early stops = new early stops -
extensions.

**Verdict:**
- **COMPETITIVE:** mean recall and mean precision each at most 2 points below FFN's,
  net new merges at most 6, and net new early stops at most 6 (2% of 300).
- **PROMISING:** mean recall and mean precision each at most 5 points below FFN's, net
  new merges at most 15, and net new early stops at most 15 (5%). This means a larger
  run is worth it.
- **FAIL:** otherwise.

Also reported, not gated:
- gross event counts;
- the same quantities against float32 FFN (fresh-fp32-20261023);
- per class (axon, dendrite);
- the leakage subset;
- the mean unlabelled fraction of each method's segments. benchmark.score's precision
  ignores unlabelled voxels for both methods, so bleeding into unlabelled space is
  shown here, not penalised;
- the dense quality on crop 12 at every threshold.

**Speed (reported, not gated).** The headline, fixed before any test crop is segmented,
is the ViT's whole-crop time against FFN's estimated time to trace every object in the
crop:
- ViT: network on the GPU, plus CPU watershed and agglomeration, over all 28 test crops.
- FFN: (number of objects passing the seed size rule) x (mean fast-tracer time per
  registered seed).

Both segment every such object. The seed-only comparison is secondary. It favours the
ViT, which segments everything while FFN traces only the registered seeds.

**Order of the runs.** Crop 12 (checkpoint, threshold, crop-12 quality and timing) is
run and inspected first, without reading any test crop. Only then are the test crops
segmented, once.

Per test crop:
- GPU time of the network, including copies;
- CPU time of watershed and agglomeration (8 cores requested; the code is
  single-threaded unless stated);
- the total.

These are compared with:
- the fast tracer's `fp16_head32` time for the same crop's registered seeds
  (fasttrace-fresh-fp16_head32-20261024);
- an estimate, clearly labelled, of FFN's time to trace every proofread object in the
  crop: the number of objects with at least 20,000 voxels times the mean time per
  traced seed.

The ViT segments every object; FFN traces one per call.

## Verification

- **Unit tests (CPU, synthetic):**
  - affinity targets and masks on small label volumes;
  - the balanced masked loss;
  - tiling covers every voxel exactly once and reassembles a known field exactly;
  - watershed and agglomeration recover known synthetic objects and merge them at the
    expected threshold;
  - the model's output shape, and the training step decreasing the loss on a tiny
    problem;
  - the scoring and event wiring against hand-computed cases;
  - the verdict tiers at their boundaries.
- **Controller example** (examples/check_vitseg.py --cpu-smoke): a tiny model (32^3
  blocks, 4^3 patches) trained for a few steps on a synthetic volume, then the full
  segmentation and scoring path, writing outputs/vitseg_check.json. It checks the
  path, not accuracy.
- **GPU runs:** owner-approved implementer runs on Wahab (timing, training, test
  inference). Recorded with source hashes, job IDs and output sha256, and checked by
  an evidence test, as before.

## Limits

- One volume, one pilot-sized model, one training run, no hyperparameter search.
- Labelled-unlabelled pairs are treated as boundaries.
- Only about half of each crop is labelled, so the model never learns from unlabelled
  regions; its behaviour there is untested.
- 8^3 patches are 144 x 144 x 192 nm, coarser than thin axons; fine boundaries depend
  on the convolutional decoder.
- V100 only. Post-processing speed depends on the CPU and on a pure-Python
  agglomeration that could be made much faster.
- A pass means competitive on these seeds, not a replacement for the authors'
  pipeline.

## Changes in revision 2 (before training; Gemini method check)

Adopted, after checking each against the code:
- **Boundary map:** it now uses edges in both directions. Revision 1 used only the
  three backward edges, so one face of every neurite looked like boundary.
- **Markers:** they now come from distance-transform peaks inside b < 0.5. The fixed
  "b < 0.1, at least 50 voxels" rule could leave thin neurites without a marker,
  particularly under class-balanced calibration, and unmasked watershed would then flood
  them into a neighbour.
- **Model:** convolutional skips at full, 1/2 and 1/4 resolution. With 8^3 patches, the
  revision-1 decoder could not resolve thin axons.
- **Tuning population:** the test seeds' size rule, and a finer threshold grid (0.05
  instead of 0.1).
- **Checkpoint selection** on crop 12 instead of the last checkpoint, against
  overfitting.
- **Leakage flag:** it now includes crop 12.
- **Reporting:** the unlabelled fractions are reported.

Not adopted:
- the owner-confirmed pass rule, including net flips and the fp16_head32 reference
  (owner decision);
- the patch-phase concern at the last tile: training uses random block corners, so the
  model sees every phase;
- quantile instead of mean agglomeration: mean affinity is a standard choice, and it
  is kept for simplicity.

Added after the oracle check (still before training): edge scores count only voxel
pairs that are both foreground. On the tuning crop with oracle affinities, this raised
recall from 0.812 to 0.874, close to the 0.887 connectivity ceiling. It also removed
the collapse at low thresholds.


## Grok method check on revision 2 (2026-10-24, training already started)

Adopted before any test crop is segmented:
- Crop 12 is run and inspected first, with the trained affinities: the threshold sweep,
  quality and timing.
- The speed headline is pre-registered: whole-crop time against FFN's all-object
  estimate.
- The voxel size is taken from liconn_v2/reference.py.

Disclosed rather than changed:
- The gated reference stays `fp16_head32`, and flips stay net, as the owner decided.
  Float32 and gross counts are reported alongside.
- FFN can bridge gaps that a 6-connected segmentation cannot (see the oracle check).
- Crop 12 is used for both checkpoint selection and the threshold.
- A pass could be carried by leaked objects. The leakage subset is reported.


## Revision 3: background as an output, and training stability (2026-10-24)

### What happened in run 1

Slurm 7433740 (staged source `/scratch/dwadduwa/liconn_v2_m18b`) trained normally to
step 22,000: validation loss 0.381 at step 2,000, 0.304 at step 22,000. At step 22,777
the loss became non-finite and every later step was skipped: 25,211 steps until the
job was cancelled at step 48,000, about 5 GPU-hours with no learning. The skipped-step
limit was only checked at the end of the run. The job used about 10 GPU-hours in all.

Diagnosis (Slurm 7446325; script, log and JSON in
`data/runs/vitseg-diag-nan-20261024/`), on the 8 crop-12 validation blocks:

| parameters | float16 loss | float32 loss | largest activation |
|---|---|---|---|
| step 22,000 (best checkpoint) | 0.296-0.342 | same | 1,830 (first decoder up-convolution) |
| frozen after step 22,777 | non-finite | 0.344-0.426 | 51,000 in the patch embedding and every block's output; 132,000-183,000 in the decoder |

The weights stayed finite (largest 0.98). The token stream, which no normalisation
bounded, grew to about 51,000. The first decoder up-convolution then exceeded the
float16 limit of 65,504, and the loss turned non-finite on every batch. A non-finite
step leaves the weights unchanged, so the run could not recover.

### Owner decisions (2026-10-24)

The owner pointed out that label 0 is background, the dark space between neurons, not
unproofread neuron. Measured on crop 12: label-0 voxels are dark (mean intensity 92,
against 154 for labelled ones), and thin: 97% lie within 100 nm of a labelled voxel, and
none is more than 300 nm from one. Of all boundary edges in crop 12, 98.9% join a neuron
to background, and 1.1% join two neurons directly.

Approved:
1. **A 7th output channel:** the probability that a voxel belongs to a neuron (label
   > 0). It is trained on every voxel with the same class-balanced cross-entropy (label 0
   is a real answer), and averaged with the six affinity channels in the loss.
2. **Segmentation outputs background as 0:**
   - neuron voxels have probability >= 0.5 (NEURON); all others are background;
   - the boundary map reads only edges whose two voxels are both neuron voxels, so it
     marks only boundaries between touching neurons (background is the neuron channel's
     job). It is 1 minus the *lowest* such edge at a voxel (see the method check below);
   - confident voxels are neuron voxels with b < 0.5. Markers are distance-transform
     peaks of the confident voxels; a connected piece of neuron voxels with no marker
     becomes one marker itself;
   - the watershed is restricted to the neuron voxels;
   - contact scores use the edges whose two voxels are both neuron voxels.
3. **Judging:** the pass rule against FFN is unchanged. A seed in predicted background
   is scored like an empty mask (recall 0, precision 0), as benchmark.score scores one.
   Under the unchanged event rule it counts as both a new early stop and, when FFN's
   precision is above 0.1, a new merge. This is conservative against the ViT. The number
   of such seeds is reported. Voxel-level background precision and recall against label
   0 are reported per crop, as is each seed's background fraction next to FFN's 9-23%.
   None of these is gated.

### Stability changes (implementer, from the diagnosis)

- LayerNorm on the token stream after the patch embedding and before the decoder.
- Gradient clipping at global norm 1.0 before AdamW.
- The trainer stops itself after 200 consecutive non-finite steps or a non-finite
  validation loss. It keeps the best checkpoint so far and records the reason.
- A non-finite validation loss never becomes the selected checkpoint.
- After the method check: a LayerNorm over channels after every convolution and
  up-convolution (the patch embedding had reached 51,000 before any normalisation),
  and the token LayerNorm before the position embedding is added, not after.

Everything else is unchanged: architecture sizes, schedule, steps (170,000), batches,
crop split, checkpoint selection, thresholds and the pass rule. Run 2 starts from
scratch.

### Cursor method check (gemini, 2026-10-24) and what changed

Adopted, each checked against the code:
- **Touching neurons:** with a mean over edges, a voxel on a direct neuron-neuron face
  has b = 1/6, counts as confident, and both neurons end up in one fragment, which no
  merge can undo (2 of 3 large neurons in crop 12 touch another somewhere). The boundary
  map now uses the lowest edge. A test with two touching slabs now separates them.
- **Affinity loss:** affinities are trained only on pairs of two neuron voxels (same
  neuron 1, different neurons 0). Before, neuron-background pairs, 98.9% of crop 12's
  boundary pairs, took nearly all of the negative weight, and segmentation no longer
  reads those edges. This refines the owner-approved design, in which the six affinity
  outputs were to stay unchanged; it is reported to the owner.
- **Stability:** normalisation after every convolution, the position embedding after the
  token LayerNorm, and no crash if a run stops before any finite validation loss; a
  stopped run is not resumed.
- **Reporting:** FFN's background fraction per seed; mean background fractions over
  non-empty segments only.

Not adopted, with reasons:
- Requiring confident voxels for contact edges, once b uses the lowest edge: every low
  crossing edge would remove its own voxels from the count and inflate contact scores.
  Contacts count all edges between two neuron voxels, as the owner approved ("both
  foreground").
- Scoring a seed in predicted background differently: the pass rule stays as approved;
  the count of such seeds is reported.
- Shifting the 0.5 neuron threshold for the class prior: thresholds are fixed before
  training.

### Owner decisions after the method check (2026-10-25)

The first check of task 20261024-214940-32b9d3db (Codex) found that the code departed
from that contract's wording in two places. The owner decided, before training run 2:
- contact scores count every edge between two neuron voxels (not only confident ones);
- a connected piece of neuron voxels with no confident voxel becomes its own fragment
  (one marker), rather than background;
- affinities are trained only on pairs of two neuron voxels.

The contract was amended accordingly (task 20261025-173459-37286541, superseding 20261024-214940-32b9d3db).

### Checks before training run 2

Oracle on crop 12 (457 tuning objects; script and output in
`data/runs/vitseg-oracle-r3-20261024/`), the training targets as outputs:

| outputs | threshold | recall | precision | F1 | background precision / recall |
|---|---|---|---|---|---|
| perfect | 0.80 | 0.887 | 0.997 | 0.924 | 1.000 / 1.000 |
| smooth noise, SD 0.2 | 0.75 | 0.828 | 0.929 | 0.840 | 0.991 / 0.994 |

With perfect outputs, recall reaches the connectivity ceiling (0.887). Revision 2
reached 0.874 / 0.991 with perfect outputs, and 0.711 / 0.826 with the same kind of
noise (counting all pairs). Post-processing took about 37 s (perfect) and 88 s (noisy)
of CPU time, with a 33 GB peak.

Timing (Slurm 7446371, staged source `/scratch/dwadduwa/liconn_v2_m18c`): 0.770 s per
step over steps 101-400, no skipped steps; 170,000 steps take about 36 h.

### Disclosed

- This is the second training run of the pilot. Run 1 produced no usable model and is
  not evaluated.
- The segmentation now differs from the oracle check above, which used six affinities.
  The ceiling argument still holds: a neuron in several pieces inside the crop cannot be
  joined.
- A pretrained start (for example SegNeuron, a convolutional EM model, MIT licence) is
  a separate later task, judged on crop 12 only.

