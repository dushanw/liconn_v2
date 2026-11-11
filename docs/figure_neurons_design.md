# Whole test-crop neurons as reconstructed by every method (figure design)

Owner request, 2026-11-10: the manuscript's whole-neuron figure should show every method compared on
the test crops (Fig. 6, Table 1), on neurons drawn at random from those crops, replacing the crop-12
figure with FFN, ViT and SegNeuron. The owner approved about 1-1.5 hours of one B200. This is a figure,
not a measurement: no timing is reported and no test result changes.

## Draw (fixed before any reconstruction is looked at)

The 4,579 test objects are listed crop by crop (41 to 50), each crop in the test's object order (the
`object_ids` of its F1 result file). Three positions are drawn from that list without replacement with
`numpy.random.default_rng(20261111).choice(4579, 3, replace=False)`, in the order drawn. The draw uses
the list only, not any score. If a drawn object cannot be shown for a method (a verification failure
below), it is shown as such; it is not replaced.

## Methods

The five B200 tracers of the test and SegNeuron, exactly as run in the test:

| method | test run | how it is re-run |
|---|---|---|
| FFN, float32, one at a time | F1 | the drawn objects only, one at a time (each trace has its own canvas) |
| FFN, `fp16_head32-s16`, 32 at once | F5 | the whole crop, 32 at once, as in the test |
| factorised student | S10 | the whole crop, 32 at once, with the crop's encoder features |
| P8 | P2 | the whole crop, 32 at once |
| P4 | Q2 | the whole crop, 32 at once |
| fine-tuned SegNeuron | B200 run (unseen-b200-segneuron-20261106) | post-processing only, from the saved network outputs |

Float32 FFN 32 at once is not shown: its traces equal F1's on every test object
(docs/fp32_batch_results.md). Each method runs in its own process (`--methods`; the job script's default), as each test run did; `figure.json` is written after every method, so a later failure keeps the verified panes. The first job ran all methods in one process (now the opt-in `ONE=1`) and crop 43 ran out of B200 memory at S10, so crop 43 was re-run one method per process. Batched methods re-run the whole crop because a trace could in principle
depend on which objects share its batch. Networks, checkpoints, number formats, stream scales,
loaders and checks are those of `examples/run_unseen.py`; SegNeuron's post-processing is that of
`examples/run_unseen_wahab.py` (threshold 0.6).

## Verification

- Tracers: each drawn object's steps, `mask_sha256` and `history_sha256` must equal the test's record
  for that method (`b200/crop-K/NAME.json`).
- SegNeuron: each drawn object's recall, precision and F1 must equal the values in the B200 run's
  `segneuron.json`. Table 1 shows SegNeuron's V100 accuracy; the B200 run agreed with it to within
  0.0004 in per-crop mean F1, and the figure states which run it shows.

A mismatch is reported and that pane is left empty with the reason; it is not drawn from an unverified
reconstruction.

## What is drawn

For each object and method, every voxel of the reconstruction or of the labelled object gets one part:
found (in both), missed (labelled object, not reconstructed), other labelled (reconstructed, in another
labelled object) and unlabelled (reconstructed, label 0). The label pane shows the labelled object.
Parts are exported in the bounding box of their union as uint8 volumes, and are rendered locally as
surfaces (marching cubes on the voxels, then Taubin smoothing; a Gaussian blur first cut thin neurites)
with the colours and viewpoint of the crop-12 figure, unlabelled voxels translucent. Each pane's caption gives the method's recorded F1, recall, precision
and merges of at least 1,000 voxels for that object.

## Run (2026-11-10 to 11)

The draw gave crop 50 object 68079, crop 45 object 353342 and crop 43 object 343390. MIT job 28663157
ran crops 45 and 50 with all methods in one process; crop 43 ran out of B200 memory at S10 (F1 and F5 had
finished). Job 28668556 re-ran crop 43 one method per process; its work finished after about 1 h 42 min,
though Slurm recorded a TIMEOUT at 5 h 10 min. Each crop's `figure.json` lists no verification
failures and a pane for every method. The three `figure.json` files are copied byte for byte into
`docs/figure_neurons_evidence/` (`crop-K.json`), with the final line of each job log and each job's source
checksum list in `logs.txt`; `tests/test_figure_neurons_evidence.py` checks them against this section. The
part volumes, full logs and source lists stay in `runs/figneurons-20261110` on MIT and in the local data
cache. The `figure.json` files as used for Fig. 5 (sha256):

| crop | object | `figure.json` sha256 | failures | recorded F1: F1, F5, S10, P2, Q2, SegNeuron |
|---|---|---|---|---|
| 43 | 343390 | `6244fd84b5df0c7f47be8901c611f6ba44364fc26cc15f0f707b69bd38b1ac0c` | none | 0.995, 0.994, 0.997, 0.996, 0.995, 0.978 |
| 45 | 353342 | `65699a9530cf63be85286ed924ab93b82d740a1208dd090987df3d3c01111b9e` | none | 0.887, 0.869, 0.852, 0.520, 0.561, 0.885 |
| 50 | 68079 | `93fa91ecc76d95e901c1ecf6344ec17612f3716baec9f789bd39a5fec6c80c5e` | none | 0.811, 0.808, 0.499, 0.348, 0.376, 0.335 |

The F1 values are the test's own per-object records, copied into `figure.json` only after the
re-run matched them; the last line of each job log repeats the crop's panes and failures.

## Limits

Three neurons cannot represent 4,579; the per-method statistics are in Table 1 and Fig. 6. Voxels are
drawn as cubes in the crop grid, without anisotropic scaling.
