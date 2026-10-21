# FFN speed benchmark: design (milestone 2)

Purpose: measure how much network compute FFN inference spends per correctly segmented
voxel, and how accurate it is against the authors' proofread segmentation. This lets later
method changes (not more hardware) be compared fairly. This milestone builds the
benchmark and records the baseline. It implements no speedup.

Owner decisions (2026-10-18): judge accuracy against the authors' proofread
segmentation; no training for now; a tolerance replaces the bitwise match for speed
variants. The exact tolerance is still to be confirmed (proposal below).

## Ground truth

- Proofread segmentation `gs://liconn-public/ExPID82_1/segmentation/231030_agg_240123`.
  This is the layer named "proofread" in the authors' Neuroglancer state
  `ng_states/expid82.json` (paper, Data availability, "Full segmentation (Fig. 2a-f)").
  It is on the same 18 x 18 x 24 nm grid as the FFN input (scale 1). It is read for the
  same 500^3 crop (x, y, z offset 1100, 1083, 111), cached locally and pinned by sha256.
- Classes: visible segment IDs in `ng_states/expid82_fig2_axons.json` (544 axons) and
  `expid82_fig2_dends.json` (311 dendrites), stored in `liconn_v2/proofread_classes.json`.
  Each state lists the other class as hidden ("!") segments; those are excluded.
- Inside the crop: 1,629 proofread labels; 42.2% of voxels are unlabeled (0).
  66 axons and 15 dendrites are present.
- Caveat: per the paper, proofreading targeted specific errors (axon splits at a face,
  axon merges, spine attachment); it is not a dense voxel-perfect annotation.
  Unlabeled voxels cannot be judged.

## Seeds (deterministic)

- Candidates: proofread axons and dendrites with at most 350,000 voxels inside the crop
  and at least 20,000 voxels inside the margin box [16, 484)^3. That excludes objects
  that only graze a crop face. The upper bound keeps one CPU trace to minutes, and is a
  disclosed bias against the largest dendrites (up to 1.56 M voxels in the crop).
  benchmark.json records the candidate counts.
- Choose up to 6 per class with one generator `rng = numpy.random.default_rng(0)`,
  calling `rng.choice(sorted_ids, size=min(6, n), replace=False)` for axons, then for
  dendrites. In this crop 43 axons qualify but only 5 dendrites do; the next dendrites are
  930k and 1.56 M voxels. So the set is 6 axons + 5 dendrites + the notebook seed =
  12 seeds. Also include the authors' notebook seed (x, y, z = 123, 171, 225;
  proofread object 97257, unclassified) as a link to milestone 1.
- Seed point within an object: the voxel with the largest anisotropic (24, 18, 18 nm, z/y/x)
  Euclidean distance to the object boundary, among voxels at least 16 voxels from every
  crop face (the FFN field-of-view margin). Ties go to the lowest (z, y, x) index.
  The distance transform zero-pads the crop, so a cut face counts as a boundary.
  Known bias: on axons this picks the widest point (often a bouton), not a typical thin
  shaft.

## Per-seed run

The run is the same as milestone 1: authors' checkpoint, notebook settings, one fresh
Canvas per seed, `segment_at(seed)`. These are independent single-object traces, like
the notebook. They are not FFN's dense `segment_all` over the whole volume, which also
involves seed policies and exclusion of already-segmented voxels. Dense mode is out of
scope for this milestone.

FFN object mask M = canvas >= logit(segment_threshold = 0.6), with no connected-component
filtering. This is how `Canvas.segment_all` turns the canvas into a segment, before it
removes voxels already claimed by other objects. It differs from milestone 1's diagnostic
`seed_component`. Each seed's M is saved (bounding-box cropped, in
outputs/benchmark_masks.npz) so later variants can also be compared voxel-wise with the
baseline (IoU).

## Metrics

With G = the proofread object at the seed voxel and L = proofread labels:
- recall = |M ∩ G| / |G| (G restricted to the crop). Single traces cover only part of a
  large neuron, so recall below 1 is expected even for the reference method.
- precision = |M ∩ G| / |M ∩ (L > 0)| (only voxels the proofread data can judge).
- unlabeled_fraction = |M ∩ (L = 0)| / |M|.
- merged_voxels = |M ∩ (L > 0) \ G| (labeled voxels of other objects).
- merges = proofread objects other than G with overlap >= 1,000 voxels with M (the IDs
  and overlaps are listed). Small spine-sized merges fall below this threshold and show
  up only in merged_voxels and precision.
- steps = network evaluations. network_gflop = steps x GFLOP per evaluation. GFLOP per
  evaluation is computed from the loaded parameters and FOV as the dense-convolution
  count, 2 x prod(kernel) x Cin x Cout x FOV voxels, summed over conv kernels. For the
  authors' model that is 81.6 GFLOP; XLA's own estimate is 77.1. FLOPs do not reflect
  lower-precision arithmetic, so the number format and wall time are recorded alongside
  them.
- Compute-independent speed: evaluations per 1,000 correctly segmented voxels
  (steps / |M ∩ G| x 1000) and GFLOP per correctly segmented voxel.
- Wall time on this CPU is secondary.

Summary: per-class and overall means of recall and precision, total merges, total merged
voxels, total steps, total GFLOP, and total steps per 1,000 correctly segmented voxels
(a ratio of totals). Single-seed evaluations-per-voxel figures reward stopping before thin,
low-yield branches. Read them together with recall.

## Tolerance change decided 2026-10-21 (definition to be written)

After float16 failed on held-out seeds through rare large merge flips
(docs/heldout_results.md), the owner decided to move to a tolerance judged over many
more seeds that accepts rare merge flips. The exact definition is still to be
proposed and confirmed: number of seeds and crops, an allowed rate of merge or unmerge
flips, and aggregate accuracy limits. Until then, the rules below remain the recorded
criterion for all results so far.

## Accuracy tolerance for speed variants (owner-confirmed 2026-10-19)

A faster variant is compared seed by seed with the reference run on the same seeds and
inputs. It passes only if all of these hold, each at 1%:
- mean recall drops by at most 1 percentage point;
- mean precision drops by at most 1 percentage point;
- mean unlabeled_fraction rises by at most 1 percentage point;
- no seed's recall drops by more than 1 percentage point;
- total merged voxels grow by at most 1%, and no seed gains a new merge.

The implementer's earlier proposal allowed 5 points for single-seed recall and 10% for
merged voxels; the owner set every rule to 1%.

Reference run (owner decision, 2026-10-19): the 12-seed GPU baseline run a on Wahab
V100s (docs/gpu_baseline.md), `data/runs/gpu-baseline-a-20261019/merged`.

Caution (from the 12-seed baseline, docs/benchmark_baseline.md):
- dendrite_1525 accounts for 31% of all steps and 89% of merged voxels, and includes one
  large merge into object 97123. Totals and means are therefore very sensitive to how a
  variant behaves on this one seed (the size of that effect has not been measured).
- axon_218 is pinned at the margin (8 steps), and its recall can jump either way.
Therefore apply the tolerance per seed, compare each seed's mask with its baseline mask
(IoU), and report results with and without these two seeds. Means and totals alone are
not sufficient.

Open issue (Cursor method check): single traces stop at narrow necks near the move
threshold. A tiny logit change can add or drop a whole branch, so per-seed recall is
jumpy, and 12 seeds is a small sample. Settle the tolerance with the owner after seeing
the baseline's per-seed spread, and consider voxel-wise IoU with the baseline masks as an
additional check.

## Verification plan

- Unit tests with synthetic arrays check the metric arithmetic against hand-computed answers.
- Seed selection: deterministic, inside the margin, inside the intended object and class.
- Cached ground truth matches its pinned sha256.
- Milestone 1's bitwise reference trace is unchanged after refactoring (existing tests).
- The controller example runs the benchmark on the notebook seed only (about 2.5 min).
  The full 12-seed baseline is a separate, owner-approved CPU run, recorded with
  output hashes.
