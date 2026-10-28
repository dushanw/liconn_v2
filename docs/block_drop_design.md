# Lever B, part 1: dropping residual blocks without retraining (milestone 7)

Owner decision (2026-10-20): after lever A, move to lever B, cheaper network
evaluations. Start with dropping residual blocks, which needs no training. This
changes the network's computation, but not its weights, the movement policy,
thresholds or box size.

## The network

The authors' ResConvStack (connectomics/jax/models/convstack.py, depth 20):
- input stage: conv 2->32, LayerNorm_0, relu, conv 32->32;
- 20 residual blocks i = 0..19: x <- x + conv_b(relu(conv_a(relu(LayerNorm_{i+1}(x)))));
- output: LayerNorm_21, relu, 1x1x1 conv to 1 channel.

Each block holds 2 of the 43 conv kernels: 3.97 of the 81.6 GFLOP per evaluation
(4.9%). Dropping block i replaces it with the identity. Its weights are simply not
used; nothing is retrained.

Implementation: liconn_v2/pruned.py defines PrunedResConvStack, a copy of the
module's forward pass with every layer named explicitly, as in the checkpoint
(pre_a, pre_b, res{i}a, res{i}b, LayerNorm_k, output), and a set of blocks to skip.
With no blocks skipped it must reproduce the authors' model output bitwise (tested).
reference.py, benchmark.py and run_benchmark.py stay byte-identical, because earlier
evidence pins them. A variant is run by replacing the tracer's compiled network
function (`Tracer._exc._apply_fn`) after the tracer is built.

## Step 1: offline screen

Inputs recorded from the reference traces (GPU, authors' settings, default policy):
for all 12 benchmark seeds, the network input of every 8th step, i.e. the image box
and the seed box exactly as FFN feeds them (`Canvas.predict`). Recording is
read-only. Each seed's canvas and visit-order sha256 must match the reference run.

For each variant, on these inputs:
- flip rate: share of output voxels whose decision (logit >= logit(0.6)) differs from
  the full network's; split into lost (object -> background) and gained;
- mean |logit difference| and mean signed logit shift;
- GFLOP per evaluation, computed from the kernels actually used.

Protocol amendment (recorded 2026-10-21): the first version of this design ranked
blocks once by single-drop flip rate and dropped prefixes of that order. After the
Cursor method check, and before any screen was run, it was changed to the
sequential greedy search below. Evidence: the screen job (Slurm 7362833, submitted
2026-10-20 19:22:35) recorded the source sha256 at submission in source_sha256.txt.
That hash for this file matches the version describing the sequential search; the
only later edit to this file is the per-block GFLOP wording. The screen results were
written at 20:12:02.

Variants: every single-block drop (20), then a sequential greedy search. At step k it
adds the block whose removal, together with the k-1 blocks already dropped, gives the
lowest measured flip rate (ties to the lower index). That accounts for interactions,
such as neighbouring blocks that are each harmless alone (Cursor method check,
2026-10-20).

## Step 2: benchmark, with variants chosen by a rule fixed in advance

From the screen, at most 4 variants go to the full 12-seed benchmark on Wahab V100s:
1. the single block with the lowest flip rate;
2. the largest greedy set with flip rate <= 0.1%;
3. the largest greedy set with flip rate <= 0.5%;
4. the largest greedy set with flip rate <= 2%.

Duplicates are removed, and sets with no qualifying k are skipped. Each variant is
compared with the reference by examples/compare_runs.py: passes_tolerance (every
rule at 1%), mask IoU and steps.

Cost is reported as GFLOP per correct voxel, decomposed into the change in GFLOP
per evaluation and the change in steps, for all seeds and without dendrite_1525 and
axon_218. A change in dendrite_1525's false merge alone could otherwise dominate the
totals, as it did in lever A. A variant passing on these 12 seeds would still need held-out
confirmation before being called a speedup. That is noted but not part of this
milestone.

## Verification

- Unit tests: PrunedResConvStack with no blocks dropped equals the authors' model
  bitwise on random input; dropping block 7 equals the authors' model with that
  block's second convolution zeroed, so the block adds exactly 0; GFLOP accounting;
  sequential greedy on a synthetic interaction; selection rule on synthetic
  screen results.
- The controller example runs the screen on CPU for the notebook seed only, with a
  small sample. Its numbers are not the screen result.
- The GPU screen and the benchmark runs are owner-approved implementer runs on Wahab,
  recorded with source hashes, job IDs and output sha256, and checked by an evidence
  test.

## Limits

Open-loop screen: a low flip rate on recorded inputs does not guarantee the closed
trace stays the same, because errors can compound over steps. Single-object traces
on one crop.
