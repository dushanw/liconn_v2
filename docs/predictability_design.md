# Can low-yield steps be predicted in advance? Design (milestone 5)

Background (docs/waste_analysis_results.md): no FFN step is idle, but 15% of steps are
low-yield under the owner-approved definition (34% under the mask-only definition).
Only 5.5% of steps are recognisable before evaluating (box fully predicted). A
smarter movement policy can skip or postpone a step only if it can tell, before
paying for the evaluation, that the step will be low-yield. This milestone measures
how well that can be done. It changes nothing in the method.

## Data

The 11 seeds of the waste analysis (the benchmark seeds without dendrite_1525; owner
decision 2026-10-19). Traces are re-run with the authors' settings on the adopted
reference backend: Wahab V100, `LICONN_BACKEND=gpu`, deterministic XLA ops.
Instrumentation is read-only. Every seed's final canvas sha256 and visit-order sha256
must equal those in the reference run (docs/gpu_baseline.md); otherwise the run is
rejected.

## Features (known before the step's evaluation)

For the queued position p about to be evaluated, with B the 33^3 box around p and
t = logit(0.6), computed from the canvas before the update:
- unpredicted_fraction: share of B never predicted (NaN).
- fg_fraction: share of B's predicted voxels at or above t.
- uncertain_fraction: share of B's predicted voxels with |value - t| < 1 (roughly
  probability 0.35-0.80).
- mean_margin: mean |value - t| over B's predicted voxels.
- center_value: the canvas value at p.
- queue_score: the probability with which p was queued, i.e. the face maximum that
  proposed it (the highest, if it was proposed more than once).
- overlap_count: number of earlier FOV centers whose boxes overlap B (|dz|, |dy|,
  |dx| <= 32).
- step_index: number of steps taken so far in this trace.

## Targets (known only after the step)

- low_yield: the owner-approved definition (primary).
- mask_low_yield: the additional mask-only definition.

Both are as in docs/waste_analysis_design.md and are computed by the same code
(liconn_v2/waste.py).

## Analysis

- Per feature: ROC AUC for separating low-yield from other steps, pooled over seeds.
  0.5 means no information and 1.0 perfect separation.
- Combined: logistic regression on standardized features, evaluated with
  leave-one-seed-out cross-validation, so every prediction comes from a model that
  never saw that seed. Reported: pooled out-of-fold AUC, and per-seed AUC where a seed
  has both classes.
- Operating points: for thresholds on the out-of-fold score, the share of low-yield
  steps caught and the share of useful (not low-yield) steps that would be wrongly
  skipped. Reported at 10%, 25% and 50% of low-yield steps caught.

## Reading the result

A high AUC (for example above 0.8) with few useful steps wrongly flagged would make a
gain-ranked or skip policy worth building and testing on the benchmark. Even then,
skipping changes later steps, so the real effect must be measured by running the
policy against the reference under the 1% tolerance. A low AUC would mean lever A
offers little beyond the already-known 5.5%, and the plan would move to lever B
(cheaper network evaluations).

## Verification

- Unit tests: feature computation on synthetic canvases; queue-score capture; AUC
  against hand-computed values; logistic regression on separable synthetic data;
  leave-one-seed-out folds never train on the held-out seed.
- The instrumented notebook-seed trace on CPU stays bitwise identical to milestone 1.
- The controller example runs the notebook seed on CPU (one seed, so no
  cross-validation; reported as such).
- The 11-seed GPU run is an owner-approved implementer run on Wahab. It is recorded
  with source hashes, job ID and output sha256, and checked against the reference's
  per-seed canvas and visit-order hashes.

## Limits

Single-object traces, one 500^3 crop, 11 seeds (about 5,800 steps). Offline prediction
quality is not a speedup.
