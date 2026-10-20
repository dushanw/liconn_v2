# Can low-yield steps be predicted in advance? Results (run 2026-10-19)

This is an implementer-run record, approved by the owner. The traces ran on a Wahab
V100 (Slurm job 7359027; base commit ac5ad6a plus uncommitted changes) with the
authors' settings. Every seed's canvas and visit order matched the adopted GPU
reference (gpu-baseline-a-20261019) bitwise. tests/test_predictability_evidence.py
checks the following against docs/predictability_evidence.json:
- the trace code is unchanged;
- the outputs are the recorded ones;
- every trace matched the reference;
- the numbers below follow from the recorded per-step data.

- Seeds: the 11 waste-analysis seeds (benchmark seeds without dendrite_1525).
- Steps analysed: 5,838, excluding each trace's first step, which is the seed itself
  and cannot be skipped.
- Every recorded queue score is the entry FFN actually popped for that step.
- Outputs: `data/runs/predictability-20261019/predictability.json` and
  `predictability_steps.npz`. `predictability_reanalysed.json` holds the extended
  analysis, recomputed from the same per-step data; it reproduces the run's analysis
  exactly.

## 1. Over all steps

| target | share of steps | model AUC (leave-one-seed-out, pooled) | per-seed AUC, quartiles | best single feature (AUC) |
|---|---|---|---|---|
| low_yield (approved definition) | 15.2% | 0.907 | 0.905 / 0.925 / 0.955 | unpredicted_fraction, lower = low-yield (0.957) |
| mask_low_yield | 34.1% | 0.834 | 0.828 / 0.835 / 0.842 | unpredicted_fraction (0.796) |

AUCs are oriented: a feature that works in the opposite direction is reported as
1 - AUC. Other features over all steps:
- uncertain_fraction: 0.705 and 0.730;
- overlap_count: 0.761 and 0.653;
- mean_margin: 0.506 and 0.632;
- fg_fraction, center_value, queue_score and step_index: 0.51-0.59.

**This high score is partly built into the definition.** A step with 359 or more
not-yet-predicted voxels in its box is never low-yield under the approved
definition, and that number is known before the step runs. So separating these steps
from the rest says nothing about the network's behaviour.

## 2. The question that matters: nearly enclosed boxes

Nearly enclosed steps have fewer than 359 not-yet-predicted voxels, which is known in
advance. They are 27.2% of all steps (1,587) and contain **every** low-yield step:
56.0% of them are low-yield (0.560 × 0.272 = 15.2% of all steps). Within this group:

| target | share within group | model AUC | per-seed AUC, quartiles | strongest features (AUC) |
|---|---|---|---|---|
| low_yield | 56.0% | **0.752** | 0.725 / 0.752 / 0.770 | unpredicted_fraction 0.696, uncertain_fraction 0.644 |
| mask_low_yield | 63.3% | **0.663** | 0.640 / 0.669 / 0.696 | uncertain_fraction 0.615, mean_margin 0.594 |

Operating points within the group, for low_yield:

| share of low-yield caught | useful steps wrongly flagged | of which frontier steps (would have queued new territory) | flagged, as share of the group |
|---|---|---|---|
| 10% | 3.9% | 4.7% | 7.3% |
| 25% | 9.2% | 11.0% | 18.1% |
| 50% | 22.1% | 26.8% | 37.7% |

Rule "box already fully predicted" (enclosed): catches 26.0% of low-yield steps and
flags 12.8% of useful steps in the group (15.4% of its frontier steps).

What the canvas values add: three nested logistic models, each evaluated
leave-one-seed-out:
- coverage only: unpredicted_fraction;
- + position: overlap_count, step_index;
- + canvas: fg_fraction, uncertain_fraction, mean_margin, center_value, queue_score.

| group, target | coverage only | + position | + canvas (all 8) |
|---|---|---|---|
| nearly enclosed, low_yield | 0.692 (median 0.687) | 0.696 (0.685) | 0.752 (0.752) |
| nearly enclosed, mask_low_yield | 0.581 (0.604) | 0.588 (0.611) | 0.663 (0.669) |
| all steps, low_yield | 0.956 (0.960) | 0.902 (0.929) | 0.907 (0.925) |
| all steps, mask_low_yield | 0.794 (0.798) | 0.785 (0.800) | 0.834 (0.835) |

Values are pooled out-of-fold AUC, with the per-seed median in brackets. Among
nearly enclosed steps, adding the canvas values raises AUC by about 0.06
(approved definition) and 0.08 (mask-only definition); position adds nothing. Over
all steps under the approved definition, coverage alone does best: with this simple
linear model, the extra features dilute the definitional coverage signal.

## What this means for lever A (a smarter movement policy)

- The strongest signal that a step will be low-yield is free: how much of its box is
  already predicted. A rule using only that number narrows the search to 27% of
  steps, which contain all low-yield steps. Among those, about 44% are still useful,
  and some of them open new territory.
- Beyond that, the canvas values add a modest amount: among nearly enclosed steps,
  AUC rises from about 0.69 (coverage only) to 0.75 under the approved definition, and
  from 0.58 to 0.66 under the mask-only definition. Skipping steps on this basis would also skip many useful and
  frontier steps. For example, catching half of the low-yield steps would wrongly flag
  about a quarter of the group's frontier steps. That risks truncating branches,
  which the 1% recall rule would catch.
- Rough scale, before any policy run: low-yield steps are 15% of all steps. Skipping
  a fraction of them could save at most a similar fraction of evaluations, and
  wrongly skipped useful steps would cost accuracy. **Postponing** nearly enclosed
  steps (ranking them last in the queue) may be safer than skipping them, because
  later steps may make them redundant. That has not been measured.

## What this does not show

- Prediction quality is not a speedup. Skipping or reordering steps changes later
  steps, so any policy must be run on the benchmark against the reference under the
  1% tolerance.
- 11 single-object traces on one crop. The model is a simple logistic regression on 8
  features; a more flexible model was not tried.
- Pooled out-of-fold scores combine 11 fold models trained on different base rates;
  the per-seed AUCs are given for that reason.
