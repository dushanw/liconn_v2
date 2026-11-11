# Studies, in logical order

Every study in this repository, grouped by the question it answers rather than by the date it ran.
Each has a design written before its runs, a results document, and, where the numbers come from
recorded runs, an evidence file and an evidence test that checks them against the run records.
"none" in a column means the study has no such file; that is stated, not hidden.

Crop roles (fixed in advance): crop A (the notebook's crop) for the 12-seed benchmark; crop B for
held-out seeds; crops 0-12 for the first 300-seed set and for training (0-11) and tuning (11, 12);
crops 13-40 for the 300 fresh seeds (never segmented by a dense model); crops 41-50 for the final
test of every method ([benchmark_design](benchmark_design.md), [heldout_design](heldout_design.md),
[manyseed_design](manyseed_design.md), [fresh_design](fresh_design.md), [segneuron_finetune_design](segneuron_finetune_design.md),
[distil_design](distil_design.md), [unseen_design](unseen_design.md)).

## R1. The reference, the cost and the yardstick

What the authors' method does, reproduced; where its time goes; and the rules every change is
judged by.

| study | design | results | evidence | evidence test |
|---|---|---|---|---|
| Method source: the authors' notebook, pinned commits | [method_source](method_source.md) | (same document) | none | [test_reference_trace](../tests/test_reference_trace.py) |
| 12-seed benchmark and per-seed rule (1%) | [benchmark_design](benchmark_design.md) | [benchmark_baseline](benchmark_baseline.md) | none | [test_benchmark](../tests/test_benchmark.py) |
| GPU reference on Wahab V100s | [gpu_design](gpu_design.md) | [gpu_baseline](gpu_baseline.md) | [gpu_baseline_evidence](gpu_baseline_evidence.json) | [test_gpu_baseline_evidence](../tests/test_gpu_baseline_evidence.py) |
| 300-seed rule (first seed set) | [manyseed_design](manyseed_design.md) | (see R3) | [manyseed_crops](manyseed_crops.json), [manyseed_seeds](manyseed_seeds.json) | [test_manyseed](../tests/test_manyseed.py) |
| 300 fresh seeds (second seed set) | [fresh_design](fresh_design.md) | (see R3) | [fresh_crops](fresh_crops.json), [fresh_seeds](fresh_seeds.json) | [test_fresh](../tests/test_fresh.py) |
| Test crops 41-50, fixed before they were fetched | [unseen_design](unseen_design.md) | (see R6) | [unseen_crops](unseen_crops.json) | [test_unseen](../tests/test_unseen.py) |

## R2. Speed-ups that keep every trace

| study | design | results | evidence | evidence test |
|---|---|---|---|---|
| Fast tracer (bookkeeping on the GPU), bitwise identical | [fasttrace_design](fasttrace_design.md) | [fasttrace_results](fasttrace_results.md) | [fasttrace_evidence](fasttrace_evidence.json) | [test_fasttrace_evidence](../tests/test_fasttrace_evidence.py) |
| PyTorch port of the authors' network | [factorised_design](factorised_design.md), [factorised_sources](factorised_sources.md) | [factorised_port_results](factorised_port_results.md) | none (unit tests repeat each check) | [test_factorised](../tests/test_factorised.py) |
| Tracing many objects at once on the B200 (timing study) | [distil_design](distil_design.md) | [distil_results](distil_results.md), "Timing study" | [distil_timing_evidence](distil_timing_evidence.json) | none for the timing extract (its numbers are checked where the manuscript cites them) |
| Float32 traced many objects at once: crop-11 widths and the test crops (added after the test) | [fp32_batch_design](fp32_batch_design.md) | [fp32_batch_results](fp32_batch_results.md) | [fp32_batch_evidence](fp32_batch_evidence.json) | [test_fp32_batch](../tests/test_fp32_batch.py), [test_fp32_batch_evidence](../tests/test_fp32_batch_evidence.py) (against the run records) |

## R3. Speed-ups that change the arithmetic

Changes to the movement policy and to the network's depth, then lower precision, judged first on
the 12 benchmark seeds and then on 300 seeds ([benchmark_design](benchmark_design.md),
[manyseed_design](manyseed_design.md)); the rule change after the fresh-seed failure is recorded in
[tolerance2_results](tolerance2_results.md).

| study | design | results | evidence | evidence test |
|---|---|---|---|---|
| Where steps are wasted | [waste_analysis_design](waste_analysis_design.md) | [waste_analysis_results](waste_analysis_results.md) | none | [test_waste](../tests/test_waste.py) |
| Can low-yield steps be predicted? | [predictability_design](predictability_design.md) | [predictability_results](predictability_results.md) | [predictability_evidence](predictability_evidence.json) | [test_predictability_evidence](../tests/test_predictability_evidence.py) |
| Skipping and postponing moves | [postpone_design](postpone_design.md) | [postpone_results](postpone_results.md) | [postpone_evidence](postpone_evidence.json) | [test_postpone_evidence](../tests/test_postpone_evidence.py) |
| Dropping residual blocks | [block_drop_design](block_drop_design.md) | [block_drop_results](block_drop_results.md) | [block_drop_evidence](block_drop_evidence.json) | [test_block_drop_evidence](../tests/test_block_drop_evidence.py) |
| float16 and bfloat16 on the benchmark | [precision_design](precision_design.md) | [precision_results](precision_results.md) | [precision_evidence](precision_evidence.json) | [test_precision_evidence](../tests/test_precision_evidence.py) |
| float16 on held-out seeds | [heldout_design](heldout_design.md) | [heldout_results](heldout_results.md) | [heldout_evidence](heldout_evidence.json) | [test_heldout_evidence](../tests/test_heldout_evidence.py) |
| float16 on 300 seeds | [manyseed_design](manyseed_design.md) | [manyseed_results](manyseed_results.md) | [manyseed_evidence](manyseed_evidence.json) | [test_manyseed_evidence](../tests/test_manyseed_evidence.py) |
| Why float16 stops traces early | [earlystop_design](earlystop_design.md) | [earlystop_results](earlystop_results.md) | [earlystop_evidence](earlystop_evidence.json) | [test_earlystop_evidence](../tests/test_earlystop_evidence.py) |
| `fp16_head32` on 300 seeds | [head32_design](head32_design.md) | [head32_results](head32_results.md) | [head32_evidence](head32_evidence.json) | [test_head32_evidence](../tests/test_head32_evidence.py) |
| `fp16_head32` on 300 fresh seeds (fails the pre-registered rule) | [fresh_design](fresh_design.md) | [fresh_results](fresh_results.md) | [fresh_evidence](fresh_evidence.json) | [test_fresh_evidence](../tests/test_fresh_evidence.py) |
| The 2% rule, set after that failure | (owner decision, in the results) | [tolerance2_results](tolerance2_results.md) | none | [test_tolerance2](../tests/test_tolerance2.py) |
| Ideas beyond float16 (brainstorm, not a study) | [beyond_fp16_ideas](beyond_fp16_ideas.md) | none | none | none |

The B200 form used on the test crops (`fp16_head32`, residual stream divided by 16, 32 objects at
once) was developed in the timing study of R2 ([distil_results](distil_results.md)) and judged in R6
([unseen_results](unseen_results.md)).

## R4. Smaller tracers by distillation

| study | design | results | evidence | evidence test |
|---|---|---|---|---|
| Feasibility and data sources | [training_feasibility](training_feasibility.md) | (same document) | none | none |
| Factorised FFN, distilled (student) | [distil_design](distil_design.md) | [distil_results](distil_results.md) | [distil_evidence_r1](distil_evidence_r1.json), [distil_timing_evidence](distil_timing_evidence.json) | [test_distil_evidence_r1](../tests/test_distil_evidence_r1.py) (round-0 diagnostic and round 1, against the run records) |
| Pruned FFNs P8 and P4, distilled | [prune_design](prune_design.md) | [prune_results](prune_results.md) | [prune_evidence](prune_evidence.json) | [test_prune_evidence](../tests/test_prune_evidence.py) |

## R5. Dense segmentation instead of tracing

| study | design | results | evidence | evidence test |
|---|---|---|---|---|
| Single-pass vision transformer (stopped pilot) | [vitseg_design](vitseg_design.md) | [vitseg_results](vitseg_results.md) | [vitseg_evidence](vitseg_evidence.json) | [test_vitseg_evidence](../tests/test_vitseg_evidence.py) |
| SegNeuron zero-shot | [segneuron_design](segneuron_design.md), [segneuron_sources](segneuron_sources.md) | [segneuron_results](segneuron_results.md) | [segneuron_evidence](segneuron_evidence.json) | [test_segneuron_evidence](../tests/test_segneuron_evidence.py) |
| SegNeuron fine-tuned on crops 0-11 | [segneuron_finetune_design](segneuron_finetune_design.md), [segneuron_finetune_sources](segneuron_finetune_sources.md) | [segneuron_finetune_results](segneuron_finetune_results.md) | [segneuron_finetune_evidence](segneuron_finetune_evidence.json) | [test_segneuron_ft_evidence](../tests/test_segneuron_ft_evidence.py) |
| FFN against fine-tuned SegNeuron on crop 12 | [headtohead_design](headtohead_design.md) | [headtohead_results](headtohead_results.md) | [headtohead_evidence](headtohead_evidence.json) | [test_headtohead_evidence](../tests/test_headtohead_evidence.py) |
| Fine-tuning SegNeuron on newer GPUs (A100, H100, B200; batch sizes) | **none** (exploratory runs, recorded afterwards from their logs) | [gpu_training_results](gpu_training_results.md) | [gpu_training_evidence](gpu_training_evidence.json) | [test_gpu_training_evidence](../tests/test_gpu_training_evidence.py) |

## R6. The unseen-crop test, and accuracy against cost

| study | design | results | evidence | evidence test |
|---|---|---|---|---|
| Every method on crops 41-50, frozen in advance | [unseen_design](unseen_design.md) | [unseen_results](unseen_results.md) | [unseen_evidence](unseen_evidence.json) | [test_unseen_evidence](../tests/test_unseen_evidence.py) |
| SegNeuron timed on the B200 (addendum, planned after the test's results) | [unseen_b200_segneuron_design](unseen_b200_segneuron_design.md) | [unseen_b200_results](unseen_b200_results.md) | [unseen_b200_evidence](unseen_b200_evidence.json) | [test_unseen_b200_evidence](../tests/test_unseen_b200_evidence.py) |
| Whole test-crop neurons as reconstructed by every method (manuscript figure, after the test) | [figure_neurons_design](figure_neurons_design.md) | three objects run, no verification failures (Run section of [figure_neurons_design](figure_neurons_design.md)); Fig. 5 (Draft 10) | [figure_neurons_evidence](figure_neurons_evidence/logs.txt) | [test_figure_neurons](../tests/test_figure_neurons.py), [test_figure_neurons_evidence](../tests/test_figure_neurons_evidence.py) |

## Records

- [records/README_as_staged_for_unseen_test.md](records/README_as_staged_for_unseen_test.md): the
  top-level README exactly as it was when the unseen-crop test's code was staged; its sha256 is
  pinned in unseen_evidence.json, and test_unseen_evidence checks it there.
