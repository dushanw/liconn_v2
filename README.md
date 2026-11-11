# liconn_v2: the computational cost of neuron tracing for LICONN

LICONN (Tavakoli et al., *Nature* 642, 398–410, 2025,
[doi:10.1038/s41586-025-08985-1](https://doi.org/10.1038/s41586-025-08985-1)) maps brain tissue
with expansion light microscopy and traces every neuron with a flood-filling network (FFN).
Tracing is the expensive step: one network evaluation costs 81.6 GFLOP, and one dendrite can take
thousands of them ([block_drop](docs/block_drop_results.md), [benchmark](docs/benchmark_baseline.md)). This repository measures that cost and asks which ways of reducing it keep the
method's output, which keep its accuracy, and what faster alternatives (smaller distilled tracers,
a fine-tuned dense segmentation model) cost in accuracy, all judged against the authors' proofread
segmentation and, finally, on ten crops that no part of the work had read.

**Overview with figures and 3D views:** <https://wadduwagelab.com/demos/liconn-ffn/>

**Every study, in logical order, with its design, results, evidence and evidence test:**
[docs/README.md](docs/README.md).

## Results, in logical order

Numbers marked as tuning-crop numbers come from crops that also chose settings, checkpoints or
thresholds; the test numbers (R6) come from crops 41-50, with methods and analysis frozen before
those crops were fetched ([unseen](docs/unseen_results.md)).

| | what | result | record |
|---|---|---|---|
| R1 | Reference | The authors' notebook example (one seed, 359 steps) reproduced bitwise on CPU. On a V100 the 12 benchmark seeds stay within the 1% tolerance of the CPU; 19.7 ms per step in float32. This GPU run is the reference for speed variants. | [method_source](docs/method_source.md), [gpu_baseline](docs/gpu_baseline.md) |
| R2 | Speed-ups that keep every trace | Fast tracer: output bitwise identical on all 624 traces checked; 1.12x per step in float32 and 1.43x in `fp16_head32`. PyTorch port: largest logit difference to the authors' network 4.1e-5. Tracing 8 objects at once on a B200 changed no trace and cut the 512 crop-11 objects from 585.6 s to 244.1 s (`fp16_head32`). | [fasttrace](docs/fasttrace_results.md), [port](docs/factorised_port_results.md), [timing study](docs/distil_results.md) |
| R3 | Speed-ups that change the arithmetic | Skipping or postponing moves and dropping residual blocks failed the per-seed rule. float16 failed on held-out seeds (two dendrites merged) and on 300 seeds (10 new early stops, limit 3). `fp16_head32` with the fast tracer is 3.2x faster per step; it failed the pre-registered rule on 300 fresh seeds (5 new merges, all in dendrites) and passes the 2% rule the owner set after that failure. | [postpone](docs/postpone_results.md), [block_drop](docs/block_drop_results.md), [heldout](docs/heldout_results.md), [manyseed](docs/manyseed_results.md), [fresh](docs/fresh_results.md), [tolerance2](docs/tolerance2_results.md), [fasttrace](docs/fasttrace_results.md) |
| R4 | Smaller tracers (tuning crop 11) | Distilled from the authors' network: at the fastest settings P8 traced the 512 crop-11 objects in 130 s at mean F1 0.870, the factorised student in 151 s at 0.876, FFN in 229 s at 0.902. | [distil](docs/distil_results.md), [prune](docs/prune_results.md) |
| R5 | Dense segmentation (tuning crop 12) | SegNeuron zero-shot: mean F1 0.500; fine-tuned on crops 0-11: 0.823; a ViT trained from scratch: 0.531. FFN on the same 457 objects: 0.891. The fine-tuning ran 7.71x faster per step on a B200 than on a V100. | [segneuron](docs/segneuron_results.md), [finetune](docs/segneuron_finetune_results.md), [vitseg](docs/vitseg_results.md), [headtohead](docs/headtohead_results.md), [gpu_training](docs/gpu_training_results.md) |
| R6 | Test on unseen crops 41-50 (4,579 objects) | FFN's fast mode passes the 2% rule against float32 (limit 91.58 objects) and is 4.8x faster. FFN is more accurate than fine-tuned SegNeuron on every crop (+0.0580 mean F1). The student, P8 and P4 are each below FFN (-0.0280, -0.0260, -0.0489; "worse than FFN" under the margin fixed in advance) and 8.4x (P8) and 13.4x (P4) faster than float32 FFN. | [unseen](docs/unseen_results.md) |
| R6 | SegNeuron timed on the B200 (addendum, planned after the test's results) | Against batched fast FFN: SegNeuron cheaper on 7 of 10 crops; break-even 389 objects per crop from the totals. | [unseen_b200](docs/unseen_b200_results.md) |

Caveats that go with these numbers:

- **The 2% rule was set after a failure.** All 5 new merges on the fresh seeds were in dendrites;
  the 2% pass is pooled over axons and dendrites. On the test crops the rule was fixed before the
  crops were read ([fresh](docs/fresh_results.md), [tolerance2](docs/tolerance2_results.md),
  [unseen](docs/unseen_results.md)).
- **Two versions of `fp16_head32`.** The fresh seeds judged the V100 version (JAX, one object at a
  time); the test crops judged the B200 version (PyTorch, residual stream divided by 16, 32 objects
  at once) ([fresh](docs/fresh_results.md), [unseen](docs/unseen_results.md)).
- **Tuning-crop numbers are optimistic:** crop 11 chose the smaller tracers' checkpoints and crop 12
  chose SegNeuron's checkpoint and threshold ([prune](docs/prune_results.md),
  [finetune](docs/segneuron_finetune_results.md)).
- **Recorded afterwards:** fine-tuning SegNeuron on newer GPUs (A100, H100, B200) had no design before
  its runs; its numbers are recomputed from the training logs ([gpu_training](docs/gpu_training_results.md)).

## How results are checked

Most studies have a design (`docs/*_design.md`), a results document (`docs/*_results.md`), an
evidence file (`docs/*_evidence.json`) and an evidence test (`tests/test_*_evidence.py`);
[docs/README.md](docs/README.md) lists them and marks the exceptions. An evidence test checks,
against the run records in the data cache, that the code that ran matches this repository (by
sha256, or for disclosed post-run edits by comparing definitions), that the outputs are the
recorded ones, and that the numbers in the evidence file follow from them. Numbers a results
document marks as outside the evidence (for example some run times) are not checked.

The work was done with Claude Code under a review workflow (ResearchFlow): each milestone had a
written task contract, an independent method check by a second model and an evidence review by
a third; the human owner made the research decisions. The webpage summarises the process.

## Repository layout

| path | contents |
|---|---|
| `liconn_v2/` | The library: reference inference (`reference.py`), benchmark and scoring (`benchmark.py`), number formats (`precision.py`, `earlystop.py`), movement policies (`policies.py`), fast tracer (`fasttrace.py`), seed sets and verdicts (`manyseed.py`, `fresh.py`, `tolerance2.py`), PyTorch port and smaller networks (`factorised.py`, `distil.py`, `pruned.py`), ViT (`vitseg.py`), SegNeuron evaluation (`segneuron.py`, `headtohead.py`), the unseen-crop test (`unseen.py`, `unseen_b200.py`), data cache (`data.py`). |
| `examples/` | Command-line entry points for every study: runs, judges and evidence writers. |
| `tests/` | Unit tests and evidence tests. |
| `docs/` | Designs, results and evidence for each study; [docs/README.md](docs/README.md) is the index. |
| `cluster/wahab/` | Slurm job scripts and setup notes for ODU's Wahab cluster (V100 GPUs). |
| `cluster/mit/` | Slurm job scripts and setup for MIT's Engaging cluster (B200 GPUs). |
| `tools/` | A local Neuroglancer viewer for benchmark traces. |

## Setup

Python 3.12. FFN is pinned to the commit the paper cites (`12d680e`); connectomics to the last
`main` commit before it (`73b76a7f`, an inference, not stated by the paper); see
`docs/method_source.md`.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -r requirements-ffn.txt
.venv/bin/python -m pip install -r examples/requirements-erl.txt   # ERL (kimimaro); needed by tests/test_erl.py
.venv/bin/python -m pip install -r examples/requirements-torch.txt # PyTorch; needed by tests/test_factorised.py
.venv/bin/python -m liconn_v2.data            # one-time download of the public sample crop and checkpoint, sha256-pinned
.venv/bin/python -m unittest discover -s tests -t .
```

The data cache defaults to `~/ResearchFlow/data/liconn-v2`; set `LICONN_DATA` to move it.
`liconn_v2.data` fetches only the sample crop and checkpoint. Many tests also need the further
crops (held-out, many-seed, fresh and test crops; see the designs) and the recorded runs under
`runs/` in the cache; those run records and the trained weights are not yet published, so the
evidence tests and some unit tests pass only on a machine that holds them.
`tools/requirements-viewer.txt` adds the viewer.

## Reproducing

- **Reference trace (CPU):** `.venv/bin/python examples/trace_reference.py` reproduces the
  authors' notebook example; its fingerprint is `liconn_v2/expected_reference.json`.
- **Benchmark:** `examples/run_benchmark.py` traces seeds and scores them against the proofread
  segmentation (`docs/benchmark_design.md`).
- **GPU runs:** set `LICONN_BACKEND=gpu`. The V100 runs (many-seed, fresh-seed, fast tracer, ViT,
  SegNeuron) used `cluster/wahab/`; the B200 runs (PyTorch port, distillation, the unseen-crop
  test, the SegNeuron timing addendum) used `cluster/mit/`. The commands are in the designs
  (`docs/*_design.md`), and each results document lists its Slurm jobs.

## Data and licences

- Data: the LICONN public release (Tavakoli et al. 2025), CC BY 4.0, read from the public
  `gs://liconn-public` bucket; FFN checkpoint `axde_59110972` from the same release. Data,
  model weights and run outputs are not stored in this repository.
- This repository's code and documents: MIT licence (`LICENSE`).
- Third-party material keeps its own licence. `liconn_v2/reference.py`, `precision.py`,
  `earlystop.py`, `pruned.py`, `policies.py` and `fasttrace.py` follow or re-implement code from
  [google/ffn](https://github.com/google/ffn) and
  [google-research/connectomics](https://github.com/google-research/connectomics) (Apache-2.0),
  and `docs/method_source.md` quotes the authors' notebook (Apache-2.0).
  `docs/segneuron_sources.md` quotes, and `tests/test_segneuron.py` re-implements part of,
  [SegNeuron](https://github.com/yanchaoz/SegNeuron) (MIT).

## Citation

If you use this work, please cite the LICONN paper above, and FFN (Januszewski et al.,
*Nature Methods* 15, 605–610, 2018) for the tracing method.
