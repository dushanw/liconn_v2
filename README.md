# liconn_v2: faster neuron tracing for LICONN

LICONN (Tavakoli et al., *Nature* 642, 398–410, 2025,
[doi:10.1038/s41586-025-08985-1](https://doi.org/10.1038/s41586-025-08985-1)) maps brain tissue
with expansion light microscopy and traces every neuron with a flood-filling network (FFN).
Tracing is the expensive step: one network evaluation costs 81.6 GFLOP, and one dendrite can take
thousands of them. This repository looks for a faster *method* that finds the same neurons,
judged against the authors' proofread segmentation.

**Overview with figures and 3D views:** <https://wadduwagelab.com/demos/liconn-ffn/>

## Results

| what | result | record |
|---|---|---|
| Reference | The authors' notebook example (one seed, 359 steps) reproduced bitwise on CPU, with the setup deviations disclosed. On a V100 the 12 benchmark seeds stay within the 1% tolerance of the CPU (10 voxel-identical, 2 dendrites diverge slightly); the GPU run is the reference for speed variants. 19.7 ms per step in float32. | [method_source](docs/method_source.md), [gpu_baseline](docs/gpu_baseline.md) |
| float16 network with a float32 output head (`fp16_head32`) plus the fast tracer | Per step: **3.2×** on the 12 benchmark seeds (5.96 vs 19.27 ms, same jobs); **3.0×** on 300 fresh seeds (6.41 vs 19.51 ms, separate jobs). Passes the owner's 2% rule on the fresh seeds (5 new merges, 3 new early stops; limit 6 each) — a rule set *after* it failed the original 1% rule there. | [fasttrace](docs/fasttrace_results.md), [tolerance2](docs/tolerance2_results.md), [fresh](docs/fresh_results.md), [head32](docs/head32_results.md) |
| Fast tracer alone | Output bitwise identical on all 624 traces checked (312 per number format); **1.12×** per step in float32, 1.43× in `fp16_head32` (benchmark seeds, same jobs). | [fasttrace](docs/fasttrace_results.md) |
| Single-pass 3D ViT, trained on crops 0–11 | Mean F1 0.531 on the tuning crop 12 (457 objects), against 0.924 for the same pipeline with labels as output. Training stopped itself at step 67,344 of 170,000; the step-66,000 checkpoint is marked unstable. Stopped by the owner before the test crops, which were never segmented. | [vitseg](docs/vitseg_results.md) |
| SegNeuron, a pretrained EM model, zero-shot | Mean F1 on crop 12 at native resolution: 0.500 (our pipeline), 0.530 (the authors' FRMC with predicted background removed; 0.553 as released). Foreground AUC 0.917; same- vs different-neuron AUC 0.67–0.69. Upsampling 2× was worse. | [segneuron](docs/segneuron_results.md) |

Caveats that go with these numbers:

- **The 2% pass is pooled over axons and dendrites.** All 5 new merges on the fresh seeds are in
  dendrites (5 of 114, 4.4%; one-sided 95% upper bound 9.0%), and merged voxels rise from
  2.93 M to 5.47 M. Without the fast tracer, `fp16_head32` is 2.21× faster per step and 2.07×
  per whole trace on the fresh seeds, because the merges add steps. The first 300 seeds
  (1 new merge, 1 new early stop) were partly used to choose `fp16_head32`.
- **Fast-tracer identity:** for 12 of the 624 traces (`fp16_head32`, benchmark seeds) no earlier
  recorded run exists, so identity rests on the comparison inside the job. Identity was observed
  on V100s with deterministic XLA ops; per-step times exclude per-trace setup.
- **Tuning-crop numbers are optimistic:** for the ViT and SegNeuron, crop 12 chose the checkpoint
  or settings and is also where they are scored. The skeleton-based run length (ERL) on the fresh
  seeds failed its pre-registered sanity gate and is withheld.

Other ideas that failed the accuracy rules (skipping or postponing moves, plain float16, bfloat16,
dropping residual blocks) are recorded in `docs/*_results.md`; the webpage shows them side by
side.

## How results are checked

Most milestones have a design (`docs/*_design.md`), a results document (`docs/*_results.md`), an
evidence file (`docs/*_evidence.json`) and an evidence test (`tests/test_*_evidence.py`); a few
are organised differently (for example `docs/gpu_baseline.md`, `docs/tolerance2_results.md` with
`tests/test_tolerance2.py`). An evidence test checks, against the run records in the data cache,
that the code that ran matches this repository (by sha256, or for disclosed post-run edits by
comparing definitions), that the outputs are the recorded ones, and that the numbers in the
evidence file follow from them. Numbers a results document marks as outside the evidence (for
example some run times) are not checked.

The work was done with Claude Code under a review workflow (ResearchFlow): each milestone had a
written task contract, an independent method check by a second model and an evidence review by
a third; the human owner made the research decisions. The webpage summarises the process.

## Repository layout

| path | contents |
|---|---|
| `liconn_v2/` | The library: reference inference (`reference.py`), benchmark and scoring (`benchmark.py`), number formats (`precision.py`, `earlystop.py`), movement policies (`policies.py`), fast tracer (`fasttrace.py`), seed sets and verdicts (`manyseed.py`, `fresh.py`, `tolerance2.py`), ViT (`vitseg.py`), SegNeuron evaluation (`segneuron.py`), data cache (`data.py`). |
| `examples/` | Command-line entry points: `trace_reference.py`, `run_benchmark.py`, `run_manyseed.py` / `judge_manyseed.py`, `run_fresh.py` / `judge_fresh.py`, `run_fasttrace.py`, `train_vitseg.py` / `run_vitseg.py`, `run_segneuron.py`, ... |
| `tests/` | Unit tests and evidence tests. |
| `docs/` | Designs, results and evidence for each milestone. |
| `cluster/wahab/` | Slurm job scripts and setup notes for ODU's Wahab cluster (V100 GPUs). |
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
.venv/bin/python -m liconn_v2.data            # one-time download of the public sample crop and checkpoint, sha256-pinned
.venv/bin/python -m unittest discover -s tests -t .
```

The data cache defaults to `~/ResearchFlow/data/liconn-v2`; set `LICONN_DATA` to move it.
`liconn_v2.data` fetches only the sample crop and checkpoint. Many tests also need the further
crops (held-out, many-seed and fresh crops; about 11 GB and 16 GB, see
`docs/manyseed_design.md` and `docs/fresh_design.md`) and the recorded runs under `runs/` in the
cache; those run records are not published, so the evidence tests and some unit tests pass only
on a machine that holds them. `tools/requirements-viewer.txt` adds the viewer.

## Reproducing

- **Reference trace (CPU):** `.venv/bin/python examples/trace_reference.py` reproduces the
  authors' notebook example; its fingerprint is `liconn_v2/expected_reference.json`.
- **Benchmark:** `examples/run_benchmark.py` traces seeds and scores them against the proofread
  segmentation (`docs/benchmark_design.md`).
- **GPU runs:** set `LICONN_BACKEND=gpu`. The many-seed, fresh-seed, fast-tracer, ViT and
  SegNeuron runs ran on Wahab V100s. The job scripts are in `cluster/wahab/` (setup in
  `cluster/wahab/README.md`), the commands in the designs (`docs/*_design.md`), and each results
  document lists its Slurm jobs.

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
