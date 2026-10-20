# liconn_v2

Speeding up FFN inference from the LICONN paper (Tavakoli et al., *Nature* 642, 398–410,
2025, doi:10.1038/s41586-025-08985-1) without changing its output. Managed with ResearchFlow.

Data, model weights, caches and virtual environments are kept out of Git (see `.gitignore`).

## Current milestone: exact CPU reference of the authors' example

`liconn_v2/reference.py` runs the authors' notebook
(`google/ffn` `notebooks/jax_ffn_inference_liconn.ipynb` at the paper's commit `12d680e`)
on the CPU. `liconn_v2/expected_reference.json` holds its fingerprint, and any faster
version must reproduce it bitwise. Sources and disclosed deviations: `docs/method_source.md`.

```sh
/Users/dushan/.local/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -r requirements-ffn.txt
.venv/bin/python -m liconn_v2.data          # one-time public download, sha256-pinned
.venv/bin/python -m unittest discover -s tests -t .   # ~2.5 min on an M5 Max CPU
.venv/bin/python examples/trace_reference.py           # writes outputs/
```

The input cache defaults to `~/ResearchFlow/data/liconn-v2` (override with `LICONN_DATA`).

## Benchmark (milestone 2)

`liconn_v2/benchmark.py` traces 12 deterministic seeds and scores each FFN object against
the authors' proofread segmentation. It reports recall, precision, merges, network
evaluations and GFLOP per correctly segmented voxel. Design: `docs/benchmark_design.md`.

```sh
.venv/bin/python examples/run_benchmark.py --only-notebook-seed   # ~2.5 min
.venv/bin/python examples/run_benchmark.py                        # all seeds; long on CPU
```

## 3D viewer

`tools/view_benchmark.py` serves a local Neuroglancer view: the image crop, the FFN
traces of a full benchmark run and the matching proofread objects, with labelled seed
points. Setup and usage are in `tools/requirements-viewer.txt`.

## GPU runs (Wahab)

Set `LICONN_BACKEND=gpu` to run on an NVIDIA GPU. The CPU stays the default, and the
bitwise reference is CPU-only. Cluster setup, job arrays and combining results:
`cluster/wahab/README.md`. Comparing two runs seed by seed: `examples/compare_runs.py`.
GPU baseline and its comparison with the CPU: `docs/gpu_baseline.md`.
