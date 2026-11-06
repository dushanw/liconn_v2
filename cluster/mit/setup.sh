#!/bin/bash
# One-time environment for the factorised FFN on MIT Engaging (run on a login node, which has
# internet access): this repository's pinned requirements (CPU JAX: FFN's movement policy) and
# torch 2.7.1 for CUDA 12.8 (B200). Data and the teacher files are copied from the local cache
# (docs/distil_design.md), not fetched here.
set -euo pipefail
ROOT=/home/dushanw/orcd/pool/liconn
ENV=$ROOT/distil/env
SRC=${SRC:-$ROOT/liconn_v2_distil}
UV=$HOME/.local/bin/uv
[ -x "$ENV/bin/python" ] || $UV venv -q --python 3.12.8 "$ENV"
export VIRTUAL_ENV=$ENV
cd "$SRC"
$UV pip install -q -r requirements.txt
$UV pip install -q --no-deps -r requirements-ffn.txt
$UV pip install -q --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1+cu128 triton==3.3.1
"$ENV/bin/python" -m pip --version >/dev/null 2>&1 || $UV pip install -q pip
"$ENV/bin/python" -c "import torch, jax, ffn.inference.movement; print(torch.__version__, jax.__version__)"
