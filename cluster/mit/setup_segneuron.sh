#!/bin/bash
# One-time SegNeuron inference environment and checkout on MIT Engaging for
# docs/unseen_b200_segneuron_design.md (run on a login node, which has internet access): the pinned
# SegNeuron commit, and the test's V100 inference versions (numpy 2.5.4, tifffile 2026.10.15,
# torch 2.7.1) with torch's CUDA 12.8 build, which supports the B200. The fine-tuned best.ckpt is
# copied from Wahab into $SN/finetuned/ and checked by sha256 in every job; it is not trained here.
set -euo pipefail
SN=/home/dushanw/orcd/pool/liconn/segneuron
COMMIT=ea6f0c963a2f60f0f130244da27e2b2d753f8ee9
UV=$HOME/.local/bin/uv
mkdir -p "$SN/finetuned"
[ -d "$SN/src/.git" ] || git clone -q https://github.com/yanchaoz/SegNeuron.git "$SN/src"
git -C "$SN/src" checkout -q "$COMMIT"
test "$(git -C "$SN/src" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SN/src" status --porcelain)"
[ -x "$SN/env-infer/bin/python" ] || $UV venv -q --python 3.12.8 "$SN/env-infer"
VIRTUAL_ENV=$SN/env-infer $UV pip install -q --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128 triton==3.3.1 numpy==2.5.4 \
  tifffile==2026.10.15
"$SN/env-infer/bin/python" -c "import numpy, tifffile, torch; print(torch.__version__, torch.version.cuda, numpy.__version__, tifffile.__version__)"
