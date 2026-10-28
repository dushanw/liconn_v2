#!/bin/bash
# One-time environment and data setup on ODU Wahab (run on the login node, which
# has internet access). Usage, from the repository checkout:
#   bash cluster/wahab/setup.sh
set -euo pipefail
ENV=${ENV:-$HOME/envs/liconn_v2}
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

# Our own pinned Python (ODU's python modules are shared containers that get replaced).
python3 -m pip install -q --user "uv==0.5.11"
UV=$HOME/.local/bin/uv
$UV python install 3.12.8
[ -x "$ENV/bin/python" ] || $UV venv -q --python 3.12.8 "$ENV"

export VIRTUAL_ENV=$ENV
cd "$REPO"
$UV pip install -q -r requirements.txt
$UV pip install -q -r requirements-cuda.txt
$UV pip install -q --no-deps -r requirements-ffn.txt

# Public LICONN data, checked against the sha256 values pinned in liconn_v2/data.py.
"$ENV/bin/python" -m liconn_v2.data
echo "Environment: $ENV   Data: $LICONN_DATA"
