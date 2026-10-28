#!/bin/bash
# Held-out confirmation on crop B (docs/heldout_design.md), one seed per array task.
# Submit from the repository checkout:
#   export RUN=/scratch/$USER/liconn_runs/<name> VARIANT=<fp32|fp16>; mkdir -p $RUN/logs
#   sbatch --export=ALL -o "$RUN/logs/%A_%a.out" cluster/wahab/heldout_array.sbatch
# then combine:
#   ~/envs/liconn_v2/bin/python examples/merge_heldout.py $RUN/merged $RUN/parts/*
#SBATCH -J liconn-heldout
#SBATCH -p timed-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-11
set -euo pipefail
: "${RUN:?set RUN to an output directory}"
: "${VARIANT:?set VARIANT (fp32 or fp16)}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$HOME/envs/liconn_v2/bin/python}
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
export LICONN_BACKEND=gpu
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
cd "$REPO"
SEED=$("$PY" - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
import numpy as np
sys.path.insert(0, '.')
from liconn_v2 import heldout
gt = np.ascontiguousarray(np.transpose(heldout.load_proofread_xyz(), [2, 1, 0]))
print([s['name'] for s in heldout.select_seeds(gt)][int(sys.argv[1])])
PY
)
echo "task $SLURM_ARRAY_TASK_ID -> seed $SEED variant $VARIANT on $(hostname) source $(git describe --always --dirty 2>/dev/null || echo unknown)"
"$PY" -I examples/run_heldout.py --variant "$VARIANT" --seed "$SEED" --out "$RUN/parts/$SEED"
