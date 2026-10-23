#!/bin/bash
# Fresh-seed runs (docs/fresh_design.md), one fresh crop per array task.
# Submit from the repository checkout, with the array over the fresh crop indices:
#   export RUN=/scratch/$USER/liconn_runs/<name> VARIANT=<fp32|fp16_head32>; mkdir -p $RUN/logs
#   sbatch --export=ALL --array=<first>-<last> -o "$RUN/logs/%A_%a.out" cluster/wahab/fresh_array.sbatch
#SBATCH -J liconn-fresh
#SBATCH -p timed-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
set -euo pipefail
: "${RUN:?set RUN to an output directory}"
: "${VARIANT:?set VARIANT (fp32 or fp16_head32)}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$HOME/envs/liconn_v2/bin/python}
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
export LICONN_BACKEND=gpu
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
cd "$REPO"
CROP=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
echo "task $SLURM_ARRAY_TASK_ID -> fresh crop $CROP variant $VARIANT on $(hostname)"
"$PY" -I examples/run_fresh.py --variant "$VARIANT" --crop "$SLURM_ARRAY_TASK_ID" --out "$RUN/crop-$CROP"
