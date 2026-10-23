#!/bin/bash
# Many-seed runs (docs/manyseed_design.md), one crop (30 seeds) per array task.
# Submit from the repository checkout:
#   export RUN=/scratch/$USER/liconn_runs/<name> VARIANT=<fp32|fp16>; mkdir -p $RUN/logs
#   sbatch --export=ALL --array=0-<crops-1> -o "$RUN/logs/%A_%a.out" cluster/wahab/manyseed_array.sbatch
# then judge (fp16 against fp32, crops in order):
#   ~/envs/liconn_v2/bin/python examples/judge_manyseed.py --ref $REF/crop-{00..09} --cand $RUN/crop-{00..09}
#SBATCH -J liconn-manyseed
#SBATCH -p timed-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --array=0-9
set -euo pipefail
: "${RUN:?set RUN to an output directory}"
: "${VARIANT:?set VARIANT (fp32 or fp16)}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$HOME/envs/liconn_v2/bin/python}
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
export LICONN_BACKEND=gpu
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
cd "$REPO"
CROP=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
echo "task $SLURM_ARRAY_TASK_ID -> crop $CROP variant $VARIANT on $(hostname) source $(git describe --always --dirty 2>/dev/null || echo unknown)"
"$PY" -I examples/run_manyseed.py --variant "$VARIANT" --crop "$SLURM_ARRAY_TASK_ID" --out "$RUN/crop-$CROP"
