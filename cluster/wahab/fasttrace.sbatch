#!/bin/bash
# Fast tracer runs (docs/fasttrace_design.md). One job per (variant, set).
# Submit from the repository checkout:
#   export RUN=/scratch/$USER/liconn_runs/<name> VARIANT=<fp32|fp16_head32> SET=<benchmark|fresh> [RECORDED=<path>] [COMPARE=1]
#   mkdir -p $RUN; sbatch --export=ALL -o "$RUN/%j.out" cluster/wahab/fasttrace.sbatch
#SBATCH -J liconn-fast
#SBATCH -p timed-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
set -euo pipefail
: "${RUN:?set RUN}"; : "${VARIANT:?set VARIANT}"; : "${SET:?set SET}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$HOME/envs/liconn_v2/bin/python}
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
export LICONN_BACKEND=gpu
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
cd "$REPO"
EXTRA=()
if [ "${COMPARE:-0}" = "1" ]; then EXTRA+=(--compare-reference); fi
if [ -n "${RECORDED:-}" ]; then EXTRA+=(--recorded "$RECORDED"); fi
echo "fasttrace $VARIANT $SET on $(hostname)"
"$PY" -I examples/run_fasttrace.py --variant "$VARIANT" --set "$SET" "${EXTRA[@]}" --out "$RUN"
