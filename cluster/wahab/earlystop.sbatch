#!/bin/bash
# Early-stop runs (docs/earlystop_design.md): 20 seeds, one job per mode.
# Submit from the repository checkout:
#   export RUN=/scratch/$USER/liconn_runs/<name> MODE=<fp32-open|fp16|fp32-shift> [SHIFT=<c>]; mkdir -p $RUN
#   sbatch --export=ALL -o "$RUN/%j.out" cluster/wahab/earlystop.sbatch
#SBATCH -J liconn-earlystop
#SBATCH -p timed-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
set -euo pipefail
: "${RUN:?set RUN to an output directory}"
: "${MODE:?set MODE (fp32-open, fp16 or fp32-shift)}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$HOME/envs/liconn_v2/bin/python}
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
export LICONN_BACKEND=gpu
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
MANY=${MANY:-/scratch/$USER/liconn_runs}
cd "$REPO"
case "$MODE" in
  fp32-open)  ARGS=(--variant fp32 --open-loop --manyseed-run "$MANY/manyseed-fp32-20261021") ;;
  fp16)       ARGS=(--variant fp16 --manyseed-run "$MANY/manyseed-fp16-20261021") ;;
  fp32-shift) ARGS=(--variant fp32 --shift "${SHIFT:?set SHIFT}" --manyseed-run "$MANY/manyseed-fp32-20261021") ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "mode $MODE on $(hostname)"
"$PY" -I examples/run_earlystop.py "${ARGS[@]}" --reference-run "$MANY/manyseed-fp32-20261021" --verdict "$MANY/manyseed-verdict-fp16-20261021/manyseed_verdict.json" --out "$RUN"
