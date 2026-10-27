#!/bin/bash
# ViT segmentation pilot (docs/vitseg_design.md). Submit from the repository checkout:
#   export RUN=/scratch/$USER/liconn_runs/<name> MODE=<timing|train|tune|run> [STEPS=N] [PARAMS=<file>]
#   mkdir -p $RUN; sbatch --export=ALL -t <limit> -o "$RUN/%j.out" cluster/wahab/vitseg.sbatch
# timing: STEPS steps, no parameters saved. train: STEPS steps, resumable from $RUN/state.msgpack.
# tune: segment crop 12 only and choose the threshold (no test crop is read).
# run: segment crop 12 and the fresh crops with PARAMS and judge against the recorded FFN runs
# (REFERENCE, REFERENCE_FP32 and FAST_RUN default to the pinned copies in fasttrace-inputs/).
#SBATCH -J liconn-vit
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
set -euo pipefail
: "${RUN:?set RUN}"; : "${MODE:?set MODE}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$HOME/envs/liconn_v2/bin/python}
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
export LICONN_BACKEND=gpu
INPUTS=/scratch/$USER/liconn_runs/vitseg-inputs
cd "$REPO"
echo "vitseg $MODE on $(hostname)"
case "$MODE" in
  timing) "$PY" -I examples/train_vitseg.py --timing --steps "${STEPS:?set STEPS}" --out "$RUN" ;;
  train) "$PY" -I examples/train_vitseg.py --steps "${STEPS:?set STEPS}" --out "$RUN" ;;
  tune) "$PY" -I examples/run_vitseg.py --tune-only --params "${PARAMS:?set PARAMS}" \
         --reference "${REFERENCE:-$INPUTS/fresh-fp16_head32-20261023}" --out "$RUN" ;;
  run) "$PY" -I examples/run_vitseg.py --params "${PARAMS:?set PARAMS}" \
         --reference "${REFERENCE:-$INPUTS/fresh-fp16_head32-20261023}" \
         --reference-fp32 "${REFERENCE_FP32:-$INPUTS/fresh-fp32-20261023}" \
         --fast-run "${FAST_RUN:-$INPUTS/fasttrace-fresh-fp16_head32-20261024.json}" --out "$RUN" ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
