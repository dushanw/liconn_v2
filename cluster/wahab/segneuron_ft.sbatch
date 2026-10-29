#!/bin/bash
# SegNeuron fine-tuned on LICONN crops 0-11, scored on crop 12 (docs/segneuron_finetune_design.md).
# Submit from the repository checkout:
#   export RUN=/scratch/$USER/liconn_runs/<name>; mkdir -p $RUN
#   probe: sbatch --export=ALL,MODE=probe -p timed-gpu --gres=gpu:1 -t 1:00:00 -o "$RUN/%j.out" cluster/wahab/segneuron_ft.sbatch
#   j=$(sbatch --parsable --export=ALL,MODE=train,ITERS=N -p gpu --gres=gpu:1 -t 24:00:00 -o "$RUN/%j.out" cluster/wahab/segneuron_ft.sbatch)
#   sbatch --export=ALL,MODE=post -p main -t 6:00:00 -d afterok:$j -o "$RUN/%j.out" cluster/wahab/segneuron_ft.sbatch
# probe: data prep (if absent) and a short training run for speed and memory.
# train: data prep (if absent), training (HOURS budget), the authors' inference.py with best.ckpt on crop 12.
# post: the authors' FRMC_post.py at each beta, then our pipeline and the scoring.
#SBATCH -J liconn-segneuron-ft
#SBATCH -c 8
#SBATCH --mem=96G
set -euo pipefail
: "${RUN:?set RUN}"; : "${MODE:?set MODE}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$HOME/envs/liconn_v2/bin/python}
SN=${SN:-/scratch/$USER/segneuron}
COMMIT=ea6f0c963a2f60f0f130244da27e2b2d753f8ee9
CKPT=$SN/weights/SegNeuronModel.ckpt
CKPT_SHA=3767a3793cbbdbf98041b8c4c270b92d2355213d11605568c003322bb759bef5
BETAS="0.1 0.25 0.4 0.5 0.6 0.75 0.9"
BATCH=${BATCH:-8}
HOURS=${HOURS:-20}
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
echo "segneuron_ft $MODE on $(hostname)"
test "$(git -C "$SN/src" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SN/src" status --porcelain)"
echo "$CKPT_SHA  $CKPT" | sha256sum -c -
prep() { [ -f "$RUN/data/manifest.json" ] || "$PY" -I examples/run_segneuron_ft.py prep --out "$RUN"; }
train() {  # $1 = output dir, $2 = iterations, $3 = hours
  "$HOME/.local/bin/uv" pip freeze -p "$SN/env-train/bin/python" > "$RUN/env-train.txt"
  nvidia-smi --query-gpu=name,driver_version --format=csv > "$RUN/gpu.txt"
  "$SN/env-train/bin/python" examples/train_segneuron.py --sn-src "$SN/src/Train_and_Inference" \
    --ckpt "$CKPT" --data "$RUN/data" --out "$1" --iters "$2" --batch "$BATCH" --hours "$3" --workers 6 \
    2>&1 | { grep -v -i 'warning' || true; }
}
case "$MODE" in
  probe)
    prep
    train "$RUN/probe-b$BATCH" 200 0.5 ;;
  train)
    : "${ITERS:?set ITERS}"
    prep
    train "$RUN/train" "$ITERS" "$HOURS"
    "$HOME/.local/bin/uv" pip freeze -p "$SN/env-infer/bin/python" > "$RUN/env-infer.txt"
    (cd "$SN/src/Train_and_Inference" && "$SN/env-infer/bin/python" inference.py --input "$RUN/native/raw.npy" \
      --checkpoint "$RUN/train/best.ckpt" --output-dir "$RUN/native/infer" --device cuda:0 | { grep -v '^Patch ' || true; }) ;;
  post)
    export MAMBA_ROOT_PREFIX=$SN/mamba
    "$SN/mamba/bin/micromamba" list -n segneuron-postprocess > "$RUN/env-frmc.txt"
    mkdir -p "$RUN/native/frmc"
    for beta in $BETAS; do
      "$SN/mamba/bin/micromamba" run -n segneuron-postprocess python "$SN/src/Postprocess/FRMC_post.py" \
        --affinities "$RUN/native/infer/affinities.npy" --boundaries "$RUN/native/infer/boundaries.tif" \
        --output "$RUN/native/frmc/beta_$beta.npy" --beta "$beta" 2>&1 | { grep -v -i 'futurewarning\|import elf' || true; }
    done
    "$PY" -I examples/run_segneuron_ft.py score --out "$RUN" ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
