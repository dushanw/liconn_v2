#!/bin/bash
# SegNeuron zero-shot on crop 12 (docs/segneuron_design.md). Submit from the repository checkout:
#   export RUN=/scratch/$USER/liconn_runs/<name>; mkdir -p $RUN
#   j=$(sbatch --parsable --export=ALL,MODE=infer -p timed-gpu --gres=gpu:1 -t 2:00:00 -o "$RUN/%j.out" cluster/wahab/segneuron.sbatch)
#   sbatch --export=ALL,MODE=post -p main -t 12:00:00 -d afterok:$j -o "$RUN/%j.out" cluster/wahab/segneuron.sbatch
# infer: input volumes, the authors' inference.py at both resolutions, xy2 outputs to the crop grid.
# post: the authors' FRMC_post.py at each beta and resolution, then our pipeline and the scoring.
# SN holds the pinned SegNeuron checkout (src/), weights/, env-infer/ and mamba/ (FRMC env).
#SBATCH -J liconn-segneuron
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
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
cd "$REPO"
echo "segneuron $MODE on $(hostname)"
test "$(git -C "$SN/src" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SN/src" status --porcelain)"
echo "$CKPT_SHA  $CKPT" | sha256sum -c -
case "$MODE" in
  infer)
    "$PY" -I examples/run_segneuron.py prep --out "$RUN"
    "$SN/env-infer/bin/python" -m pip freeze 2>/dev/null > "$RUN/env-infer.txt" || \
      "$HOME/.local/bin/uv" pip freeze -p "$SN/env-infer/bin/python" > "$RUN/env-infer.txt"
    nvidia-smi --query-gpu=name,driver_version --format=csv > "$RUN/gpu.txt"
    for res in native xy2; do
      (cd "$SN/src/Train_and_Inference" && "$SN/env-infer/bin/python" inference.py --input "$RUN/$res/raw.npy" \
        --checkpoint "$CKPT" --output-dir "$RUN/$res/infer" --device cuda:0 | { grep -v '^Patch ' || true; })
    done
    "$PY" -I examples/run_segneuron.py grid --out "$RUN" ;;
  post)
    export MAMBA_ROOT_PREFIX=$SN/mamba
    "$SN/mamba/bin/micromamba" list -n segneuron-postprocess > "$RUN/env-frmc.txt"
    for res in native xy2; do
      d=$RUN/$res/infer; [ "$res" = xy2 ] && d=$RUN/$res/grid
      mkdir -p "$RUN/$res/frmc"
      for beta in $BETAS; do
        "$SN/mamba/bin/micromamba" run -n segneuron-postprocess python "$SN/src/Postprocess/FRMC_post.py" \
          --affinities "$d/affinities.npy" --boundaries "$d/boundaries.tif" \
          --output "$RUN/$res/frmc/beta_$beta.npy" --beta "$beta" 2>&1 | { grep -v -i 'futurewarning\|import elf' || true; }
      done
    done
    "$PY" -I examples/run_segneuron.py score --out "$RUN" ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
