#!/bin/bash
# Unseen crops 41-50: FFN (fp16_head32, fast tracer) or fine-tuned SegNeuron on one V100 per crop
# (docs/unseen_design.md), as the crop-12 head-to-head (cluster/wahab/headtohead.sbatch). METHOD=ffn
# or METHOD=segneuron: each method is its own job, so a crash or time-out of one cannot stop the other.
# Submit from the repository checkout:
#   export RUN=/scratch/$USER/liconn_runs/unseen-20261104; mkdir -p $RUN
#   for M in ffn segneuron; do   # smoke, crop 11
#     sbatch --array=11 --export=ALL,SMOKE=20,METHOD=$M -o "$RUN/%A_%a-$M.out" cluster/wahab/unseen.sbatch; done
#   for M in ffn segneuron; do   # the test
#     sbatch --array=41-50 --export=ALL,METHOD=$M -o "$RUN/%A_%a-$M.out" cluster/wahab/unseen.sbatch; done
#SBATCH -J liconn-unseen
#SBATCH -p timed-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
set -euo pipefail
: "${RUN:?set RUN}"
K=$SLURM_ARRAY_TASK_ID
OUT=$RUN${SMOKE:+/smoke$SMOKE}/crop-$(printf %02d "$K")
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$HOME/envs/liconn_v2/bin/python}
SN=${SN:-/scratch/$USER/segneuron}
FT=${FT:-/scratch/$USER/liconn_runs/segneuron-ft-20261028}
COMMIT=ea6f0c963a2f60f0f130244da27e2b2d753f8ee9
FT_CKPT_SHA=e726bb43201aefbcf1e19b6feb8be555656f635241c07e3d7803329dbf55f228
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
export LICONN_BACKEND=gpu
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
case "${METHOD:?set METHOD to ffn or segneuron}" in ffn|segneuron) ;; *) echo "METHOD=$METHOD"; exit 2;; esac
echo "unseen crop $K $METHOD SMOKE=${SMOKE:-} on $(hostname)"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/repo-commit-$METHOD.txt"
git status --porcelain > "$OUT/repo-status-$METHOD.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv > "$OUT/gpu-$METHOD.txt"
echo "cpus $SLURM_CPUS_PER_TASK" >> "$OUT/gpu-$METHOD.txt"
if [ "$METHOD" = ffn ]; then
  "$PY" -I examples/run_unseen_wahab.py ffn --crop "$K" --out "$OUT" ${SMOKE:+--smoke $SMOKE} 2>&1 | { grep -v -i 'warning' || true; }
  exit 0
fi
test "$(git -C "$SN/src" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SN/src" status --porcelain)"
echo "$FT_CKPT_SHA  $FT/train/best.ckpt" | sha256sum -c -
"$PY" -I examples/run_unseen_wahab.py prep --crop "$K" --out "$OUT" ${SMOKE:+--smoke $SMOKE}
"$HOME/.local/bin/uv" pip freeze -p "$SN/env-infer/bin/python" > "$OUT/env-infer.txt"
t0=$(date +%s.%N)
(cd "$SN/src/Train_and_Inference" && "$SN/env-infer/bin/python" inference.py --input "$OUT/raw.npy" \
  --checkpoint "$FT/train/best.ckpt" --output-dir "$OUT/infer" --device cuda:0 | { grep -v '^Patch ' || true; })
t1=$(date +%s.%N)
rm "$OUT/raw.npy"  # a copy of the crop's image; the pinned file stays in LICONN_DATA
"$PY" -I examples/run_unseen_wahab.py segneuron --crop "$K" --out "$OUT" ${SMOKE:+--smoke $SMOKE} \
  --infer-wall-seconds "$("$PY" -I -c "print($t1 - $t0)")" --sn-checkpoint "$FT/train/best.ckpt"
