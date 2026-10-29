#!/bin/bash
# FFN (fp16_head32, fast tracer) against fine-tuned SegNeuron on crop 12 (docs/headtohead_design.md).
# One GPU job, both methods on the same node. Submit from the repository checkout:
#   export RUN=/scratch/$USER/liconn_runs/<name>; mkdir -p $RUN
#   sbatch --export=ALL -o "$RUN/%j.out" cluster/wahab/headtohead.sbatch
#SBATCH -J liconn-headtohead
#SBATCH -p timed-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
set -euo pipefail
: "${RUN:?set RUN}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$HOME/envs/liconn_v2/bin/python}
SN=${SN:-/scratch/$USER/segneuron}
FT=${FT:-/scratch/$USER/liconn_runs/segneuron-ft-20261028}
COMMIT=ea6f0c963a2f60f0f130244da27e2b2d753f8ee9
FT_CKPT_SHA=e726bb43201aefbcf1e19b6feb8be555656f635241c07e3d7803329dbf55f228
FFN_RECORDED=${FFN_RECORDED:-/scratch/$USER/liconn_runs/manyseed-fp16_head32-20261022/crop-12/benchmark.json}
export LICONN_DATA=${LICONN_DATA:-/scratch/$USER/liconn_v2_data}
export LICONN_BACKEND=gpu
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
echo "headtohead on $(hostname)"
test "$(git -C "$SN/src" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SN/src" status --porcelain)"
echo "$FT_CKPT_SHA  $FT/train/best.ckpt" | sha256sum -c -
test -f "$FFN_RECORDED"
nvidia-smi --query-gpu=name,driver_version --format=csv > "$RUN/gpu.txt"
echo "cpus $SLURM_CPUS_PER_TASK" >> "$RUN/gpu.txt"
"$PY" -I examples/run_headtohead.py ffn --out "$RUN" 2>&1 | { grep -v -i 'warning' || true; }
"$HOME/.local/bin/uv" pip freeze -p "$SN/env-infer/bin/python" > "$RUN/env-infer.txt"
t0=$(date +%s.%N)
(cd "$SN/src/Train_and_Inference" && "$SN/env-infer/bin/python" inference.py --input "$FT/native/raw.npy" \
  --checkpoint "$FT/train/best.ckpt" --output-dir "$RUN/infer" --device cuda:0 | { grep -v '^Patch ' || true; })
t1=$(date +%s.%N)
"$PY" -I examples/run_headtohead.py segneuron --out "$RUN" --infer-wall-seconds "$("$PY" -I -c "print($t1 - $t0)")"
"$PY" -I examples/run_headtohead.py compare --out "$RUN" --ffn-recorded "$FFN_RECORDED" \
  --sn-recorded "$FT/segneuron_ft_crop12.json" --sn-recorded-infer "$FT/native/infer"
