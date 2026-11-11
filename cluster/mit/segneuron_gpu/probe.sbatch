#!/bin/bash
# 200-step SegNeuron fine-tuning timing probe on MIT Engaging. Same command as the repository's
# cluster/wahab/segneuron_ft.sbatch probe mode (train(), batch 4), but WORKERS is a variable.
#   sbatch --export=ALL,WORKERS=6 -p mit_preemptable --gres=gpu:a100:1 -c 18 probe.sbatch
#SBATCH -J liconn-sn-probe
#SBATCH --mem=96G
#SBATCH -t 1:00:00
set -euo pipefail
: "${WORKERS:?set WORKERS}"
D=/home/dushanw/orcd/pool/liconn
SN=$D/segneuron
RUN=$D/runs/segneuron-mit-probe-20261030
OUT=$RUN/probe-b4-w$WORKERS-$SLURM_JOB_ID
CKPT_SHA=3767a3793cbbdbf98041b8c4c270b92d2355213d11605568c003322bb759bef5
cd $D/liconn_v2_segneuron_ft
echo "probe on $(hostname), cpus $SLURM_CPUS_PER_TASK, workers $WORKERS"
test "$(git -C "$SN/src" rev-parse HEAD)" = ea6f0c963a2f60f0f130244da27e2b2d753f8ee9
test -z "$(git -C "$SN/src" status --porcelain)"
echo "$CKPT_SHA  $SN/weights/SegNeuronModel.ckpt" | sha256sum -c -
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
lscpu | grep -E "^Model name"
"$SN/env-train/bin/python" examples/train_segneuron.py --sn-src "$SN/src/Train_and_Inference" \
  --ckpt "$SN/weights/SegNeuronModel.ckpt" --data "$RUN/data" --out "$OUT" --iters 200 --batch 4 \
  --hours 0.5 --workers "$WORKERS" 2>&1 | { grep -v -i 'warning' || true; }
