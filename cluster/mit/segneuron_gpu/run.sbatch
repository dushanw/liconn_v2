#!/bin/bash
# B200 batch-size experiment. MODE=check: the repository script, 200 steps, batch 4 (environment check
# against the V100/A100/H100 probes). MODE=batch: train_segneuron_bs.py with BATCH, ITERS, LR, WARMUP, VALID.
#SBATCH -J sn-b200
#SBATCH --mem=128G
#SBATCH -c 16
#SBATCH -t 2:00:00
set -euo pipefail
D=/home/dushanw/orcd/pool/liconn
SN=$D/segneuron
R=$D/runs/segneuron-b200-batch-20261030
PY=$SN/env-train-cu128/bin/python
cd $D/liconn_v2_segneuron_ft
echo "$MODE on $(hostname)"
test "$(git -C "$SN/src" rev-parse HEAD)" = ea6f0c963a2f60f0f130244da27e2b2d753f8ee9
test -z "$(git -C "$SN/src" status --porcelain)"
echo "3767a3793cbbdbf98041b8c4c270b92d2355213d11605568c003322bb759bef5  $SN/weights/SegNeuronModel.ckpt" | sha256sum -c -
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
common=(--sn-src "$SN/src/Train_and_Inference" --ckpt "$SN/weights/SegNeuronModel.ckpt" --data "$R/data" --hours "${HOURS:-1.8}" --workers 6)
if [ "$MODE" = check ]; then
  "$PY" examples/train_segneuron.py "${common[@]}" --out "$R/check-b4-$SLURM_JOB_ID" --iters 200 --batch 4
else
  "$PY" "$R/train_segneuron_bs.py" "${common[@]}" --out "$R/b$BATCH-$SLURM_JOB_ID" --iters "$ITERS" --batch "$BATCH" \
    --base-lr "$LR" --warmup-steps "$WARMUP" --valid-every "$VALID"
fi 2>&1 | { grep -v -i 'warning' || true; }
