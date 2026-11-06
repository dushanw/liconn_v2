#!/bin/bash
# Unseen crops 41-50 on MIT Engaging, one B200 per crop and run (docs/unseen_design.md). From the staged source:
#   sbatch --array=0-4 --export=ALL,SMOKE=20 cluster/mit/unseen.sbatch   # smoke: 20 objects of crop 11, each run
#   sbatch --array=0-49 cluster/mit/unseen.sbatch                        # the test: 10 crops x 5 runs
# Task i runs crop 41 + i/5 (crop 11 with SMOKE) and run F1, F5, S10, P2, Q2[i % 5] (examples/run_unseen.py),
# writing OUT/crop-K/RUN.json; each run is its own task, so a crash or time-out of one cannot stop another.
#SBATCH -J liconn-unseen
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 02:00:00
#SBATCH -o /home/dushanw/orcd/pool/liconn/runs/unseen-20261104/%x-%A_%a.out
set -euo pipefail
ROOT=/home/dushanw/orcd/pool/liconn
SRC=$ROOT/liconn_v2_unseen
PY=$ROOT/distil/env/bin/python
RUN=$ROOT/runs/distil-20261031
OUT=$ROOT/runs/unseen-20261104${SMOKE:+/smoke$SMOKE}
I=$SLURM_ARRAY_TASK_ID
RUNS=(F1 F5 S10 P2 Q2)
if [ -n "${SMOKE:-}" ]; then K=11; test "$I" -lt 5; else K=$((41 + I / 5)); fi  # smoke: tasks 0-4 only
NAME=${RUNS[$((I % 5))]}
export LICONN_DATA=$ROOT/data
export TF_FORCE_GPU_ALLOW_GROWTH=true  # TensorFlow is imported with FFN but runs nothing
export JAX_PLATFORMS=cpu
cd "$SRC"
sha256sum -c --quiet source_sha256.txt
mkdir -p "$OUT"
cp source_sha256.txt "$OUT/source-${SLURM_ARRAY_JOB_ID}_$I.txt"
nvidia-smi > "$OUT/gpu-${SLURM_ARRAY_JOB_ID}_$I.txt"
$PY -m pip freeze > "$OUT/env-${SLURM_ARRAY_JOB_ID}_$I.txt" 2>&1 || true
echo "crop $K run $NAME SMOKE=${SMOKE:-} job ${SLURM_ARRAY_JOB_ID}_$I $(hostname)"
$PY examples/run_unseen.py --run-dir "$RUN" --crop "$K" --run "$NAME" --out "$OUT" ${SMOKE:+--smoke $SMOKE}
