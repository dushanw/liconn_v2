#!/bin/bash
# Float32 FFN traced many objects at once on MIT Engaging, one B200 (docs/fp32_batch_design.md). From the staged source:
#   PART=widths [MAXO=N] sbatch cluster/mit/fp32_batch.sbatch                    # part 1: crop 11, eight runs, one job
#   PART=widths-each [RUNS='H16x1 ...'] [MAXO=N] sbatch -t 01:00:00 cluster/mit/fp32_batch.sbatch  # part 1 runs, each in its own process
#   PART=test sbatch --array=0-9 -t 01:30:00 cluster/mit/fp32_batch.sbatch       # part 2: crop 41 + task, float32 32 at once
#   PART=test SMOKE=20 sbatch --array=0 -t 00:20:00 cluster/mit/fp32_batch.sbatch  # part 2 code on 20 objects of crop 11
#SBATCH -J liconn-fp32batch
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 03:00:00
#SBATCH -o /home/dushanw/orcd/pool/liconn/runs/fp32batch-20261109/%x-%A_%a.out
set -euo pipefail
ROOT=/home/dushanw/orcd/pool/liconn
SRC=$ROOT/liconn_v2_fp32batch
PY=$ROOT/distil/env/bin/python
RUN=$ROOT/runs/distil-20261031
OUT=$ROOT/runs/fp32batch-20261109
TAG=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
export LICONN_DATA=$ROOT/data
export TF_FORCE_GPU_ALLOW_GROWTH=true  # TensorFlow is imported with FFN but runs nothing
export JAX_PLATFORMS=cpu
cd "$SRC"
sha256sum -c --quiet source_sha256.txt
mkdir -p "$OUT"
cp source_sha256.txt "$OUT/source-$TAG.txt"
nvidia-smi > "$OUT/gpu-$TAG.txt"
$PY -m pip freeze > "$OUT/env-$TAG.txt" 2>&1 || true
echo "PART=$PART RUNS=${RUNS:-} MAXO=${MAXO:-} SMOKE=${SMOKE:-} job $TAG $(hostname)"
case "$PART" in
  widths)
    $PY examples/time_widths.py --export "$RUN/export" --out "$OUT/widths${MAXO:+-smoke$MAXO}" ${MAXO:+--max-objects $MAXO} ;;
  widths-each)
    for r in ${RUNS:-H16x1 H16x8 H16x32 H16x64}; do  # a fresh process, so each setting starts with an empty GPU
      $PY examples/time_widths.py --export "$RUN/export" --out "$OUT/widths-each${MAXO:+-smoke$MAXO}/$r" --runs "$r" \
        ${MAXO:+--max-objects $MAXO}
    done ;;
  test)
    if [ -n "${SMOKE:-}" ]; then K=11; else K=$((41 + SLURM_ARRAY_TASK_ID)); fi
    $PY examples/run_unseen_fp32x32.py --run-dir "$RUN" --crop "$K" --out "$OUT/test${SMOKE:+-smoke$SMOKE}" \
      ${SMOKE:+--smoke $SMOKE} ;;
  *) echo "unknown PART=$PART"; exit 2 ;;
esac
