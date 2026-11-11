#!/bin/bash
# Whole test-crop neurons for the figure (docs/figure_neurons_design.md): every method re-run on the crop of
# each drawn object, verified against the test's records. From the staged source, one task per crop, each
# method in its own process (as each test run did), merged into figure.json:
#   sbatch --array=K1,K2,K3 cluster/mit/figure_neurons.sbatch
# ONE=1 runs all methods in one process instead; that ran out of B200 memory on crop 43 (job 28663157).
#SBATCH -J liconn-figneurons
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 02:00:00
#SBATCH -o /home/dushanw/orcd/pool/liconn/runs/figneurons-20261110/%x-%A_%a.out
set -euo pipefail
ROOT=/home/dushanw/orcd/pool/liconn
SRC=$ROOT/liconn_v2_figneurons
PY=$ROOT/distil/env/bin/python
OUT=$ROOT/runs/figneurons-20261110
K=$SLURM_ARRAY_TASK_ID
test "$K" -ge 41 && test "$K" -le 50
export LICONN_DATA=$ROOT/data
export TF_FORCE_GPU_ALLOW_GROWTH=true  # TensorFlow is imported with FFN but runs nothing
export JAX_PLATFORMS=cpu
cd "$SRC"
sha256sum -c --quiet source_sha256.txt
mkdir -p "$OUT"
cp source_sha256.txt "$OUT/source-${SLURM_ARRAY_JOB_ID}_$K.txt"
echo "crop $K job ${SLURM_ARRAY_JOB_ID}_$K $(hostname)"
run() {
  $PY examples/trace_figure_neurons.py --run-dir "$ROOT/runs/distil-20261031" --test-dir "$ROOT/runs/unseen-20261104" \
    --sn-dir "$ROOT/runs/unseen-b200-segneuron-20261106" --crop "$K" --out "$OUT" "$@"
}
test ! -e "$OUT/crop-$K/figure.json"  # this job builds it; an earlier crop result is never overwritten or deleted
if [ "${ONE:-0}" = 1 ]; then
  run
else
  for m in F1 F5 S10 P2 Q2 segneuron; do run --methods "$m"; done
fi
