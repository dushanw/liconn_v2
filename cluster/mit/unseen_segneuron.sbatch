#!/bin/bash
# The fine-tuned SegNeuron on one B200 per crop 41-50 (docs/unseen_b200_segneuron_design.md), run
# exactly as the test's V100 run (cluster/wahab/unseen.sbatch, METHOD=segneuron): the authors'
# inference.py timed by this job, then examples/run_unseen_wahab.py, unchanged. From the staged source:
#   sbatch --array=11 --export=ALL,SMOKE=20 cluster/mit/unseen_segneuron.sbatch   # smoke: crop 11 only
#   sbatch --array=41-50 cluster/mit/unseen_segneuron.sbatch                      # the ten crops
# Each crop is its own task, writing OUT/crop-K/segneuron.json.
#SBATCH -J liconn-unseen-sn
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 02:00:00
#SBATCH -o /home/dushanw/orcd/pool/liconn/runs/unseen-b200-segneuron-20261106/%x-%A_%a.out
set -euo pipefail
ROOT=/home/dushanw/orcd/pool/liconn
SRC=$ROOT/liconn_v2_unseen_b200sn
PY=$ROOT/distil/env/bin/python  # the test's MIT environment: prep and post-processing
SN=$ROOT/segneuron              # src/ (pinned checkout), env-infer/, finetuned/best.ckpt (cluster/mit/setup_segneuron.sh)
FT_CKPT=$SN/finetuned/best.ckpt
COMMIT=ea6f0c963a2f60f0f130244da27e2b2d753f8ee9
FT_CKPT_SHA=e726bb43201aefbcf1e19b6feb8be555656f635241c07e3d7803329dbf55f228
K=$SLURM_ARRAY_TASK_ID
if [ -n "${SMOKE:-}" ]; then test "$K" -eq 11; else test "$K" -ge 41 && test "$K" -le 50; fi
OUT=$ROOT/runs/unseen-b200-segneuron-20261106${SMOKE:+/smoke$SMOKE}/crop-$(printf %02d "$K")
export LICONN_DATA=$ROOT/data
export JAX_PLATFORMS=cpu
export TF_FORCE_GPU_ALLOW_GROWTH=true  # TensorFlow is imported with the scoring code but runs nothing
export PYTHONDONTWRITEBYTECODE=1  # keeps the SegNeuron checkout clean
cd "$SRC"
sha256sum -c --quiet source_sha256.txt
test "$(git -C "$SN/src" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SN/src" status --porcelain)"
echo "$FT_CKPT_SHA  $FT_CKPT" | sha256sum -c -
echo "SegNeuron crop $K SMOKE=${SMOKE:-} job ${SLURM_ARRAY_JOB_ID}_$K on $(hostname)"
# A rerun keeps the earlier attempt, renamed (job and time, so a requeue cannot reuse the name), so that
# only this job's records and result can be judged.
if [ -e "$OUT" ]; then
  KEEP="$OUT.before-${SLURM_ARRAY_JOB_ID}_$K-$(date +%Y%m%dT%H%M%S)"
  test ! -e "$KEEP"
  mv -T "$OUT" "$KEEP"
fi
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
cp source_sha256.txt "$OUT/source_sha256.txt"
sha256sum docs/unseen_b200_segneuron_design.md > "$OUT/design-sha256.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv > "$OUT/gpu-segneuron.txt"
echo "host $(hostname) job ${SLURM_ARRAY_JOB_ID}_$K cpus $SLURM_CPUS_PER_TASK" >> "$OUT/gpu-segneuron.txt"
"$HOME/.local/bin/uv" pip freeze -p "$SN/env-infer/bin/python" > "$OUT/env-infer.txt"
"$PY" -m pip freeze > "$OUT/env-post.txt" 2>&1 || true
"$PY" -I examples/run_unseen_wahab.py prep --crop "$K" --out "$OUT" ${SMOKE:+--smoke $SMOKE}
t0=$(date +%s.%N)
(cd "$SN/src/Train_and_Inference" && "$SN/env-infer/bin/python" inference.py --input "$OUT/raw.npy" \
  --checkpoint "$FT_CKPT" --output-dir "$OUT/infer" --device cuda:0 | { grep -v '^Patch ' || true; })
t1=$(date +%s.%N)
rm "$OUT/raw.npy"  # a copy of the crop's image; the pinned file stays in LICONN_DATA
"$PY" -I examples/run_unseen_wahab.py segneuron --crop "$K" --out "$OUT" ${SMOKE:+--smoke $SMOKE} \
  --infer-wall-seconds "$("$PY" -I -c "print($t1 - $t0)")" --sn-checkpoint "$FT_CKPT"
