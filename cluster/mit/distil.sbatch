#!/bin/bash
# Factorised FFN, part 2 on MIT Engaging, one B200 (docs/distil_design.md). From the staged source:
#   MODE=check sbatch cluster/mit/distil.sbatch                       # GPU numerics of the teacher
#   MODE=teacher CROPS="0 1 2 3 4 5" sbatch cluster/mit/distil.sbatch  # teacher traces and samples
#   MODE=probe NAME=dilated DIL="1 2 4 1" sbatch cluster/mit/distil.sbatch
#   MODE=train NAME=dilated DIL="1 2 4 1" STEPS=N sbatch cluster/mit/distil.sbatch
#   MODE=trace NAME=dilated sbatch cluster/mit/distil.sbatch           # student traces on crop 11
#   On-policy round (docs/distil_design.md):
#   MODE=onpolicy SOURCE=train-dilated/best.pt CROPS="0 1 2 3 4 5" sbatch cluster/mit/distil.sbatch
#   MODE=train NAME=dilated-r1 DIL="1 2 4 1" STEPS=N INIT=train-dilated/best.pt EXTRA=onpolicy \
#     sbatch --mem=320G cluster/mit/distil.sbatch
#   MODE=trace NAME=dilated-r1 CKPT=last sbatch cluster/mit/distil.sbatch  # CKPT: best (default) or last
#   MODE=moves SOURCE=train-dilated/best.pt SAMPLES=teacher CROPS="11" OUTNAME=moves-r0-teacher11 \
#     [MAXS=N] sbatch cluster/mit/distil.sbatch                         # face-move diagnostics
#   Part 3: MODE=score12 NAME=dilated-r1 [MAXO=N] sbatch cluster/mit/distil.sbatch  # FFN and student on crop 12
#   Timing study: MODE=timing NAME=dilated-r1 [RUNS="S1 S2"] [TAG=a] [MAXO=N] sbatch cluster/mit/distil.sbatch  # crop 11
#   Pruned FFN (docs/prune_design.md); RESUME=1 continues a round from train-NAME/resume.pt:
#   MODE=prune-probe NAME=p4 KEEP="16 17 18 19" DIL="1 2 4 1" sbatch cluster/mit/distil.sbatch
#   MODE=prune-train NAME=p4 KEEP="16 17 18 19" DIL="1 2 4 1" STEPS=400000 [RESUME=1] sbatch cluster/mit/distil.sbatch
#   MODE=onpolicy SOURCE=train-p4/best.pt OUTNAME=onpolicy-p4 CROPS="0 1 2 3 4 5" sbatch cluster/mit/distil.sbatch
#   MODE=prune-train NAME=p4-r1 KEEP="16 17 18 19" DIL="1 2 4 1" STEPS=150000 INIT=train-p4/best.pt \
#     EXTRA=onpolicy-p4 sbatch --mem=320G cluster/mit/distil.sbatch
#   MODE=trace NAME=p4-r1 [CKPT=last] sbatch cluster/mit/distil.sbatch   # either checkpoint kind
#   MODE=timing NAME=dilated-r1 RUNS="F1 F5 S10 P1 P2 Q1 Q2" P8=train-p8-r1/best.pt P4=train-p4-r1/best.pt TAG=e \
#     sbatch cluster/mit/distil.sbatch
#SBATCH -J liconn-distil
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 06:00:00
#SBATCH -o /home/dushanw/orcd/pool/liconn/runs/distil-20261031/%x-%j.out
set -euo pipefail
ROOT=/home/dushanw/orcd/pool/liconn
SRC=$ROOT/liconn_v2_distil
PY=$ROOT/distil/env/bin/python
RUN=$ROOT/runs/distil-20261031
export LICONN_DATA=$ROOT/data
export TF_FORCE_GPU_ALLOW_GROWTH=true  # TensorFlow is imported with FFN but runs nothing; never let it reserve the GPU
export JAX_PLATFORMS=cpu
cd "$SRC"
sha256sum -c --quiet source_sha256.txt
cp source_sha256.txt "$RUN/source-$SLURM_JOB_ID.txt"  # the source this job ran
nvidia-smi > "$RUN/gpu-$SLURM_JOB_ID.txt"
$PY -m pip freeze > "$RUN/env-$SLURM_JOB_ID.txt" 2>&1 || true
echo "MODE=$MODE NAME=${NAME:-} DIL=${DIL:-} STEPS=${STEPS:-} CROPS=${CROPS:-} SOURCE=${SOURCE:-} INIT=${INIT:-} EXTRA=${EXTRA:-} CKPT=${CKPT:-} MAXO=${MAXO:-} RUNS=${RUNS:-} TAG=${TAG:-} KEEP=${KEEP:-} RESUME=${RESUME:-} OUTNAME=${OUTNAME:-} P8=${P8:-} P4=${P4:-} job $SLURM_JOB_ID $(hostname)"
case "$MODE" in
  check)
    $PY examples/check_gpu_port.py --export "$RUN/export" --raw "$LICONN_DATA/raw_xyz_uint8.npy" --out "$RUN/check" ;;
  teacher)
    $PY examples/distil_teacher.py --export "$RUN/export" --out "$RUN/teacher" --crops $CROPS ;;
  probe)
    $PY examples/train_distil.py --export "$RUN/export" --teacher-out "$RUN/teacher" --dilations $DIL \
      --steps 200 --probe --out "$RUN/probe-$NAME" ;;
  train)
    $PY examples/train_distil.py --export "$RUN/export" --teacher-out "$RUN/teacher" --dilations $DIL \
      --steps "$STEPS" --out "$RUN/train-$NAME" ${INIT:+--init "$RUN/$INIT"} ${EXTRA:+--extra "$RUN/$EXTRA"} ;;
  onpolicy)
    $PY examples/distil_onpolicy.py --export "$RUN/export" --teacher-out "$RUN/teacher" \
      --checkpoint "$RUN/$SOURCE" --out "$RUN/${OUTNAME:-onpolicy}" --crops $CROPS ;;
  moves)
    $PY examples/move_diagnostics.py --export "$RUN/export" --checkpoint "$RUN/$SOURCE" \
      --samples "$RUN/$SAMPLES" --crops $CROPS --out "$RUN/diagnostics/$OUTNAME.json" ${MAXS:+--max-samples $MAXS} ;;
  trace)
    $PY examples/trace_student.py --export "$RUN/export" --teacher-out "$RUN/teacher" \
      --checkpoint "$RUN/train-$NAME/${CKPT:-best}.pt" --out "$RUN/trace-$NAME${CKPT:+-$CKPT}" ;;
  score12)
    $PY examples/score_crop12.py --export "$RUN/export" --checkpoint "$RUN/train-$NAME/${CKPT:-best}.pt" \
      --out "$RUN/score12-$NAME${MAXO:+-smoke$MAXO}" ${MAXO:+--max-objects $MAXO} ;;
  timing)
    $PY examples/time_crop11.py --export "$RUN/export" --checkpoint "$RUN/train-$NAME/${CKPT:-best}.pt" \
      --out "$RUN/timing-$NAME${TAG:+-$TAG}${MAXO:+-smoke$MAXO}" ${RUNS:+--runs $RUNS} ${MAXO:+--max-objects $MAXO} \
      ${P8:+--p8 "$RUN/$P8"} ${P4:+--p4 "$RUN/$P4"} ;;
  prune-probe)
    $PY examples/train_pruned.py --export "$RUN/export" --teacher-out "$RUN/teacher" --keep $KEEP --dilations $DIL \
      --steps 200 --probe --out "$RUN/probe-$NAME" ;;
  prune-train)
    $PY examples/train_pruned.py --export "$RUN/export" --teacher-out "$RUN/teacher" --keep $KEEP --dilations $DIL \
      --steps "$STEPS" --out "$RUN/train-$NAME" ${INIT:+--init "$RUN/$INIT"} ${EXTRA:+--extra "$RUN/$EXTRA"} \
      ${RESUME:+--resume} ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
