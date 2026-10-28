# Running liconn_v2 on ODU Wahab

Wahab is ODU's Slurm cluster. The `gpu` and `timed-gpu` partitions have nodes with
4 × V100, and `high-gpu-mem` has nodes with 2 × A100 80 GB. `timed-gpu` jobs are
limited to 2 hours and usually start quickly.

## 1. Log in (each session)

Wahab accepts password + DUO only (no SSH keys), so log in from a terminal.
`~/.ssh/config` on the Mac has:

    Host wahab
        HostName wahab.hpc.odu.edu
        User dwadduwa
        ControlMaster auto
        ControlPath ~/.ssh/cm/%r@%h:%p
        ControlPersist 8h

Run `ssh -fN wahab` in Terminal and approve DUO. Later `ssh`, `scp` and `git push`
calls reuse that connection. The login shell is tcsh; to run bash scripts remotely,
use `ssh wahab bash -s < script.sh`.

## 2. Code

The cluster has no GitHub access. The Mac pushes to a private bare repository on
Wahab instead:

    git push wahab main                                     # on the Mac (remote: wahab:repos/liconn_v2.git)
    git -C /scratch/$USER/liconn_v2 pull                     # on Wahab

## 3. Environment and data (once)

    bash cluster/wahab/setup.sh

This builds `~/envs/liconn_v2` (Python 3.12.8 via uv; requirements.txt +
requirements-cuda.txt + FFN at 12d680e) and downloads the public LICONN data to
`/scratch/$USER/liconn_v2_data`, checked against the pinned sha256.
requirements-cuda.txt pins cuDNN 9.6 / CUDA 12.6. The newest cuDNN (9.27) fails on
the V100s with cuDNN status 5003.

## 4. Run the benchmark on GPUs

    export RUN=/scratch/$USER/liconn_runs/gpu-baseline    # keep RUN for the merge step
    mkdir -p $RUN/logs
    sbatch --export=ALL -o "$RUN/logs/%A_%a.out" cluster/wahab/benchmark_array.sbatch
    squeue -u $USER
    ~/envs/liconn_v2/bin/python examples/merge_benchmark.py $RUN/merged $RUN/parts/*

Copy `$RUN/merged` back to the Mac's `data/runs/` and compare it with another run:

    .venv/bin/python examples/compare_runs.py data/runs/baseline-20261018 data/runs/<gpu-run>

Jobs set `LICONN_BACKEND=gpu` and `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. Each
record stores the GPU model, node, Slurm job, XLA flags and library versions.

## 5. Many-seed runs (docs/manyseed_design.md)

The new crops are fetched once on the Mac (`.venv/bin/python -m liconn_v2.manyseed`)
and copied to `/scratch/$USER/liconn_v2_data/manyseed/`. One array task runs one crop
(up to 30 seeds); set the array to the number of crops in docs/manyseed_crops.json:

    export RUN=/scratch/$USER/liconn_runs/manyseed-fp32 VARIANT=fp32; mkdir -p $RUN/logs
    sbatch --export=ALL --array=0-12 -o "$RUN/logs/%A_%a.out" cluster/wahab/manyseed_array.sbatch

Repeat with `VARIANT=fp16` and its own RUN, then judge (crops in order):

    ~/envs/liconn_v2/bin/python examples/judge_manyseed.py \
        --ref  $REF/crop-{00..12} --cand $RUN/crop-{00..12}
