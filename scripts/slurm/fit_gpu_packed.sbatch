#!/usr/bin/env bash
# Packed GPU fitting: one job holding G GPUs (usually 1) runs PER_GPU concurrent fits per GPU over a
# slice of the grid (CLAUDE.md §3.5 "pack many fits onto the one GPU"). Measure memory first with
# scripts/profile_memory.py. Normally submitted by scripts/slurm/submit_fit.sh; by hand:
#   sbatch --account=research --partition=long --qos=medium --gres=gpu:1 --cpus-per-task=10 \
#     --export=ALL,GRID=<grid>,START=0,COUNT=40,PER_GPU=4 scripts/slurm/fit_gpu_packed.sbatch
# (Ada gres has no type names: gpu:N. G must not exceed the GPUs claimed in GPU_CLAIMS.md.)
#SBATCH --job-name=hd-fitpack
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/ada_env.sh"
activate
# five minutes before the time limit: requeue ourselves; fits resume from their checkpoints
trap 'echo "time limit near: requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
rc=0
python scripts/run_many.py --grid "${GRID:?}" --bench-dir "$HD_BENCH" --runs-dir "$HD_RUNS" \
  --start "${START:-0}" --count "${COUNT:-1000000}" --per-gpu "${PER_GPU:-4}" --threads 2 \
  --extra="${EXTRA_ARGS:-}" &
wait $! || rc=$?
# one ledger row for the whole job (GPU-hours = elapsed x GPUs held), not one per fit
python scripts/ledger.py --job-dir "$HD_RUNS/_ledger" --kind gpu --gpus "${SLURM_GPUS_ON_NODE:-1}" \
  --note "packed fit $GRID start=${START:-0} count=${COUNT:-all}" || true
exit $rc
