#!/usr/bin/env bash
# Benchmark generation (CPU, float64). One instance per array task.
#   sbatch --account=<acct> --partition=<cpu_partition> --array=0-K%20 \
#          --export=ALL,GRID=configs/experiment/grid_reduced.yaml scripts/slurm/simulate.sbatch
#SBATCH --job-name=hd-sim
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --open-mode=append
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/ada_env.sh"
activate
python scripts/make_benchmark.py --grid "${GRID:?}" --out "$HD_BENCH" --index "${SLURM_ARRAY_TASK_ID:?}"
