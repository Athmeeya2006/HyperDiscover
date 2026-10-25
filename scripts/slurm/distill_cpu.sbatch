#!/usr/bin/env bash
# Stage 4–5 symbolic distillation + refit on CPU (PySR/Julia), one fitted run per task.
#   sbatch --account=<acct> --partition=<cpu_partition> --array=0-K%P \
#          --export=ALL,RUNS_LIST=<file with one run dir per line> scripts/slurm/distill_cpu.sbatch
# CLAUDE.md §3.5: distil only a representative subset (one seed per config, noise ≤ 5%),
# 10-minute PySR budget per run, 3 restarts; record which subset was distilled.
# Throttle (%P) to 1-2: each task takes 16 CPUs and QoS medium allows 40 per user.
#SBATCH --job-name=hd-distill
#SBATCH --cpus-per-task=16   # QoS medium caps a user at 40 CPUs: throttle arrays to %1-%2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/ada_env.sh"
activate
RUNS_LIST="${RUNS_LIST:?set RUNS_LIST}"
OUT="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_LIST")"
NAME="$(basename "$(dirname "$OUT")")"
[[ -f "$HD_BENCH/$NAME.h5" ]] || { echo "benchmark $HD_BENCH/$NAME.h5 missing (run simulate.sbatch / the fit first)"; exit 1; }
python scripts/run_pipeline.py --benchmark-file "$HD_BENCH/$NAME.h5" --stage distill --engine pysr --out "$OUT"
python scripts/ledger.py --run "$OUT" --kind cpu
