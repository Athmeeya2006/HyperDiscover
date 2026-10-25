#!/usr/bin/env bash
# Config sweep (scripts/run_sweep.py) on CPU: small systems (N ≤ 50) belong on CPU (CLAUDE.md §3.5).
#     EXPERIMENT=configs/experiment/sanity_s4.yaml SWEEP=configs/sweeps/alt_fixed.yaml \
#         sbatch --account=research --partition=u22 scripts/slurm/sweep_cpu.sbatch
# Runs every variant in parallel (WORKERS × THREADS ≤ cpus-per-task); results in $HD_RUNS/sweeps/<name>/.
#SBATCH --job-name=hd-sweep
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
source "$(dirname "$0")/ada_env.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/slurm/ada_env.sh"
activate
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=<experiment yaml>}"
SWEEP="${SWEEP:?set SWEEP=<sweep yaml>}"
WORKERS="${WORKERS:-6}"
THREADS="${THREADS:-2}"
OUT="$HD_RUNS/sweeps/$(basename "$SWEEP" .yaml)"
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname); commit $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  > "$OUT/slurm_job.txt"
# run_sweep skips variants whose metrics.json exists, so a requeued job resumes where it stopped
python scripts/run_sweep.py --experiment "$EXPERIMENT" --sweep "$SWEEP" --out "$OUT" \
  --cpu-workers "$WORKERS" --threads "$THREADS" --engine library
python scripts/ledger.py --run "$OUT" --kind cpu || true
