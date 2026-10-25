#!/usr/bin/env bash
# Joint-kick structure + observational equations for every YAML in configs/experiment/kick_fit/ (array task = file).
#     sbatch --account=research --partition=u22 --array=0-9%4 scripts/slurm/kick_fit_cpu.sbatch
# Small systems (N ≤ 50) run on CPU (CLAUDE.md §3.5).
#SBATCH --job-name=hd-kickfit
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
source "$(dirname "$0")/ada_env.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/slurm/ada_env.sh"
activate
mapfile -t FILES < <(ls configs/experiment/kick_fit/*.yaml | sort)
EXP="${FILES[$SLURM_ARRAY_TASK_ID]}"
OUT="$HD_RUNS/kick_fit/$(basename "$EXP" .yaml)"
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname)" > "$OUT/slurm_job.txt"
python scripts/kick_then_fit.py --experiment "$EXP" --out "$OUT" --mode hybrid --engine library
python scripts/ledger.py --run "$OUT" --kind cpu || true
