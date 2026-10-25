#!/usr/bin/env bash
# Stage 1–3 structure fitting, one benchmark instance per array task, ONE GPU per task.
#
# Submit only after the GPU coordination protocol (CLAUDE.md §3.5) — use submit_fit.sh, which
# checks squeue + GPU_CLAIMS.md and fills in account/partition from configs/paths/ada.yaml:
#     bash scripts/slurm/submit_fit.sh <grid.yaml> <n_gpus_claimed>
# Throttle (%K) MUST equal the number of GPUs claimed.
#SBATCH --job-name=hd-fit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
source "$(dirname "$0")/ada_env.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/slurm/ada_env.sh"
activate
GRID="${GRID:?set GRID=<grid yaml>}"
IDX="${SLURM_ARRAY_TASK_ID:?must run as an array job}"
NAME="$(python scripts/make_benchmark.py --grid "$GRID" --list | awk -v i="$IDX" '$1==i {print $2}')"
BENCH="$HD_BENCH/$NAME.h5"
OUT="$HD_RUNS/$NAME/seed0"
[[ -f "$BENCH" ]] || python scripts/make_benchmark.py --grid "$GRID" --out "$HD_BENCH" --index "$IDX"
mkdir -p "$OUT"
# Small systems (N ≤ 50) belong on CPU (CLAUDE.md §3.5); this script is for N ≥ 100.
python scripts/run_pipeline.py --benchmark-file "$BENCH" --stage fit --device cuda --out "$OUT" ${EXTRA_ARGS:-}
python scripts/ledger.py --run "$OUT" --kind gpu
