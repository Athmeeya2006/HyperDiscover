#!/usr/bin/env bash
# Structure fitting for small systems (N ≤ 50) on CPU only — no GPU claim needed.
#   sbatch --account=<acct> --partition=<cpu_partition> --array=0-K%P --export=ALL,GRID=<grid> scripts/slurm/fit_cpu.sbatch
#SBATCH --job-name=hd-fitcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/ada_env.sh"
activate
GRID="${GRID:?set GRID=<grid yaml>}"
IDX="${SLURM_ARRAY_TASK_ID:?must run as an array job}"
NAME="$(python scripts/make_benchmark.py --grid "$GRID" --list | awk -v i="$IDX" '$1==i {print $2}')"
BENCH="$HD_BENCH/$NAME.h5"
OUT="$HD_RUNS/$NAME/seed0"
[[ -f "$BENCH" ]] || python scripts/make_benchmark.py --grid "$GRID" --out "$HD_BENCH" --index "$IDX"
python scripts/run_pipeline.py --benchmark-file "$BENCH" --stage fit --device cpu --out "$OUT" ${EXTRA_ARGS:-}
python scripts/ledger.py --run "$OUT" --kind cpu
