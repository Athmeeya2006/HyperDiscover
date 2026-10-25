#!/usr/bin/env bash
# GPU submission wrapper that enforces the coordination protocol (CLAUDE.md §3.5).
# It PRINTS the sbatch commands and only submits with --yes (and only if the human allowed it).
#   bash scripts/slurm/submit_fit.sh <grid.yaml> <n_gpus> [--yes] [--per-gpu P]
#
# Submits at most <n_gpus> packed jobs (scripts/slurm/fit_gpu_packed.sbatch), each holding ONE GPU and
# running P concurrent fits over a contiguous slice of the grid. This respects QoS `medium`
# (<= 4 GPUs, <= 4 running / 8 submitted jobs per user) instead of one array task per instance.
set -euo pipefail
GRID="${1:?grid yaml}"; NGPU="${2:?number of GPUs to claim}"; YES="${3:-}"
PER_GPU=4
[[ "${4:-}" == "--per-gpu" ]] && PER_GPU="${5:?}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"          # sbatch --output paths are relative to the submit directory
mkdir -p logs
source "$REPO_DIR/scripts/slurm/ada_env.sh"
activate                # python with PyYAML for make_benchmark --list
ACCT="$(need slurm.account)" || exit 2; PART="$(need slurm.partition)" || exit 2
QOS="$(yget slurm.qos)"; CPG="$(yget slurm.cpus_per_gpu)"; CPG="${CPG:-10}"
QOSARG=""; [[ -n "$QOS" && "$QOS" != "null" ]] && QOSARG="--qos=$QOS"
TOTAL="$(yget budget.total_gpus)"

echo "== squeue -u $USER (source of truth)"
squeue -u "$USER" -o "%.10i %.20j %.8T %.12b" || true
IN_USE=$(squeue -h -u "$USER" -t RUNNING,PENDING -o "%b" | { grep -o 'gpu[^,]*' || true; } \
         | awk -F: '{n=$NF; if (n ~ /^[0-9]+$/) s+=n; else s+=1} END {print s+0}')
echo "== GPUs in use/queued by \$USER jobs: $IN_USE (other automated jobs count too; see $HD_GPU_CLAIMS)"
echo "== $HD_GPU_CLAIMS (last 15 lines)"
tail -n 15 "$HD_GPU_CLAIMS" 2>/dev/null || echo "(claims file not found)"
FREE=$((TOTAL - IN_USE))
if (( NGPU > FREE )); then
  echo "REFUSING: requested $NGPU GPUs but only $FREE of $TOTAL are free." >&2; exit 3
fi
N=$(python scripts/make_benchmark.py --grid "$GRID" --list | grep -vc '^#')
CHUNK=$(( (N + NGPU - 1) / NGPU ))
echo "== $N instances -> $NGPU packed job(s) of <= $CHUNK instances, $PER_GPU fits per GPU"
CMDS=()
for ((q = 0; q < NGPU; q++)); do
  START=$((q * CHUNK)); (( START >= N )) && break
  CMDS+=("sbatch --account=$ACCT --partition=$PART $QOSARG --gres=gpu:1 --cpus-per-task=$CPG \
--export=ALL,GRID=$GRID,START=$START,COUNT=$CHUNK,PER_GPU=$PER_GPU scripts/slurm/fit_gpu_packed.sbatch")
done
printf '%s\n' "${CMDS[@]}"
echo "Append to $HD_GPU_CLAIMS:  $(date +%F) | HyperDiscover | ${#CMDS[@]} GPU(s) | <expected duration> | fit $GRID | ACTIVE"
if [[ "$YES" == "--yes" ]]; then
  for c in "${CMDS[@]}"; do eval "$c"; done
else
  echo "(dry run; re-run with --yes to submit)"
fi
