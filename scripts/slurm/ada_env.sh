#!/usr/bin/env bash
# Source from every sbatch script. Reads configs/paths/ada.yaml and refuses to continue while
# any required value is still TODO (CLAUDE.md §3.4: "Do not guess").
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# real values live in the git-ignored ada.local.yaml; ada.yaml is the committed TODO template
CFG="$REPO_DIR/configs/paths/ada.local.yaml"
[[ -f "$CFG" ]] || CFG="$REPO_DIR/configs/paths/ada.yaml"

yget() {  # yget section.key  (tiny YAML reader, no dependencies; values must be on one line)
  python3 - "$CFG" "$1" <<'PY'
import sys, re
path, key = sys.argv[1], sys.argv[2].split(".")
sec = None
for line in open(path):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    if not line.startswith(" "):
        sec = line.split(":")[0].strip(); continue
    k, _, v = line.strip().partition(":")
    if sec == key[0] and k.strip() == key[1]:
        v = re.sub(r"\s+#.*$", "", v).strip().strip('"').strip("'")
        print(v); break
PY
}

need() {
  local v; v="$(yget "$1")"
  if [[ -z "$v" || "$v" == "TODO" ]]; then
    echo "ERROR: $1 is not set in configs/paths/ada.yaml (fill it in configs/paths/ada.local.yaml)" >&2; exit 2
  fi
  echo "$v"
}

# `need` runs inside $( ), where `exit` only leaves the subshell, so check each value here.
_req() { local v; v="$(need "$1")" || exit 2; v="${v/#\~/$HOME}"; v="${v//\$USER/$USER}"; printf '%s' "$v"; }
HD_CONDA_SH="$(_req paths.conda_sh)" || exit 2
HD_ENV="$(_req paths.conda_env)" || exit 2
HD_RUNS="$(_req paths.runs)" || exit 2
HD_BENCH="$(_req paths.benchmarks)" || exit 2
JULIA_DEPOT_PATH="$(_req paths.julia_depot)" || exit 2
HD_GPU_CLAIMS="$(_req paths.gpu_claims)" || exit 2
export HD_CONDA_SH HD_ENV HD_RUNS HD_BENCH JULIA_DEPOT_PATH HD_GPU_CLAIMS
export JULIA_PKG_OFFLINE=true          # compute nodes have no internet
export PYTHON_JULIAPKG_OFFLINE=yes
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE   # HDF5 locks fail on NFS
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

activate() {
  # conda env if present, else the venv made by install_ada.sh from module u18/python/3.10.2
  if [[ -f "${HD_CONDA_SH/#\~/$HOME}" ]]; then
    # shellcheck disable=SC1090
    source "${HD_CONDA_SH/#\~/$HOME}"
    set +u; conda activate "$HD_ENV"; set -u   # conda hooks reference unset vars
  else
    module load u18/python/3.10.2 2>/dev/null || true
    # shellcheck disable=SC1090
    source "$HOME/venvs/$HD_ENV/bin/activate"
  fi
  cd "$REPO_DIR"
}
