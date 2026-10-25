#!/usr/bin/env bash
# Phase-0 installation ON THE ADA LOGIN NODE (light work only: installs + Julia precompile).
# Do not run training here. Afterwards verify offline on compute nodes (1080 Ti AND 2080 Ti):
#   srun -p short -A research --gres=gpu:1 -c 10 --time=00:30:00 --nodelist=gnode01 \
#        bash -lc 'source scripts/slurm/ada_env.sh && activate && python scripts/check_env.py --pysr'
# (gnode01-40 = 1080 Ti; for a 2080 Ti use -p long --nodelist=gnode43 or any gnode43-92)
#
# Uses conda if available (configs/paths/ada.local.yaml: paths.conda_sh); otherwise a plain venv
# from the site module u18/python/3.10.2.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_DIR/scripts/slurm/ada_env.sh"
ldd --version | head -1
unset JULIA_PKG_OFFLINE PYTHON_JULIAPKG_OFFLINE   # the login node has internet
CONDA_SH="${HD_CONDA_SH/#\~/$HOME}"
if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
  conda env create -f "$REPO_DIR/environment.yml" || conda env update -f "$REPO_DIR/environment.yml"
  set +u; conda activate "$HD_ENV"; set -u
else
  echo "conda not found at $CONDA_SH: using module u18/python/3.10.2 + venv at ~/venvs/$HD_ENV"
  module load u18/python/3.10.2
  python3 -m venv "$HOME/venvs/$HD_ENV"
  source "$HOME/venvs/$HD_ENV/bin/activate"
  pip install --upgrade pip
  pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu118
  pip install "numpy<2" -e "$REPO_DIR[sr,baselines,dev]"
fi
mkdir -p "$JULIA_DEPOT_PATH"
# First import installs Julia + SymbolicRegression.jl into JULIA_DEPOT_PATH and precompiles.
python -c "import juliacall; import pysr; print('pysr', pysr.__version__)"
python "$REPO_DIR/scripts/check_env.py" --cpu-only --pysr
