#!/usr/bin/env bash
# One-time Ada environment setup — run on the LOGIN node (compute nodes have no internet; CLAUDE.md §3.2).
#   bash scripts/slurm/ada_setup_env.sh > ~/hd_store/setup.log 2>&1
# Installs Miniforge (no conda on Ada), a python 3.10 env "hyperdiscover", torch 2.4.1+cu121 (wheels include
# sm_61 for the 1080 Ti and sm_75 for the 2080 Ti), the package in editable mode, and PySR's Julia depot on
# shared storage (precompiled here so jobs import PySR offline).
set -euo pipefail
STORE=${HD_STORE:-$HOME/hd_store}
CONDA=${HD_CONDA:-$HOME/miniforge3}
REPO=${HD_REPO:-$HOME/HyperDiscover}
mkdir -p "$STORE"/{runs,bench,julia}

if [ ! -x "$CONDA/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA"
fi
source "$CONDA/etc/profile.d/conda.sh"
conda env list | grep -q '^hyperdiscover ' || conda create -y -n hyperdiscover python=3.10
conda activate hyperdiscover

pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" scipy networkx xgi h5py pandas scikit-learn sympy matplotlib seaborn \
  hydra-core omegaconf tqdm pysindy pysr tensorboard pytest ruff tabulate pyyaml torchdiffeq
pip install -e "$REPO"

# PySR / Julia: depot on shared storage, installed + precompiled on the login node
export JULIA_DEPOT_PATH="$STORE/julia"
python -c "import pysr; print('pysr', pysr.__version__)"
python - <<'PY'
import numpy as np
from pysr import PySRRegressor
x = np.linspace(-3, 3, 200)[:, None]
m = PySRRegressor(niterations=5, binary_operators=["+", "*"], unary_operators=["sin"],
                  progress=False, verbosity=0, timeout_in_seconds=60)
m.fit(x, np.sin(x[:, 0]))
print("pysr ok:", m.sympy())
PY
python -c "import torch; print('torch', torch.__version__, 'arch list', torch.cuda.get_arch_list())"
echo "SETUP DONE"
