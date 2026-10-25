#!/usr/bin/env python
"""Phase-0 environment check (run on a 1080 Ti node AND a 2080 Ti node, and offline for PySR).

    python scripts/check_env.py            # torch / CUDA / arch list / forward-backward
    python scripts/check_env.py --pysr     # + 10 s offline PySR fit of y = sin(x)

Exit code 0 only if every requested check passes. Paste the output into RESEARCH_LOG.md.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time


def check(name, fn):
    t0 = time.time()
    try:
        msg = fn()
        print(f"[PASS] {name}: {msg} ({time.time() - t0:.1f}s)")
        return True
    except Exception as e:  # report and continue
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def glibc():
    out = subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout.splitlines()
    return out[0] if out else "unknown"


def torch_info():
    import torch

    s = f"torch {torch.__version__}, CUDA build {torch.version.cuda}, available={torch.cuda.is_available()}"
    if torch.cuda.is_available():
        s += f", device={torch.cuda.get_device_name(0)}, capability={torch.cuda.get_device_capability(0)}"
        s += f", arch_list={torch.cuda.get_arch_list()}"
    return s


def arch_supported():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (run this on a GPU compute node)")
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"{arch} not in torch arch list {torch.cuda.get_arch_list()}")
    return arch


def forward_backward(device):
    import numpy as np
    import torch

    from hyperdiscover.model.structure_model import StructureModel

    N = 20
    pairs = np.array([(i, j) for i in range(N) for j in range(N) if i != j])
    tris = np.array([(0, 1, 2), (1, 0, 2), (2, 0, 1)])
    m = StructureModel(N, 1, pairs, tris).to(device)
    x = torch.randn(64, N, 1, device=device, dtype=torch.float32)
    y = m(x)
    loss = (y**2).mean() + m.l0_pair() + m.l0_tri()
    loss.backward()
    if device != "cpu":
        torch.cuda.synchronize()
    return f"ok on {device}, output dtype {y.dtype}"


def pysr_fit():
    import numpy as np
    from pysr import PySRRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (200, 1))
    y = np.sin(X[:, 0])
    m = PySRRegressor(
        niterations=40,
        binary_operators=["+", "*"],
        unary_operators=["sin"],
        timeout_in_seconds=10,
        progress=False,
        verbosity=0,
        random_state=0,
    )
    m.fit(X, y)
    best = m.sympy()
    err = float(np.mean((m.predict(X) - y) ** 2))
    if err > 1e-6:
        raise RuntimeError(f"best expression {best} has MSE {err:.2e}")
    return f"recovered {best}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pysr", action="store_true", help="also run the offline PySR check")
    ap.add_argument("--cpu-only", action="store_true", help="skip CUDA checks (laptop / CI)")
    a = ap.parse_args()
    if a.pysr:
        import juliacall  # noqa: F401  (Julia must start before torch is imported)
    print(f"host={platform.node()} python={sys.version.split()[0]} glibc={glibc()}")
    ok = [
        check("torch", torch_info),
        check("forward/backward cpu", lambda: forward_backward("cpu")),
    ]
    if not a.cpu_only:
        ok += [
            check("arch supported", arch_supported),
            check("forward/backward cuda", lambda: forward_backward("cuda")),
        ]
    if a.pysr:
        ok.append(check("pysr offline sin(x)", pysr_fit))
    print("ALL PASS" if all(ok) else "SOME CHECKS FAILED")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
