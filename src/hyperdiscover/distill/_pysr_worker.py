"""Subprocess worker for PySR, launched by ``pysr_runner.run_pysr`` via ``runpy`` (not ``-m``).

Runs in a fresh interpreter that imports juliacall *before* anything else, because Julia
(PythonCall.jl) fails to start in a process that has already imported torch. Reads
``job.npz`` + ``job.json`` from job_dir and writes ``result.json``.
"""

from __future__ import annotations

import inspect  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import juliacall  # noqa: F401  (must be first)
import numpy as np  # noqa: E402


def main(job_dir: str) -> None:
    from pysr import PySRRegressor

    d = Path(job_dir)
    arr = np.load(d / "job.npz")
    job = json.loads((d / "job.json").read_text())
    X, y = arr["X"], arr["y"]
    opts = job["options"]
    sig = set(inspect.signature(PySRRegressor.__init__).parameters)
    kw = {k: v for k, v in opts.items() if k in sig}
    kw["random_state"] = job["seed"]
    if "output_directory" in sig:
        kw["output_directory"] = job["output_directory"]
    if "procs" in sig and "procs" not in kw and job.get("procs"):
        kw["procs"] = job["procs"]
    model = PySRRegressor(**kw)
    model.fit(X, y, variable_names=[f"x{q}" for q in range(X.shape[1])])
    rows = []
    for idx in range(len(model.equations_)):
        try:
            rows.append(
                {
                    "sympy": str(model.sympy(idx)),
                    "complexity": int(model.equations_.iloc[idx]["complexity"]),
                    "loss": float(model.equations_.iloc[idx]["loss"]),
                }
            )
        except Exception as e:  # keep going; record the failure
            rows.append({"error": repr(e)})
    (d / "result.json").write_text(
        json.dumps({"equations": rows, "ignored_options": sorted(set(opts) - sig)})
    )


if __name__ == "__main__":
    main(sys.argv[1])
