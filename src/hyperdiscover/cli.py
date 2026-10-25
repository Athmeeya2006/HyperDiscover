"""Console entry point: ``hyperdiscover-run --benchmark S4 --N 10`` (wraps scripts/run_pipeline.py)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_pipeline_main() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_pipeline.py"
    if not script.exists():
        sys.exit("scripts/run_pipeline.py not found (the CLI needs a source checkout)")
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
