"""Tables built exclusively from run metrics JSON files (no hand-copied numbers)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            continue  # curves are not table cells
        else:
            out[key] = v
    return out


def config_hash(path: Path) -> str:
    """Short hash of a run's resolved config with every ``seed`` key removed ("-" if absent), so runs
    that differ only by seed share a hash and runs with different configs never pool."""
    if not path.exists():
        return "-"

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "seed"}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    blob = json.dumps(strip(json.loads(path.read_text())), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def load_metrics(root: str | Path, pattern: str = "**/metrics.json") -> pd.DataFrame:
    rows = []
    for p in sorted(Path(root).glob(pattern)):
        with open(p) as fh:
            m = json.load(fh)
        row = _flatten(m)
        row["run_dir"] = str(p.parent)
        row["config_hash"] = config_hash(p.parent / "config.resolved.json")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """mean ± std over seeds for each group; also reports n (number of runs)."""
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(by)[metrics]
    mean, std, n = g.mean(), g.std(), g.size().rename("n")
    out = mean.copy()
    for m in metrics:
        out[m] = [
            f"{a:.3f} ± {b:.3f}" if pd.notna(b) else f"{a:.3f}" for a, b in zip(mean[m], std[m])
        ]
    return out.join(n).reset_index()


def to_latex(df: pd.DataFrame, path: str | Path, caption: str = "", label: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(index=False, escape=True, caption=caption or None, label=label or None)
    Path(path).write_text(tex)


def to_markdown(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        Path(path).write_text(df.to_markdown(index=False))
    except ImportError:  # tabulate not installed
        Path(path).write_text(df.to_csv(index=False))
