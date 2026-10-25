#!/usr/bin/env python
"""Run HyperDiscover fits on Kaggle GPUs (first-choice GPU source; Ada takes free slots).

One-time setup (by the human): kaggle.com → Settings → API Tokens → "Generate New Token", saved as
~/.kaggle/access_token (chmod 600), or the legacy ~/.kaggle/kaggle.json. Never paste the token into
chat or commit it. The account must be phone-verified for GPUs and internet in kernels.

    python scripts/kaggle/kaggle_job.py upload-code                 # (re)upload source as a PRIVATE dataset
    python scripts/kaggle/kaggle_job.py push --name s4-sanity --grid configs/experiment/grid_sanity.yaml \
        --per-gpu 2 [--stage all --engine library] [--extra "--config configs/train/fast.yaml"]
    python scripts/kaggle/kaggle_job.py push --name gate-sweep --experiment configs/experiment/sanity_s4.yaml \
        --sweep configs/sweeps/gate_saturation.yaml --per-gpu 4
    python scripts/kaggle/kaggle_job.py status --name s4-sanity
    python scripts/kaggle/kaggle_job.py pull --name s4-sanity       # -> runs/kaggle/s4-sanity/output/results/

Everything created on Kaggle is private. Kernels request a T4 by default (`--accelerator` to change).
Kaggle limits (verify current values on kaggle.com): about 30 GPU-hours per week, sessions of
about 9-12 hours. Log the hours in RESEARCH_LOG.md like Ada jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE_ITEMS = ["src", "scripts", "configs", "pyproject.toml", "README.md"]
DATASET_SLUG = "hyperdiscover-src"
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.egg-info", "*.pyc", "*.local.yaml"
)  # no personal configs


def kaggle_bin() -> str:
    exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
    if not Path(exe).exists() and not shutil.which(exe):
        sys.exit("kaggle CLI not found: pip install kaggle")
    return exe


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for p in (
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ):
        if p.exists():
            return json.loads(p.read_text())["username"]
    # new-style tokens (~/.kaggle/access_token or KAGGLE_API_TOKEN): ask the CLI who we are
    r = subprocess.run([kaggle_bin(), "config", "view"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.strip().startswith("- username:"):
            name = line.split(":", 1)[1].strip()
            if name and name != "None":
                return name
    sys.exit(
        "No Kaggle credentials: save your API token as ~/.kaggle/access_token (see this script's docstring)"
    )


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True)


def git_commit() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    if not sha:
        return "uncommitted"
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO).returncode != 0
    return sha + ("-dirty" if dirty else "")


def wait_dataset_ready(user: str, timeout: float = 900.0) -> None:
    """Block until Kaggle has processed the dataset (a push before that may use the old version)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = subprocess.run(
            [kaggle_bin(), "datasets", "status", f"{user}/{DATASET_SLUG}"],
            capture_output=True,
            text=True,
        )
        if r.stdout.strip().endswith("ready"):
            print("dataset ready", flush=True)
            return
        time.sleep(15)
    sys.exit("dataset still processing after 15 min; check kaggle.com before pushing")


def upload_code(a):
    user = username()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        for item in BUNDLE_ITEMS:
            src = REPO / item
            if src.is_dir():
                shutil.copytree(src, stage / item, ignore=IGNORE)
            else:
                shutil.copy2(src, stage / item)
        (stage / "GIT_COMMIT").write_text(git_commit() + "\n")  # no .git in the bundle
        meta = {
            "title": "hyperdiscover-src",
            "id": f"{user}/{DATASET_SLUG}",
            "licenses": [{"name": "other"}],
        }
        (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
        exists = (
            subprocess.run(
                [kaggle_bin(), "datasets", "status", f"{user}/{DATASET_SLUG}"],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        if exists:
            run(
                [
                    kaggle_bin(),
                    "datasets",
                    "version",
                    "-p",
                    str(stage),
                    "-m",
                    a.message,
                    "--dir-mode",
                    "zip",
                ]
            )
        else:
            run(
                [kaggle_bin(), "datasets", "create", "-p", str(stage), "--dir-mode", "zip"]
            )  # private by default
    wait_dataset_ready(user)


def push(a):
    user = username()
    if not a.grid and not a.sweep:
        sys.exit("push needs --grid (fits) or --sweep + --experiment (config sweep)")
    if a.sweep and not a.experiment:
        sys.exit("--sweep needs --experiment")
    job = {
        "grid": a.grid,
        "sweep": a.sweep,
        "experiment": a.experiment,
        "start": a.start,
        "count": a.count,
        "per_gpu": a.per_gpu,
        "extra": a.extra,
        "stage": a.stage,
        "engine": a.engine,
        "baselines": a.baselines,  # with --grid: run these baselines instead of HyperDiscover fits
        "nri_steps": a.nri_steps,
    }
    entry = (
        (REPO / "scripts" / "kaggle" / "kernel_entry.py")
        .read_text()
        .replace("__JOB_JSON__", json.dumps(job))
    )
    kdir = REPO / "runs" / "kaggle" / a.name / "kernel"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "hd_kernel.py").write_text(entry)
    meta = {
        "id": f"{user}/hd-{a.name}",
        "title": f"hd-{a.name}",
        "code_file": "hd_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not a.cpu,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    if not a.cpu:
        meta["machine_shape"] = (
            a.accelerator or "NvidiaTeslaT4"
        )  # T4 x2; "NvidiaTeslaP100" = 1x P100
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    run([kaggle_bin(), "kernels", "push", "-p", str(kdir)])


def status(a):
    run([kaggle_bin(), "kernels", "status", f"{username()}/hd-{a.name}"])


def pull(a):
    out = REPO / "runs" / "kaggle" / a.name / "output"
    out.mkdir(parents=True, exist_ok=True)
    run([kaggle_bin(), "kernels", "output", f"{username()}/hd-{a.name}", "-p", str(out)])
    tar = out / "results.tar.gz"
    if tar.exists():
        shutil.unpack_archive(str(tar), str(out / "results"))
        tar.unlink()
    print("pulled to", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload-code")
    u.add_argument("--message", default="update")
    p = sub.add_parser("push")
    p.add_argument("--name", required=True)
    p.add_argument("--grid", default=None, help="grid YAML path relative to the repo")
    p.add_argument(
        "--sweep", default=None, help="sweep YAML (with --experiment) relative to the repo"
    )
    p.add_argument("--experiment", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--per-gpu", type=int, default=2)
    p.add_argument("--extra", default="")
    p.add_argument("--stage", choices=["fit", "all"], default="fit")
    p.add_argument("--engine", default="library")
    p.add_argument(
        "--baselines", nargs="+", default=None, help="with --grid: baseline methods, e.g. nri"
    )
    p.add_argument("--nri-steps", type=int, default=2000)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--accelerator", default=None, help='e.g. "NvidiaTeslaT4" (default), "NvidiaTeslaP100"'
    )
    for name in ("status", "pull"):
        q = sub.add_parser(name)
        q.add_argument("--name", required=True)
    a = ap.parse_args()
    {"upload-code": upload_code, "push": push, "status": status, "pull": pull}[a.cmd](a)


if __name__ == "__main__":
    main()
