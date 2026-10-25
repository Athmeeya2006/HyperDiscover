# HyperDiscover

Joint discovery of **higher-order network structure** (pairwise edges *and* 3-node
hyperedges) and **closed-form governing equations** from node time series.

```
dx_i/dt = f(x_i) + c_i + Σ_j A_ij g(x_i, x_j) + Σ_{j<k} B_ijk h(x_i, x_j, x_k)
```

Given trajectories X ∈ R^{M×T×N×d} and no network, HyperDiscover infers A (who interacts
pairwise), B (which triples interact jointly) and symbolic f, g, h shared by all nodes.

> **Status: research prototype (pre-Phase 1).** All pipeline stages are implemented and
> tested on CPU, but the neural structure learner does **not yet** pass the S4 N=10 sanity
> gate (many false-positive edges; see `RESEARCH_LOG.md`). No results from this repo should
> be cited yet. The project brief is `CLAUDE.md`; design choices are in `DECISIONS.md`.

## Pipeline

| stage | module | what it does |
|---|---|---|
| 0 targets | `data/derivatives.py`, `data/weak_form.py` | Savitzky–Golay / Tikhonov derivatives, or weak-form (integration against test functions) |
| 1 screening | `screen/` | cheap scores to keep top-K candidate pairs (N > 200) and triples (N > 20) |
| 2 model | `model/` | hard-concrete (L0) gate per candidate; shared MLPs f, g, h (h symmetric in j, k) |
| 3 training | `train/` | pairwise → triadic → joint schedule, λ_triadic > λ_pairwise (Occam) |
| 4 distillation | `distill/pysr_runner.py` | PySR on sampled g, h, f messages (subprocess; Julia) |
| 5 refit + selection | `distill/refit.py`, `select.py` | least-squares refit of all constants (float64), BIC + rollout selection |

Benchmarks S1–S5 (`systems/`): higher-order Kuramoto, Lotka–Volterra, nonlinear consensus,
SIS, and a synthetic mutualistic variant; S6 (Rössler, d = 3) is a stretch placeholder.

## Install (local, CPU)

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev,baselines,demo]"      # add ",sr" for PySR
```

PySR needs Julia. The first `import pysr` installs it into `JULIA_DEPOT_PATH`. Set the
variable before running, and keep it set when running distillation:

```bash
export JULIA_DEPOT_PATH=~/.julia_hyperdiscover
python -c "import juliacall, pysr"
```

Ada cluster: see `environment.yml`, `scripts/slurm/install_ada.sh` and
`configs/paths/ada.yaml`. Every value there must be filled in with the human before any job
runs, and the scripts refuse to start while any `TODO` remains.

## Quick start

```python
from hyperdiscover import HyperDiscover, benchmarks

b = benchmarks.load("S4", N=10, noise=0.0, hyper_kind="open", n_hyper=2, M=50, T=200, seed=0)
hd = HyperDiscover(state_dim=1, max_order=3, config="fast").fit(b.data.X, b.data.t)

hd.pairwise_edges_        # [(i, j, score)]
hd.hyperedges_            # [(i, j, k, score)]
hd.equations_             # {"f": sympy, "g": sympy, "h": sympy}
hd.simulate(b.data.X[0, 0], b.data.t)
print(hd.report())        # markdown, with Pareto fronts
```

Presets: `"default"` (paper settings), `"fast"` (CPU-sized, for demos/tests). A YAML path or
dict overrides any key of `pipeline.DEFAULT_CONFIG`.

## See it working (visualisations)

```bash
make viz      # no training: each system animated on its hypergraph + Stage 0/1 views
make smoke    # seconds-long fit + every stage figure + training GIF
python scripts/visualize_pipeline.py --experiment configs/experiment/sanity_s4.yaml --preset fast --out runs/viz_s4
```

Each writes an `index.html` that walks through, in order:
1. dynamics GIF;
2. trajectories;
3. derivative / weak-form targets;
4. screening scores;
5. gate probabilities over training;
6. hypergraph-forming GIF;
7. learned vs true g, h;
8. inferred vs true structure (TP / FP / FN);
9. Pareto fronts;
10. rollout;
11. transition sweep.

Demo app: `python -m hyperdiscover.demo.app` (Gradio).

## Scripts

| script | purpose |
|---|---|
| `scripts/check_env.py [--pysr] [--cpu-only]` | Phase-0 environment gate |
| `scripts/make_benchmark.py --grid G --out D` | benchmark files + `DATASHEET.md` (`--list`, `--index i` for arrays) |
| `scripts/run_pipeline.py` | one run → full run directory (provenance, config, gates, metrics, equations); `--stage fit/distill` splits GPU and CPU work |
| `scripts/run_baselines.py` | SINDy (oracle / generic, with the 5% coefficient check), THIS (port of the official code), ARNI-style, NRI-style (`--methods nri`, trained), correlation, partial correlation, Granger, TE |
| `scripts/validate_this.py` | checks the THIS port against the official Julia code (equal to ~1e-15) |
| `scripts/run_many.py` | packs many fits onto the allocated GPUs (or CPU workers) |
| `scripts/profile_memory.py` | step time and peak memory vs N, used to choose fits-per-GPU and batch size |
| `scripts/evaluate.py --runs R --out paper` | all tables (Markdown + LaTeX) and curves, from metrics JSON only |
| `scripts/visualize_pipeline.py` | visual walkthrough (see above) |
| `scripts/ledger.py` | GPU-h / CPU core-h ledger per job |
| `scripts/slurm/` | `fit_gpu`, `fit_gpu_packed`, `fit_cpu`, `distill_cpu`, `simulate` sbatch files; `submit_fit.sh` enforces the shared-GPU protocol (dry run by default) |

## Benchmark suite

```python
from hyperdiscover import benchmarks
b = benchmarks.load("S4", N=50, noise=0.05, hyper_kind="closed", seed=0)
b.data.X, b.data.t, b.structure, b.truth      # truth: sympy equations, params, structure
benchmarks.save(b, "S4.h5"); benchmarks.load_file("S4.h5")
```

Each instance is reproducible from its arguments and seed. Hyperedge kinds: `open` (no
pairwise edges among the three nodes), `closed` (all three present, the hard case), and
`mixed`. Pairwise-only variants: `pairwise_only=True`.

## GPUs: Kaggle first, then Ada

Kaggle is the interim GPU source. Put your API token in `~/.kaggle/access_token` (chmod 600, never
in the repo). Everything created on Kaggle is private.

```bash
python scripts/kaggle/kaggle_job.py upload-code            # source -> private Kaggle dataset
python scripts/kaggle/kaggle_job.py push --name s4-sanity --grid configs/experiment/grid_sanity.yaml --stage all
python scripts/kaggle/kaggle_job.py status --name s4-sanity
python scripts/kaggle/kaggle_job.py pull --name s4-sanity   # -> runs/kaggle/s4-sanity/output
```

The kernel regenerates its benchmark instances deterministically, runs an environment check on
the GPU, fits with `run_many.py`, and returns run directories in the same format as local runs and
Ada. Ada: see `docs/ADA_RUNBOOK.md`; personal settings live in the git-ignored
`configs/paths/ada.local.yaml`.

## Docs

- `docs/ADA_RUNBOOK.md`: step-by-step for the cluster, once the Ada details are in.
- `docs/API.md`: the estimator, configuration keys, benchmarks and metrics.

## Development

```bash
make test     # fast tests
make lint
pytest -m pysr        # PySR tests (needs JULIA_DEPOT_PATH)
pytest -m slow        # S4 sanity gate (currently fails; that is the open research problem)
```

CI (`.github/workflows/`):
- `ci.yml`: lint and tests on Python 3.10 with torch 2.4.1 (the Ada pins) and on 3.12, plus an end-to-end smoke job.
- `pysr.yml`: weekly or on demand, with Julia.
- `gate.yml`: the sanity gate, manual trigger.

## Licence and citation

Not chosen yet (to be decided by the project owner).
