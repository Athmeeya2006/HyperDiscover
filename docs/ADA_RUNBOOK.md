# Ada runbook

The order of operations once the Ada details are available. Each step names its gate (CLAUDE.md
§13) and what to paste into `RESEARCH_LOG.md`. **Nothing here has been run yet.**

## 0. Personal settings: `configs/paths/ada.local.yaml` (git-ignored)

`configs/paths/ada.yaml` is the committed template. Your real values go in
`configs/paths/ada.local.yaml`, which git ignores; `scripts/slurm/ada_env.sh` reads the local file
when it exists and exits with code 2 while any required value is still `TODO`. The local file was
pre-filled on 2026-10-25 from the Ada user guide:
- **Account and QoS:** `research`, QoS `medium` (at most 4 GPUs, 40 CPUs and 4 running jobs per user).
- **Partitions:** `long` for jobs (no time limit), `short` for debugging (6 h).
- **GPU requests:** plain `gpu:N`. 1080 Ti nodes are gnode01–40 and 2080 Ti nodes are gnode43–92; pick one with `--nodelist`.
- **CPU ratio:** about 10 CPUs per GPU.

Check the values marked `VERIFY`. Never put a password in any file. Use SSH keys instead
(`ssh-copy-id athmeeya.kashyap@ada-gw1.iiit.ac.in`); Ada's policy forbids sharing an account.
Other automated jobs may already be running under your account. `squeue -u $USER` is the source
of truth, and HyperDiscover only uses free slots.

| key | what it is |
|---|---|
| `slurm.account`, `slurm.partition`, `slurm.cpu_partition` | from `sacctmgr show assoc user=$USER` / `sinfo` |
| `slurm.gpu_type` | the gres type string, e.g. as listed in `sinfo -o "%P %G %N"` |
| `paths.repo`, `paths.conda_sh`, `paths.conda_env` | checkout location and conda init script |
| `paths.shared_storage`, `paths.runs`, `paths.benchmarks` | outputs (not `$HOME` if quota is small) |
| `paths.julia_depot` | `JULIA_DEPOT_PATH` on shared storage (compute nodes are offline) |
| `paths.gpu_claims` | the shared `GPU_CLAIMS.md`, outside both repos |

Budget (confirmed): 4 GPUs in total, shared with TipNet-H and your other automated jobs. **Kaggle comes first** (see the README's Kaggle section); Ada takes work only when slots are free. Still to ask: are CPU-only jobs
limited too?

## 1. Phase 0: environment (login node: installs only)

```bash
ldd --version | head -1; nvidia-smi; sinfo -o "%P %G %N" | head     # paste into RESEARCH_LOG
bash scripts/slurm/install_ada.sh                                   # conda env + Julia/PySR precompile
```

Then on compute nodes (no internet), once on a 1080 Ti node and once on a 2080 Ti node:

```bash
srun --account=<acct> --partition=<part> --gres=gpu:<1080ti-type>:1 --time=00:20:00 \
     bash -lc 'source scripts/slurm/ada_env.sh && activate && python scripts/check_env.py --pysr'
```

**Gate:** both print `ALL PASS` (arch sm_61 / sm_75 in the torch arch list, CUDA
forward/backward, offline PySR recovers sin(x)). If torch 2.4.1 wheels fail on glibc 2.27,
try 2.4.1+cu118 before anything newer, and record the decision.

## 2. Phase 1: benchmark v0.1 (CPU)

```bash
N=$(python scripts/make_benchmark.py --grid configs/experiment/grid_reduced.yaml --list | tail -1 | awk '{print $2}')
sbatch --account=<acct> --partition=<cpu_part> --array=0-$((N-1))%20 \
       --export=ALL,GRID=configs/experiment/grid_reduced.yaml scripts/slurm/simulate.sbatch
python scripts/make_benchmark.py --grid configs/experiment/grid_reduced.yaml --out $HD_BENCH   # writes DATASHEET.md
```

**Gate:** `pytest tests/test_systems.py` passes on Ada, and `DATASHEET.md` exists.
The generation is deterministic, so the local v0.1 files and Ada's are identical for
the same code version.

## 3. Phase 2: baselines (CPU)

```bash
python scripts/validate_this.py --this-dir <THIS-v2.0> --julia <julia> --out $HD_RUNS/validate_this
for f in $HD_BENCH/*_N20_*.h5; do python scripts/run_baselines.py --benchmark-file $f --out $HD_RUNS/baselines/$(basename $f .h5); done
python scripts/evaluate.py --runs $HD_RUNS --baselines $HD_RUNS/baselines --out paper
```

**Gate:**
- The SINDy oracle on S4 at N=10 is exact. This already passes locally: 0.24% max coefficient error.
- The THIS port matches the official code. This passes locally at about 1e-15. The paper example is still open; see DECISIONS D-022.
- A baseline table exists on small N.

## 4. Before any GPU job

1. Run `python scripts/profile_memory.py --device cuda --N 20 100 500 1000` on each GPU type. This sets `--per-gpu` and the batch size, and checks the K_t caps.
2. Follow the coordination protocol by running `bash scripts/slurm/submit_fit.sh <grid> <n_gpus>`. This is a dry run that prints `squeue`, the claims file and the exact `sbatch` command. Append the claim line to `GPU_CLAIMS.md`, then re-run with `--yes` only if the human allows submission.
3. For many small fits, prefer `scripts/slurm/fit_gpu_packed.sbatch` (several fits per GPU). Systems with N ≤ 50 go to `fit_cpu.sbatch` (CLAUDE.md §3.5).

## 5. Phases 3–4: tuning + training (the part deliberately left for Ada)

The local pilot (RESEARCH_LOG 2026-10-24) fails the S4 N=10 sanity gate because the gates do
not saturate. Fix that first. The gate is a CPU-sized case, so iterate with `fit_cpu.sbatch`
or interactively on a compute node. Only after the S4 gate passes:
- the Phase-3 pairwise gate (`configs/experiment/pairwise_gate.yaml`);
- the Phase-4 identifiability pilot (schedules a/b/c, ANOVA, perturbation test). The human
  chooses the approach.

## 6. Distillation (CPU)

```bash
ls -d $HD_RUNS/*/seed0 | grep -E 'noise(0|0.01|0.05)_' > runs_to_distill.txt   # representative subset
sbatch --account=<acct> --partition=<cpu_part> --array=0-$(( $(wc -l < runs_to_distill.txt) - 1 ))%8 \
       --export=ALL,RUNS_LIST=runs_to_distill.txt scripts/slurm/distill_cpu.sbatch
```

Record which subset was distilled (CLAUDE.md §3.5), and keep the ledger total
(`python scripts/ledger.py --summary $HD_RUNS`) in RESEARCH_LOG at every gate.

## 7. After training: the project dashboard (requested 2026-10-25)

Once real trained results exist (Ada, or Kaggle in the meantime), build the project dashboard.
- **Design:** use the Impeccable design skill (github.com/pbakaus/impeccable).
- **Content:**
  - what HyperDiscover is;
  - what has been done and how;
  - every result, generated from the runs' `metrics.json` and never typed by hand;
  - the visualizations and simulations from `scripts/visualize_pipeline.py`;
  - a reading guide to the repo (`CLAUDE.md`, `docs/API.md`, this runbook, `DECISIONS.md`, `RESEARCH_LOG.md`) and to the key papers, with links checked.
