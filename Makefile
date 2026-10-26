PY ?= .venv/bin/python

.PHONY: install test test-all lint format smoke viz gate clean tables tables-cheap tables-heavy \
	tab-identifiability tab-data-design tab-joint-kick tab-active-kick tab-transfer tab-timevarying \
	tab-delays tab-hidden tab-amortized tab-main

install:            ## local CPU dev install
	python3 -m venv .venv
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -e ".[dev,baselines,demo]"

test:               ## fast unit tests (no PySR, no slow gate)
	$(PY) -m pytest -m "not slow and not pysr"

test-all:           ## everything incl. PySR tests (needs JULIA_DEPOT_PATH) and the S4 gate
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

format:
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

smoke:              ## seconds-long end-to-end run (wiring check)
	$(PY) scripts/run_pipeline.py --experiment configs/experiment/smoke.yaml --out runs/smoke --visualize

viz:                ## animate S1–S5 + Stage 0/1 views, no training
	$(PY) scripts/visualize_pipeline.py --systems-only --out runs/viz_systems

gate:               ## Phase-4 sanity gate (S4, N=10, default data) — passes on recipe data, see RESEARCH_LOG
	$(PY) scripts/run_pipeline.py --experiment configs/experiment/sanity_s4.yaml --out runs/sanity_s4 --visualize

clean:
	rm -rf runs/smoke runs/ci_local .pytest_cache .ruff_cache

# ---- paper tables: one command per table (CLAUDE.md §18.4 P0 reproducibility) ---------------------------
# Every target writes paper/tables/<name>.csv/.md from a fixed seed. "cheap" = minutes on a laptop CPU (no
# training); "heavy" = run on Kaggle/Ada (scripts/kaggle/kaggle_job.py push --script ... / --grid ...).
T := paper/tables

tab-identifiability:   ## pure-interaction fraction of the true h on benchmark v0.1 (no training)
	$(PY) scripts/analyze_identifiability.py --bench runs/benchmarks_v0.1 --pattern "*_N20_noise0_*" --out $(T)

tab-data-design:       ## pure-interaction fraction under 9 data designs (no training)
	$(PY) scripts/data_design_identifiability.py --out $(T)

tab-joint-kick:        ## exhaustive joint-kick detection, S1–S5 (seconds per setting)
	$(PY) scripts/joint_kick_experiment.py --out $(T)

tab-active-kick:       ## budgeted (screened/adaptive/random) joint-kick designs vs exhaustive
	$(PY) scripts/active_kick_experiment.py --Ns 20 50 100 --noises 0.01 0.05 --r0s 4 25 --max-per-target 5 8 --out $(T)

tab-transfer:          ## E12: discovered f,g,h on unseen networks (needs configs/transfer_sources/)
	$(PY) scripts/transfer_experiment.py --graphs er ba ws sbm --Ns 20 50 100 --kinds open closed --out $(T)

tab-timevarying:       ## G4a: rewiring detection with known/discovered f,g,h
	$(PY) scripts/timevarying_experiment.py --systems S1 S2 S3 S4 S5 --noises 0.0 0.01 0.05 --out $(T)

tab-delays:            ## G4b: edges + coupling delays
	$(PY) scripts/delay_experiment.py --out $(T)

tab-hidden:            ## G4c: hidden nodes (trains one model per setting: heavy)
	$(PY) scripts/hidden_nodes_experiment.py --out $(T)

tab-amortized:         ## G2: learned triadic screening, in-distribution + leave-one-family-out (heavy)
	$(PY) scripts/pretrain_amortized.py --out runs/amortized && cp runs/amortized/amortized.* $(T)/

tab-main:              ## E1/E2 tables from pulled run metrics (no training here)
	$(PY) scripts/evaluate.py --runs runs/ --out paper/

tables-cheap: tab-identifiability tab-data-design tab-joint-kick tab-delays tab-timevarying
tables-heavy: tab-active-kick tab-transfer tab-hidden tab-amortized
tables: tables-cheap tables-heavy tab-main
