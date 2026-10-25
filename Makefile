PY ?= .venv/bin/python

.PHONY: install test test-all lint format smoke viz gate clean

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

gate:               ## Phase-4 sanity gate (S4, N=10) — currently expected to FAIL (see RESEARCH_LOG)
	$(PY) scripts/run_pipeline.py --experiment configs/experiment/sanity_s4.yaml --out runs/sanity_s4 --visualize

clean:
	rm -rf runs/smoke runs/ci_local .pytest_cache .ruff_cache
