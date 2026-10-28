# Makefile — Sharavathi Corridor Theft Detection Project
# Requires: Python 3.12 (override with `make PYTHON=/path/to/python ...`), Node 20 for frontend

PYTHON ?= python
PYTEST := $(PYTHON) -m pytest
PIP    := $(PYTHON) -m pip

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: install install-dev
install:                  ## Install Python dependencies (CPU PyTorch + pinned requirements.txt)
	$(PIP) install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r sharavathi-theft/requirements.txt

install-dev: install
	$(PIP) install black isort mypy

# ---------------------------------------------------------------------------
# WF1 — pandapower grid
# ---------------------------------------------------------------------------
.PHONY: grid grid-check grid-theft grid-baseline grid-viz

grid: grid-check           ## Build network and run sanity check

grid-check:                ## python -m grid.build --check
	cd sharavathi-theft && $(PYTHON) -m grid.build --check

grid-theft:                ## Inject 20 MW theft at km 150 and check
	cd sharavathi-theft && $(PYTHON) -m grid.build --check \
	    --theft-km 150 --theft-mw 20 --theft-pf 0.9

grid-baseline:             ## Generate data/raw/baseline_states.parquet (200 pts)
	cd sharavathi-theft && $(PYTHON) -m grid.baseline --n 200

grid-viz:                  ## Generate diagnostic figures/ (voltage, loading, theft)
	cd sharavathi-theft && $(PYTHON) -m grid.viz_check \
	    --save figures/ --theft-km 150 --theft-mw 20

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
.PHONY: test test-grid test-fast test-sim test-ml test-future test-rag test-vectordb rag-index

test: test-grid test-sim test-ml test-api test-future test-rag test-vectordb   ## Run all Python tests

test-sim:                  ## Run WF2 tests
	cd sharavathi-theft && $(PYTEST) tests/test_simdata.py -v

test-vectordb:             ## Run local vector-DB tests
	cd sharavathi-theft && $(PYTEST) tests/test_vectordb.py -v

test-rag:                  ## Run 09 RAG advisor tests (offline)
	cd sharavathi-theft && $(PYTEST) tests/test_rag.py -v

rag-index:                 ## (09) Rebuild the RAG advisor index (data/models/rag_index.joblib)
	cd sharavathi-theft && $(PYTHON) -m rag.ingest

test-future:               ## Run WF6 mitigation-stub contract tests
	cd sharavathi-theft && $(PYTEST) tests/test_future.py -v

test-ml:                   ## Run WF3 tests (artifact tests need `make train`)
	cd sharavathi-theft && $(PYTEST) tests/test_ml.py -v

test-grid:                 ## Run WF1 acceptance tests (tests/test_grid.py)
	cd sharavathi-theft && $(PYTEST) tests/test_grid.py -v

test-fast:                 ## Run tests skipping the 100-scale convergence test
	cd sharavathi-theft && $(PYTEST) tests/test_grid.py -v \
	    -k "not test_pf_100_random_scales"

# ---------------------------------------------------------------------------
# Placeholder targets for future workflows
# ---------------------------------------------------------------------------
.PHONY: data train train-fast train-gnn train-tune study-data sensor-study api test-api openapi dev compose vector-index web types test-web e2e all

data:                      ## (WF2) Generate synthetic dataset
	cd sharavathi-theft && $(PYTHON) -m simdata --profile major_only --seed 42 --n-jobs 8

train:                     ## (WF3) Features, baseline, XGBoost, localizers, GNN (3 seeds), metrics.json
	cd sharavathi-theft && $(PYTHON) -m ml.train

train-fast:                ## (WF3) Same without the GNN (~90 s)
	cd sharavathi-theft && $(PYTHON) -m ml.train --skip-gnn

train-gnn:                 ## (WF3 step 5) GNN only (3 seeds) -> gnn.pt, gnn_results.json
	cd sharavathi-theft && $(PYTHON) -m ml.gnn.train

train-tune:                ## (WF3) Same, with Optuna tuning (pip install optuna)
	cd sharavathi-theft && $(PYTHON) -m ml.train --tune

study-data:                ## (WF3 step 6) Generate 3 profiles x 4 noise datasets (slow)
	cd sharavathi-theft && $(PYTHON) -m simdata.study_datasets --n-jobs 16 --skip-existing

sensor-study:              ## (WF3 step 6) Retrain per profile/noise -> sensor_study.json
	cd sharavathi-theft && $(PYTHON) -m ml.sensor_study

api:                       ## (WF4) Start FastAPI backend on :8000 (docs at /docs)
	cd sharavathi-theft/backend && $(PYTHON) -m uvicorn app.main:app --reload --port 8000

test-api:                  ## (WF4) Backend tests
	cd sharavathi-theft/backend && $(PYTEST) tests -v

openapi:                   ## (WF4) Export openapi.json for the frontend type generator
	cd sharavathi-theft/backend && $(PYTHON) scripts/export_openapi.py

dev:                       ## (06) API :8000 + web :5173 with hot reload (Ctrl+C stops both)
	(cd sharavathi-theft/backend && $(PYTHON) -m uvicorn app.main:app --reload --port 8000) & \
	(cd sharavathi-theft/frontend && npm run dev)

compose:                   ## (06) docker compose up --build (api + web)
	cd sharavathi-theft && docker compose up --build

vector-index:              ## (06/09) Rebuild the similar-case vector index
	cd sharavathi-theft && $(PYTHON) -m vectordb.cases

web:                       ## (WF5) Start React dev server on :5173 (needs `make api`)
	cd sharavathi-theft/frontend && npm run dev

types: openapi             ## (WF5) Regenerate TS API types from openapi.json
	cd sharavathi-theft/frontend && npm run gen:types

test-web:                  ## (WF5) Vitest unit/component tests
	cd sharavathi-theft/frontend && npm test

e2e:                       ## (WF5) Playwright smoke (starts api + web itself)
	cd sharavathi-theft/frontend && npx playwright test

all: install grid-baseline grid-viz test
	@echo "WF1 complete. Proceed to WF2 (make data)."

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:                      ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
