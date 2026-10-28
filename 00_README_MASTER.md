# 00 — MASTER SPEC: Sharavathi Corridor Theft Detection & Localization

> Give this file FIRST to Claude Code, then feed the workflow files one by one (01 → 07).
> Each workflow file is self-contained and talks to the others ONLY through `08_DATA_CONTRACTS.md`.

## 1. Scope (what we build now)
- Simulated **Sharavathi-origin 220 kV corridor** (software only, NO real grid control).
- Detect + localize **suspected unauthorized extraction** using physics residuals + ML (XGBoost, optional GNN).
- Interactive web app: FastAPI backend + React frontend.
- **Load balancing / mitigation = NOT implemented**, but the UI tab, API routes, and code interfaces exist as stubs (see `07`).

## 2. Modelling assumptions (must be labeled in UI + report)
- Sharavathi = generation-side slack bus (km 0).
- Default 5-node chain is a **proposed simplified model**, NOT a verified KPTCL replica:

| Node | Role | km (assumed) |
|---|---|---|
| SHARAVATHI | generation / slack | 0 |
| SHIMOGA | intermediate | 110 |
| SIRA_KBCROSS | intermediate (proposed) | 215 |
| NELAMANGALA | intermediate (proposed) | 320 |
| PEENYA | Bengaluru load end | 361 |

- Topology lives in `config/topology.yaml` so it can be edited once verified KPTCL/SLDC data is available.
- Each line is subdivided into **virtual tap buses every 10 km** = candidate theft locations (unmetered).
- Output wording: "suspected unauthorized extraction / unexplained anomaly", never "theft confirmed".

## 3. Repo layout
```
sharavathi-theft/
├─ config/            topology.yaml, sensors.yaml, scenarios.yaml, feature_flags.yaml
├─ grid/              (01) pandapower model
├─ simdata/           (02) scenario + dataset generator
├─ ml/                (03) features, detector, XGBoost, GNN, eval
├─ backend/           (04) FastAPI
├─ frontend/          (05) React + Vite + TS
├─ future/            (07) mitigation / load-balancing stubs
├─ data/              raw/ processed/ models/ (gitignored, parquet)
├─ tests/
├─ Makefile
└─ docker-compose.yml
```

## 4. Workstation setup (do this first)
- Python 3.11, Node 20, Git. Use `uv` or `venv`.
- Python deps: `pandapower numpy pandas pyarrow scipy scikit-learn xgboost networkx pyyaml pydantic fastapi uvicorn[standard] joblib shap pytest`
- Optional GNN: `torch torch_geometric`
- Frontend deps: `react vite typescript plotly.js react-plotly.js reactflow zustand @tanstack/react-query tailwindcss`
- Makefile targets: `make grid`, `make data`, `make train`, `make api`, `make web`, `make all`, `make test`.
- `docker-compose.yml`: services `api` (8000), `web` (5173). Data volume mounted.

## 5. Build order & done-criteria
| # | File | Done when |
|---|---|---|
| 1 | 01_PANDAPOWER_GRID | `python -m grid.build` runs AC PF, converges, losses plausible |
| 2 | 02_SYNTHETIC_DATA | parquet dataset with all classes + ground truth |
| 3 | 03_ML_MODELS | detector + XGB + (GNN) + metrics.json saved |
| 4 | 04_BACKEND_API | all endpoints in contracts return valid JSON |
| 5 | 05_FRONTEND | full demo flow works in browser |
| 6 | 06 (inside 03/04) | sensor placement study endpoint + page |
| 7 | 07_FUTURE_MITIGATION | stubs + disabled UI tab only |

## 6. Rules for Claude Code
1. Never invent line parameters silently — every assumed value carries `source: assumed` in config.
2. Theft MUST be injected as a real `pp.create_load` at a tap bus, then re-run power flow. Never subtract from meter values.
3. Split data by **scenario group** and **leave-location-out**; no leakage.
4. Report metrics only from held-out data. Compare vs residual baseline.
5. Write tests per module; `make test` must pass before moving on.
6. Keep every module importable standalone (workflows get shared as separate zips).
