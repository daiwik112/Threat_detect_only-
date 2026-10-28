# 04 — WORKFLOW: Backend API (FastAPI)

**Goal:** serve simulation, detection, localization, metrics, sensor study; expose mitigation stubs.
**Depends on:** `01` (grid), `03` (`ml.infer`), `08` contracts. **Port:** 8000.

## Layout
```
backend/
├─ app/main.py           FastAPI app, CORS (5173), lifespan loads models + net once
├─ app/schemas.py        pydantic models = 08_DATA_CONTRACTS API schemas
├─ app/deps.py           settings, feature flags loader
├─ app/routers/
│  ├─ network.py   GET  /api/network
│  ├─ simulate.py  POST /api/simulate
│  ├─ analyze.py   POST /api/detect, POST /api/localize, POST /api/analyze (detect+localize in one)
│  ├─ metrics.py   GET  /api/metrics, GET /api/sensor-study, GET /api/dataset-card
│  ├─ scenarios.py GET  /api/scenarios?label=&limit=, GET /api/scenarios/{id}
│  ├─ stream.py    WS   /ws/replay  (replay a scenario sequence over time)
│  └─ mitigation.py  STUB routes (see below)
├─ app/services/       simulation_service.py, analysis_service.py
└─ tests/
```

## Endpoints
| Method | Path | Behavior |
|---|---|---|
| GET | /api/health | status, model version, dataset hash |
| GET | /api/network | buses, lines, taps, assumptions list |
| POST | /api/simulate | build net (optional theft/fault) → PF → measure → return measurements + expected + ground truth |
| POST | /api/detect | measurements → status, score, residuals, SHAP top contributors, likely cause |
| POST | /api/localize | measurements → est_km, interval, ranked sections/taps |
| POST | /api/analyze | one call: simulate (or take measurements) + detect + localize — used by the UI "Run" button |
| GET | /api/metrics | `metrics.json` |
| GET | /api/sensor-study | `sensor_study.json` |
| GET | /api/scenarios | browse dataset scenarios (for "replay known case") |
| WS | /ws/replay | pushes one scenario per second: measurements + live detection result |
| GET | /api/features | feature flags (`mitigation_enabled`, `gnn_enabled`) |

## Mitigation stubs (`routers/mitigation.py`)
- `POST /api/mitigation/plan` → if `mitigation_enabled=false`: HTTP 501 `{status:"not_implemented", message:"Load balancing & mitigation planned for a future release"}`.
- `GET /api/mitigation/strategies` → static list: `load_balance`, `redispatch`, `shed` each with `enabled:false`.
- Calls `future.mitigation.get_planner()` (see 07) so enabling later needs no router change.

## Requirements
1. Load model artifacts + base `net` once at startup; PF per request must be < 300 ms.
2. Validate theft range: km ∈ [0, 361], magnitude 0–60 MW; clamp & warn.
3. Always include `assumptions` + disclaimer string: "Simulated results; suspected anomalies require field verification."
4. Output language: "suspected unauthorized extraction", never "theft confirmed".
5. Structured logging (request id, latency); `/metrics` optional Prometheus.
6. Auto docs at `/docs`; export `openapi.json` to `frontend/src/api/openapi.json` for TS client generation (`openapi-typescript`).
7. Tests with `httpx.AsyncClient`: each endpoint, bad input (422), non-converging PF (graceful 422), missing model files (503 with hint).

## Run
`uvicorn app.main:app --reload --port 8000` · `make api` · Docker service `api`.

## Acceptance
- Posting `{theft:{km:150, magnitude_mw:20}, sensor_profile:"major_only"}` to `/api/analyze` returns `suspicious`, est_km within the interval, and ground truth for comparison.
- OpenAPI schema matches `08` names exactly.
