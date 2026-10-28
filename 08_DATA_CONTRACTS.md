# 08 — DATA CONTRACTS (shared by all workflows)

> Every workflow reads/writes ONLY these schemas. Change here first, then update modules.
> **Last updated:** activity console (§13). 06 full-stack additions + vector DB (§12). 09 — RAG mitigation advisor (§11). WF6 — mitigation stubs (§5). WF4 + WF5 — API implemented (§9), dashboard consumes it. WF3 — features, models, metrics.json, infer API (§8). WF2 — dataset.parquet, splits.json, dataset_card.md.


---

## 1. Config files (`config/`)

### topology.yaml — IMPLEMENTED SCHEMA
```yaml
base_kv: 220
tap_spacing_km: 10          # virtual theft-candidate buses every 10 km

buses:
  - {id: SHARAVATHI,   km: 0,   role: slack_gen,  metered: true,  source: assumed}
  - {id: SHIMOGA,      km: 110, role: junction,   metered: true,  source: assumed}
  - {id: SIRA_KBCROSS, km: 215, role: junction,   metered: false, source: assumed}
  - {id: NELAMANGALA,  km: 320, role: junction,   metered: true,  source: assumed}
  - {id: PEENYA,       km: 361, role: load,       metered: true,  source: assumed}

# 2-circuit equivalent (parallel circuits already folded):
#   r and x divided by 2; c multiplied by 2 vs single-circuit values
lines:
  r_ohm_per_km: 0.020         # source: assumed (twin-bundle ACSR, 220 kV, 2-ckt parallel)
  x_ohm_per_km: 0.18          # source: assumed
  c_nf_per_km:  20.0          # source: assumed
  max_i_ka:     1.60          # source: assumed (0.80 kA per circuit × 2 circuits)
  circuits:     1             # already folded — pandapower sees 1 equivalent circuit

loads:
  - {bus: SHIMOGA,      p_mw: 60,  q_mvar: 20, source: assumed}
  - {bus: SIRA_KBCROSS, p_mw: 40,  q_mvar: 12, source: assumed}
  - {bus: NELAMANGALA,  p_mw: 80,  q_mvar: 25, source: assumed}
  - {bus: PEENYA,       p_mw: 180, q_mvar: 60, source: assumed}

# Shunt capacitor banks — reactive compensation (ADDITION vs original spec)
# Required for AC PF convergence on this 361 km corridor at full load.
# All values source: assumed until verified with KPTCL/SLDC data.
shunts:
  - {bus: SHIMOGA,      q_mvar_cap: 20, source: assumed}
  - {bus: SIRA_KBCROSS, q_mvar_cap: 15, source: assumed}
  - {bus: NELAMANGALA,  q_mvar_cap: 30, source: assumed}
  - {bus: PEENYA,       q_mvar_cap: 40, source: assumed}

gen:
  bus:   SHARAVATHI
  vm_pu: 1.05                 # source: assumed (higher setpoint needed for long corridor)
```

**Physics outcomes (verified, WF1):**
- Losses at peak load (scale=1.0): **3.49 %** ✅ (target 1–4 %)
- Voltage range at peak: **1.003–1.050 pu** ✅ (target 0.95–1.05 pu)
- Max line loading: **58.4 %** ✅ (well under 100 %)
- 200/200 PF runs converged across load_scale ∈ [0.5, 1.15] ✅

> **Assumption note for UI/API:** The Ferranti effect causes voltage > 1.05 pu at light loads
> (scale < 0.9) — shunt caps are always-on in this model. A production model would add
> switched capacitors. Surface this in `NetworkResponse.assumptions`.

---

### sensors.yaml
```yaml
profile: major_only            # major_only | major_plus_mid | dense
noise: {vm_pct: 0.2, p_pct: 0.5, q_pct: 1.0, va_deg: 0.05}
profiles:
  major_only:     [SHARAVATHI, SHIMOGA, NELAMANGALA, PEENYA]
  major_plus_mid: [SHARAVATHI, SHIMOGA, SIRA_KBCROSS, NELAMANGALA, PEENYA]
  dense:          all_buses   # expands to all buses + tap buses at runtime
```

---

### scenarios.yaml
```yaml
n_normal:       4000
n_theft:        6000          # magnitude 1–40 MW, any tap bus
n_sensor_bias:  1500
n_sensor_drift: 1000
n_missing:       800
n_line_outage:   500
load_scale_range: [0.5, 1.15]
holdout_locations_km: [45, 135, 255]   # leave-location-out test split
```

---

### feature_flags.yaml
```yaml
mitigation_enabled: false      # flips on Future Work tab (see 07)
gnn_enabled: true
```

---

## 2. Raw data: `data/raw/baseline_states.parquet`

Written by `grid/baseline.py` (WF1). One row per load_scale point (default 200 rows).

| column | type | description |
|--------|------|-------------|
| `load_scale` | float | scaling factor applied to all loads |
| `gen_p_mw` | float | total generation from ext_grid |
| `load_p_mw` | float | total accounted load |
| `total_loss_mw` | float | sum of all line segment losses |
| `loss_pct` | float | `total_loss_mw / gen_p_mw × 100` |
| `min_vm_pu` | float | minimum bus voltage |
| `max_vm_pu` | float | maximum bus voltage |
| `max_line_loading` | float | maximum segment loading % |
| `converged` | bool | PF convergence flag |
| `vm_{bus_idx}` | float | estimated voltage per bus (39 buses → 39 columns) |
| `va_{bus_idx}` | float | voltage angle per bus |
| `p_from_line{idx}` | float | active power entering each segment |
| `p_to_line{idx}` | float | active power leaving each segment |
| `q_from_line{idx}` | float | reactive power entering each segment |
| `q_to_line{idx}` | float | reactive power leaving each segment |
| `loading_line{idx}` | float | segment loading % |
| `pl_line{idx}` | float | segment real-power loss MW |

> Shape: **200 rows × 315 columns** (verified, WF1).

---

## 3. Processed dataset: `data/processed/dataset.parquet` (WF2 writes this)

One row = one simulated scenario.

| column | type | notes |
|--------|------|-------|
| `scenario_id` | str | unique UUID |
| `group_id` | str | split group (same base operating point) |
| `label` | str | `normal` / `theft` / `sensor_bias` / `sensor_drift` / `missing` / `line_outage` |
| `true_km` | float \| NaN | theft tap location (km from SHARAVATHI) |
| `true_tap_id` | str \| NaN | tap bus name e.g. `TAP_150` |
| `magnitude_mw` | float | injected unmetered load (0 if normal) |
| `load_scale` | float | system demand factor |
| `sensor_profile` | str | from sensors.yaml profiles |
| `m_<BUS>_vm` | float | noisy voltage measurement (NaN if missing) |
| `m_<BUS>_va` | float | noisy angle measurement |
| `m_<BUS>_p` | float | noisy bus active power injection |
| `m_<BUS>_q` | float | noisy bus reactive power injection |
| `m_LINE<idx>_p_from` | float | noisy line active power from-side |
| `m_LINE<idx>_p_to` | float | noisy line active power to-side |
| `m_LINE<idx>_q_from` | float | noisy line reactive power from-side |
| `m_LINE<idx>_q_to` | float | noisy line reactive power to-side |
| `f_*` | float | engineered features — computed on the fly by WF3 `ml/features.py`, NOT stored in the parquet (see §8.1) |

**Key invariants (enforced by WF2 tests):**
- `label == "theft"` ↔ `true_km` is not NaN, and a real `pp.create_load` was injected
- `label == "normal"` → `magnitude_mw == 0`
- Train/test split is **by `group_id`**, not random rows (no leakage)
- Holdout set uses `true_km ∈ holdout_locations_km` (leave-location-out)

---

## 4. API schemas (JSON — pydantic model names)

```
NetworkResponse
  buses:       [{id, km, role, metered, lat?, lon?}]
  lines:       [{id, from_bus, to_bus, km_len}]
  taps:        [{id, km, line_id}]
  shunts:      [{bus, q_mvar_cap}]          ← NEW vs original spec
  assumptions: [str]                        ← always populated for assumed params

SimulateRequest
  load_scale:     float                       (clamped to [0.5, 1.15] with a warning)
  theft:          {km, magnitude_mw, pf=0.95} | null   (km clamped [0,361] & snapped to tap; MW clamped [0,60])
  fault:          {type: bias|drift|missing|line_outage, channel?, param?} | null
                    bias/drift/missing: channel = m_<BUS>_<vm|p|q> of a metered bus; param = bias % / drift %/step
                    line_outage: param = line index 0–37
  sensor_profile: "major_only" | "major_plus_mid" | "dense"
  noise_mult:     float = 1.0                 (multiplier on sensors.yaml noise)  ← NEW
  seed:           int

SimulateResponse
  scenario_id:   str
  measurements:  {m_<key>: float | null}    ← ONLY physically metered channels for the profile
  expected:      {m_<key>: float | null}    ← noiseless, theft-free, fault-free twin at the same load
  true_label:    str
  true_km:       float | null               ← snapped tap km
  true_tap_id:   str | null
  true_magnitude_mw: float
  sensor_profile: str
  warnings:      [str]                      ← clamping / snapping notices
  disclaimer:    str

DetectResponse
  status:           "normal" | "suspicious"     (anomaly_score ≥ 0.5)
  anomaly_score:    float  (0–1)
  residuals:        [{section, r_p_mw, r_q_mvar, z}]   section = "SHARAVATHI-SHIMOGA" etc.; values may be null if channels missing
  top_contributors: [{feature, shap}]
  likely_cause:     {extraction: float, sensor_error: float, other: float}

LocalizeResponse
  method:           "xgb_reg" | "section_rank" | "gnn"
  est_km:           float
  interval_km:      [lo, hi]
  ranked_sections:  [{section, prob}]        (from section_rank model, always populated)
  ranked_taps:      [{tap_id, km, prob}]      (top 10, from section_rank model)

MetricsResponse   (= top-level keys of data/models/metrics.json)
  detection:     {precision, recall, f1, fpr, roc_auc, recall_at_fpr_1pct, recall_at_fpr_5pct}   test split
  classification:{confusion (6×6), per_class_recall, labels}
  localization:  {method, split, mae_km, median_km, within_10km, within_25km, within_50km}  test_loc_holdout
  baseline:      {same detection keys + same localization keys, for residual/physics baseline}
  (extra keys in the file: detection_detail, classification_detail, localization_detail,
   robustness, beats_baseline, honest_claim, selected_localizer, artifact_meta)

SensorStudyResponse  (= data/models/sensor_study.json)
  configs: [{name, profile, noise_factor, detection_recall, fpr, mae_km,
             recall_at_fpr_1pct, median_km, within_10km, top3_tap_acc, ...}]
  indistinguishability: {adjacent_pairs[], unresolvable_ranges_km[], ...}
```

Errors: `{error: {code: str, message: str}}` with proper HTTP status codes.

---

## 5. Mitigation placeholder schemas (stub only — see `07_FUTURE_MITIGATION`) — IMPLEMENTED (WF6)

```
MitigationRequest
  scenario_id: str
  strategy:    "load_balance" | "redispatch" | "shed"

MitigationResponse                       POST /api/mitigation/plan
  status:          "not_implemented"     (HTTP 501 while mitigation_enabled=false)
  planned_actions: []
  message:         "Load balancing & mitigation planned for a future release"

StrategiesResponse                       GET /api/mitigation/strategies
  strategies:         [{id, label, enabled: false}]   (load_balance, redispatch, shed)
  mitigation_enabled: bool
```

Python interfaces (`future/`, never imported by `grid/`, `simdata/` or `ml/`; tested):

| Module | Interface | Stub today |
|---|---|---|
| `future/mitigation.py` | `MitigationPlanner.plan(net, anomaly) -> MitigationPlan{status, actions[MitigationAction{kind,target,delta_mw,rationale}], predicted_effect, requires_human_approval=True}` | `NotImplementedPlanner` |
| `future/forecasting.py` | `DemandForecaster.forecast(history, horizon_steps) -> DemandForecast{status, horizon_steps, step_minutes, per_bus_mw, interval_mw}` | `NotImplementedForecaster` |
| `future/overload.py` | `OverloadPredictor.predict(net, horizon_steps) -> OverloadForecast{status, horizon_steps, risks[SegmentRisk{line_id,p_overload,expected_peak_pct}]}` | `NotImplementedOverloadPredictor` |

With the flag on, the router passes a deep copy of the base network and
`{scenario_id, strategy}` to `get_planner().plan(...)`. Enabling a real planner only needs a
change to the `get_*()` factory, not to the API or UI.

---

## 6. WF1 completion checklist

| Task | File | Done | Verified |
|------|------|------|----------|
| 1. Config loader + pydantic validation | `grid/config.py` | ✅ | 5 tests pass |
| 2. Network builder with tap buses | `grid/build.py` | ✅ | 5 tests pass |
| 3. AC power flow wrapper | `grid/flow.py` | ✅ | 5 tests pass |
| 4. Sensor / fault simulation | `grid/sensors.py` | ✅ | 3 tests pass |
| 5. WLS state estimator | `grid/estimator.py` | ✅ | 2 tests pass |
| 6. Baseline parquet sweep | `grid/baseline.py` | ✅ | 2 tests pass |
| 7. Diagnostic plots | `grid/viz_check.py` | ✅ | figures/ generated |
| 8. CLI `python -m grid.build --check` | `grid/__main__.py` | ✅ | runs & prints |
| **ALL TESTS** | `tests/test_grid.py` | ✅ | **27/27 PASSED** |
| Baseline parquet on disk | `data/raw/baseline_states.parquet` | ✅ | 200×315 rows |
| Makefile targets | `Makefile` | ✅ | make grid, make test |

---

## 7. WF2 completion checklist

| Task | File | Done | Verified |
|------|------|------|----------|
| 1. Operating points (daily curve + seasonal + jitter) | `simdata/operating_points.py` | ✅ | 4 tests pass |
| 2. All 6 scenario builders + mixed + small theft | `simdata/scenarios.py` | ✅ | 4 tests pass |
| 3. Parallel generator → `dataset_major_only.parquet` | `simdata/generate.py` | ✅ | 8 tests pass |
| 4. Group + leave-location-out splits → `splits.json` | `simdata/splits.py` | ✅ | 5 tests pass |
| 5. Multi-profile generation | `simdata/profiles.py` | ✅ | (via generate) |
| 6. Dataset card markdown | `simdata/dataset_card.py` | ✅ | 1 test pass |
| 7. CLI `python -m simdata.generate` | `simdata/__main__.py` | ✅ | runs & writes |
| **ALL TESTS** | `tests/test_simdata.py` | ✅ | **23/23 PASSED** |
| `data/processed/dataset_major_only.parquet` | on disk | ✅ | 14,100 scenarios |
| `data/processed/splits.json` | on disk | ✅ | 4 splits |
| `data/processed/dataset_card.md` | on disk | ✅ | full stats |

**Key invariants verified:**
- No duplicate `scenario_id`s ✅
- No `group_id` appears in >1 of train/val/test ✅
- All theft rows have `true_km` + `magnitude_mw` > 0 ✅
- All normal rows have `magnitude_mw == 0` and `true_km == NaN` ✅
- Holdout theft rows (near km 45/135/255) only in `test_loc_holdout` ✅
- Same seed → identical `scenario_id` sequence (reproducible) ✅
- All 6 labels present ✅



---

## 8. WF3 — ML interfaces & artifacts

### 8.1 Features (`ml/features.py`)
`build_feature_matrix(df, baseline_df, profile="major_only")` appends 41 `f_*` columns (major_only).
Only channels a profile **physically meters** are used: `m_<BUS>_*` for metered buses and the
line-end flow **at** a metered bus (major_only: `LINE0_from, LINE10_to, LINE11_from, LINE32_to,
LINE33_from, LINE37_to`). Expected values = `baseline_states.parquet` interpolated at the
**estimated** load scale (median of metered-load / nominal). True `load_scale` and labels are never read.

| group | columns |
|---|---|
| global | `f_load_scale_est`, `f_load_scale_spread`, `f_n_missing`, `f_r_p_total` |
| per section `<A>-<B>` | `f_r_p_`, `f_r_q_`, `f_dp_in_`, `f_dva_`, `f_dvm_`, `f_dva_mw_`, `f_xfrac_` |
| per metered bus | `f_vm_dev_`, `f_va_dev_`, `f_kcl_p_`, `f_kcl_q_` |

NaN is kept for unavailable values (XGBoost handles it).

> ⚠ **Known WF1/WF2 issue:** `grid/sensors.py::measure` emits `m_LINE<i>_*` for **all 38
> segments**, including unmetered tap segments (the "only metered endpoints" filter in the
> comment was never implemented). WF3 ignores those columns (tested). WF4 should not display
> them as measurements. Fix in `grid/sensors.py` + regenerate when convenient.

### 8.2 Artifacts (`data/models/`, written by `make train`)
| file | content |
|---|---|
| `baseline_detector.joblib` | residual-threshold + χ²-style detector (normal-train 99.5th pct) |
| `xgb_classifier.joblib` | binary + 6-class XGBoost, SHAP via `shap_top()` |
| `localizer_xgb_reg.joblib` | XGBRegressor km + split-conformal 80 % interval |
| `localizer_section_rank.joblib` | XGBClassifier over taps → ranked taps / sections |
| `feat_cols.json` | ordered feature list |
| `model_meta.json` | `{version, dataset_hash, profile, seed, trained_at, selected_localizer, gnn}` |
| `metrics.json` | MetricsResponse + details (§4) |
| `eval_arrays.json` | `{xgb_reg_holdout: {true_km[], est_km[], magnitude_mw[]}, …, roc: {xgb, baseline}}` for frontend charts |
| `eval_localization.png`, `eval_detection.png` | report plots |
| `sensor_study.json` | SensorStudyResponse (`make study-data && make sensor-study`) |

Every joblib model has `.artifact_meta` = `{version, dataset_hash, profile, seed, trained_at}`.

### 8.3 Inference API for WF4 (`ml/infer.py`)
```python
from ml.infer import load_models
bundle = load_models()                      # once at startup
bundle.detect(measurements)   -> DetectResponse dict
bundle.localize(measurements, method=None)  -> LocalizeResponse dict   # None = selected_localizer
```
`measurements` = raw `{m_<key>: float | None}` (e.g. `SimulateResponse.measurements`).
Features are recomputed inside, identical to training (tested).

### 8.4 WF3 completion checklist

| Task | File | Done | Verified |
|------|------|------|----------|
| 1. Physics features (metered-only, baseline-interpolated) | `ml/features.py` | ✅ | leakage + tap-channel tests |
| 2. Residual baseline detector | `ml/detector_baseline.py` | ✅ | FPR ≤ 2 % on train normals |
| 3. XGBoost binary + 6-class + SHAP | `ml/xgb_classifier.py` | ✅ | |
| 4. Localization: regression + tap ranking | `ml/localize.py` | ✅ | |
| 5. GNN (GraphSAGE, 3 seeds) | `ml/gnn/` | ✅ | **kept** — holdout MAE 21.7 ± 1.2 km vs XGBoost 23.1; top-3 0.58 vs 0.59; shipped localizer = `gnn` |
| 6. Sensor placement study | `ml/sensor_study.py`, `simdata/study_datasets.py` | ✅ | 12/12 configs; shared study split (`study/splits_study.json`) |
| 7. Evaluation → metrics.json | `ml/evaluate.py` | ✅ | |
| 8. Inference API | `ml/infer.py` | ✅ | schema tests |
| **ALL TESTS** | `tests/test_ml.py` | ✅ | **41 passed** (incl. GNN + artifact/sensor-study schema tests) |


---

## 9. WF4 — API (implemented, `backend/`)

Run: `make api` (→ `uvicorn app.main:app --port 8000`, docs at `/docs`). Schema names in
`backend/app/schemas.py` match §4; `make openapi` writes `frontend/src/api/openapi.json`,
`make types` regenerates `frontend/src/api/schema.d.ts`.

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | `{status: ok|degraded, model_version, dataset_hash, models_loaded{…}, selected_localizer}` |
| GET | `/api/features` | `{mitigation_enabled, gnn_enabled, gnn_available}` |
| GET | `/api/network` | NetworkResponse + `lines[].loading_pct` (base case) + `sections[]` + `disclaimer` |
| POST | `/api/simulate` | SimulateRequest → SimulateResponse |
| POST | `/api/detect` | `{measurements}` → DetectResponse (+ `threshold_z`, baseline per-section alarm level) |
| POST | `/api/localize` | `{measurements, method?: xgb_reg|section_rank|gnn}` → LocalizeResponse |
| POST | `/api/analyze` | AnalyzeRequest → AnalyzeResponse (below) — the UI "Run" button |
| GET | `/api/metrics`, `/api/eval-arrays`, `/api/gnn-results`, `/api/sensor-study`, `/api/dataset-card` | files from `data/models/` (404 `not_ready` if absent) |
| GET | `/api/scenarios?label=&tap=&split=&limit=&offset=` | `{total, items: ScenarioSummary[]}` |
| GET | `/api/scenarios/{id}` | ScenarioSummary + metered `measurements` |
| WS | `/ws/replay` | held-out test scenarios, 1/tick; client cmds `play`/`pause`/`speed` |
| GET | `/api/mitigation/strategies` | `{strategies: [{id,label,enabled:false}], mitigation_enabled}` |
| POST | `/api/mitigation/plan` | 501 `{status:"not_implemented", planned_actions:[], message}` while flag is false |

```
AnalyzeRequest  = SimulateRequest + {measurements?: {m_*}, scenario_id?: str, localize_method?: str}
                  (precedence: scenario_id → measurements → simulate)
AnalyzeResponse
  request_id, latency_ms
  simulate:        SimulateResponse | null   (null when raw measurements were posted)
  detect:          DetectResponse
  localize:        LocalizeResponse          (always computed; UI marks it "reference" when status=normal)
  recommendations: [str]                     (rule-based inspection guidance; "suspected", never "confirmed")
  disclaimer:      str

Replay message: {t, scenario_id, measurements, detect, localize, truth: {label, km, magnitude_mw}}
ScenarioSummary: {scenario_id, label, split, true_km, true_tap_id, magnitude_mw, load_scale}
Errors: {error: {code, message, details?}} — 422 validation_error | bad_fault | pf_not_converged |
        no_measurements; 404 scenario_not_found | not_ready; 503 models_missing | gnn_unavailable
Every response carries headers X-Request-ID and X-Latency-ms.
```

## 10. WF5 — Dashboard (implemented, `frontend/`)

`make web` (Vite :5173, proxies `/api` + `/ws` to :8000). Pages: Overview, Scenario Lab, Replay,
Metrics, Sensor Study, Mitigation (gated by `/api/features`). Types come from `openapi.json`;
loosely-typed files (`metrics.json`, `sensor_study.json`) are described in `src/api/types.ts`.


## 11. 09 — RAG mitigation advisor (implemented, `rag/`)

Design: `09_RAG_MODULE.md`. Recommendation-only, and `requires_human_approval` is always true.
Flag: `feature_flags.yaml → rag_enabled: true`. Config: `config/rag.yaml`. Index:
`data/models/rag_index.joblib` (`make rag-index`; rebuilt automatically when the corpus or
chunker changes).

| Method | Path | Body → Response |
|---|---|---|
| GET | `/api/rag/status` | → `RagStatus {enabled, llm_available, model, n_docs, n_chunks, sources[{doc_id,title,source_type,n_chunks}], built_at}` |
| POST | `/api/rag/advise` | `AdviseRequest {analysis: {detect: DetectResponse, localize?: LocalizeResponse}, question?, k=6}` → `AdviceResponse` |
| POST | `/api/rag/query` | `RagQueryRequest {question (3–2000 chars), k=6}` → `AdviceResponse` |

```
AdviceResponse
  mode:        "llm" | "extractive"     (extractive when no Claude credentials / on refusal or API error)
  answer:      str                      (cites passages as [n]; invalid [n] removed server-side)
  citations:   [{n, chunk_id, doc_id, title, section, source_type: playbook|external|project_doc,
                 score, cited: bool, snippet}]
  query:       str
  model:       str | null               ("claude-opus-5-5" in llm mode)
  requires_human_approval: true
  warnings:    [str]
  disclaimer:  str
Errors: 503 rag_disabled | rag_unavailable; 422 validation_error
```
`/api/features` now also returns `rag_enabled`. Situational advice (`/advise`) cites guidance
sources only (playbooks and `corpus/external/`); free-text questions (`/query`) may also cite
project specs. Control-action requests get a fixed refusal note.


## 12. 06 full-stack additions + local vector DB (implemented)

| Method | Path | Body → Response |
|---|---|---|
| POST | `/api/batch-analyze` | `BatchAnalyzeRequest {requests: AnalyzeRequest[1..50], explain=false}` → `BatchAnalyzeResponse {request_id, latency_ms, n_ok, results: [{index, result: AnalyzeResponse \| null, error: {code,message} \| null}]}` |
| GET | `/api/sample-requests` | → `[SampleRequest {kind, label, request: SimulateRequest}]` (kinds: theft, small_theft, normal, sensor_fault, line_outage) |
| GET | `/api/sample-request?kind=` | → `SampleRequest` (404 `unknown_kind`) |
| POST | `/api/similar-cases` | `{measurements, k=5 (1–25), label?}` → `SimilarCasesResponse {items: [{scenario_id, similarity, label, true_km, magnitude_mw, load_scale}], n_indexed, dim, embedding, latency_ms}` |

- With `explain=false`, batch items skip SHAP (`top_contributors: []`) and recommendations (`[]`).
  A failing item (e.g. `pf_not_converged`) is reported in `error`, and the rest of the batch completes.
- `request_id` / `latency_ms`: in the body of analyze and batch responses; every response also carries the
  `X-Request-ID` / `X-Latency-ms` headers.
- Replay client commands: `{"cmd":"play"|"pause"}`, `{"cmd":"speed","value":0.25–10}`.

**Vector DB (`vectordb/`):** `VectorStore` keeps L2-normalised float32 vectors plus metadata and does exact
cosine top-k with metadata filters. It is persisted with joblib.
- **Similar-case index** (`data/models/case_index.joblib`, `make vector-index`): 11,025 train+val
  scenarios × 41 standardized physics features (train mean/std). Held-out test scenarios are excluded.
  On 300 test cases, the 5-NN majority label agrees with the true class 79% of the time. It is reference
  context, not a classifier.
- **RAG dense passages:** LSA (TruncatedSVD of the word TF-IDF, ≤128-d) vectors stored in the
  `rag_index`; retrieval score = 0.45·word + 0.25·char + 0.30·dense cosine.

Env (`.env.example`): `CORS_ORIGINS`, `VITE_API_URL`, `VITE_USE_MOCKS` (MSW mock API), optional `ANTHROPIC_API_KEY`.


## 13. Activity console (event log)

In-memory event bus (`backend/app/events.py`, ring buffer of 2,000 events, not persisted).

```
Event {seq, ts (unix s), level: debug|info|warn|alert|error,
       category: system|api|simulation|detection|localization|rag|replay|batch|mitigation|user,
       message, data: {}, source: server|client, request_id | null}
```
| Method | Path | Notes |
|---|---|---|
| GET | `/api/events?since=&limit=&min_level=&category=a,b` | → `{events: Event[], last_seq}` |
| POST | `/api/events` | browser user actions: `{message (1–300), level?, category: "user", data?}` → 201 Event. Clients can only post `user` events. |
| GET | `/api/events/meta` | levels, categories, last_seq |
| WS | `/ws/events?backlog=200` | backlog, then live events; `{"type":"ping"}` keep-alive every 20 s |

Server emits: startup/shutdown (system); every `/api/*` request (api; health/events polling at debug level,
4xx = warn, 5xx = error); the injected scenario (simulation); the detection outcome (detection —
**alert** = suspected unauthorized extraction near km X with its interval; warn = sensor fault / network
event); the localization section and top tap; batch summaries; vector-DB similar-case searches; RAG
questions; replay connect, pause, speed and alert ticks; blocked mitigation requests; API errors.
All events from one HTTP request share its `request_id`.
UI: a dock on every page (toggle with `` ` `` or the sidebar **Console** button, which shows an unread-alert
badge; Esc closes) plus a `/console` page. It has level, category and search filters, pause, follow,
clear and JSON export, and alert rows link to the km in Scenario Lab.


## 14. Final results snapshot (held-out data)

| | Value |
|---|---|
| Detection (test) | XGBoost recall 91.5%, FPR 2.8%, recall @1% FPR 89.8% (baseline 56.0%); theft-vs-normal recall @1% FPR 92.5% |
| Localization (unseen locations) | GNN MAE 21.3 km (3-seed mean 21.7 ± 1.2), median 14.3 km; XGBoost 23.1 km; physics baseline 49.9 km |
| Sensor study, noise ×1 (MAE / top-3) | major-only 21.2 km / 60% · +mid 18.7 km / 67% · dense 15.5 km / 92% |
| Sensor study, noise ×4, major-only | recall 81.8%, FPR 8.6%, MAE 41.3 km |
| Separability (major-only) | 145 of 378 tap pairs are not separable; within-section locations blur together, sections are distinct |
