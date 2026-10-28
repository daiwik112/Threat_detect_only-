# 03 — WORKFLOW: Features, Detector, XGBoost, GNN, Sensor Study, Evaluation

**Goal:** detect anomalies, classify cause, localize theft (km + ranked taps), prove it beats a baseline.
**Depends on:** `02` dataset + splits, `01` for expected-state lookups, `08` contracts.
**Writes:** `ml/` package, `data/models/*.joblib|*.pt`, `data/models/metrics.json`, `data/models/sensor_study.json`

## Step 1 — Physics features (`ml/features.py`)
Per scenario, per metered section (between consecutive metered buses):
- `r_p = P_in − P_out − P_loss_expected − P_accounted_loads` (expected from the no-theft baseline at the same `load_scale` estimate)
- `r_q` same for reactive.
- Voltage deviation vs expected (`vm − vm_exp`), angle deviation, current deviation.
- Mismatch of the two ends of each line (`p_from + p_to` vs expected loss).
- WLS state-estimation outputs: J-statistic, max normalized residual, which channel.
- Sensor-quality flags: n_missing, missing mask.
- Load-scale estimate from head-end flow (the model does NOT see true `load_scale`).
Add everything as `f_*` columns. No label-derived leakage.

## Step 2 — Baseline detector (`ml/detector_baseline.py`)
- Threshold on `|r_p|/σ` from normal-only train data (e.g. 99.5th percentile) → `suspicious`.
- Separate chi-square test on WLS J.
- This is the benchmark every ML model must beat.

## Step 3 — XGBoost anomaly classifier (`ml/xgb_classifier.py`)
- Target: label (6 classes) and binary `is_anomaly`.
- `scale_pos_weight` / class weights; early stopping on val; Optuna light tuning (≤50 trials).
- Output prob per class → map to `likely_cause {extraction, sensor_error, other}`.
- SHAP top contributors per prediction (`shap.TreeExplainer`) for the dashboard.

## Step 4 — Localization (`ml/localize.py`)
Train on theft rows only. Two approaches, compare both:
- **A. Regression:** `XGBRegressor` → `true_km`. Interval via quantile regression (5th/95th) or conformal prediction on val.
- **B. Section/tap ranking:** `XGBClassifier` over tap ids (or 10-km bins) → probability vector → `ranked_taps`, `ranked_sections`; est_km = prob-weighted mean; interval = smallest set covering 80 % prob.
- Report both; ship whichever is better on `test_loc_holdout`.

## Step 5 — GNN (advanced, `ml/gnn/`) — only after Steps 1–4 pass
- Graph: nodes = all buses (incl. tap buses), edges = line segments (bidirectional).
- Node features: `[vm_dev, va_dev, p_inj_resid, q_inj_resid, metered_flag, km_norm, is_junction]`; unmetered taps get zeros + `metered_flag=0` (the GNN must infer via propagation).
- Model: 3-layer GCN/GraphSAGE (PyTorch Geometric) → node logits (theft-at-node softmax) + graph-level anomaly head.
- Train on `train`, early stop on `val`, eval on `test` and `test_loc_holdout`. Seed ×3, report mean±std.
- Keep ONLY if it improves MAE / top-k accuracy vs XGBoost; otherwise document the negative result.

## Step 6 — Sensor placement & observability study (`ml/sensor_study.py`)
Run Steps 3–4 for each profile: `major_only`, `major_plus_mid`, `dense`, and noise ×{0.5,1,2,4}.
- Output table: detection recall, FPR, MAE km, top-3 tap accuracy per config → `sensor_study.json`.
- Also compute **indistinguishability**: for pairs of tap locations, distance between mean feature vectors; report location ranges that are not separable under `major_only`.

## Step 7 — Evaluation (`ml/evaluate.py`) → `metrics.json`
- Detection: precision, recall, F1, FPR (target FPR reported at fixed recall).
- Classification: confusion matrix, per-class recall.
- Localization: MAE km, median error, % within 10/25/50 km, by magnitude bucket (1–5, 5–15, 15–40 MW).
- Robustness: performance with 1–3 missing channels, noise ×2.
- Compare vs baseline in the same table. Plot: error vs magnitude, error vs true_km.
- Save plots as PNG + the arrays for the frontend charts.

## Step 8 — Packaging
- `ml/infer.py`: `load_models()`, `detect(measurements)->DetectResponse`, `localize(measurements)->LocalizeResponse` matching `08`.
- Version stamp + dataset hash in every artifact.

## Acceptance
- `make train` reproduces `metrics.json`; pytest checks shapes/schemas.
- Models beat residual baseline on held-out set for at least one of: recall @ fixed FPR, localization MAE. If not, report honestly.
- Honest claim template: "Under the simulated measurement setup, location is resolvable to ±X km for thefts ≥ Y MW."
