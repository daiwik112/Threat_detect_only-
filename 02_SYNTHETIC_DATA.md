# 02 — WORKFLOW: Synthetic Data Generation

**Goal:** produce a labeled, leak-free scenario dataset (normal + theft + sensor faults + outage) using the pandapower model.
**Depends on:** `01` (`grid.build`, `grid.flow`, `grid.sensors`) and `08_DATA_CONTRACTS.md`.
**Writes:** `data/processed/dataset.parquet`, `data/processed/splits.json`, `data/processed/dataset_card.md`

## Tasks
1. `simdata/operating_points.py`
   - Generate base operating points: daily load curve (sinusoid + noise) × seasonal factor, `load_scale` ∈ [0.5, 1.15]; per-bus load jitter ±5 %.
   - Each base point gets a `group_id`. Many scenarios (normal + variants) share a group.
2. `simdata/scenarios.py` — one function per class, each returns a `ScenarioSpec`:
   | class | how it is created |
   |---|---|
   | normal | base point only |
   | theft | `build_net(theft={km∈taps, mw∈[1,40], pf∈[0.85,0.98]})` — **real extra load**, re-run PF |
   | sensor_bias | normal PF, then +/- 2–10 % bias on 1–2 random channels |
   | sensor_drift | linear drift on a channel over the window |
   | missing | drop 1–3 channels (NaN) |
   | line_outage | take one circuit out of service (double circuit → flows shift) |
   - Also add a **small theft** regime (1–3 MW) near noise level to test sensitivity limits.
   - Mixed case (theft + sensor bias) at 5 % of data for robustness.
3. `simdata/generate.py` — parallel (`joblib`) loop: build → PF → measure → record row per `08` schema. Skip + log non-converged cases. Deterministic via `seed`.
4. `simdata/splits.py`
   - Group split by `group_id` (train 70 / val 15 / test 15).
   - **Leave-location-out** test: all theft rows with `true_km` within ±5 km of `holdout_locations_km` go to `test_loc_holdout` only.
   - Save indices to `splits.json`.
5. `simdata/profiles.py` — generate the same scenarios under **each sensor profile** (`major_only`, `major_plus_mid`, `dense`) for the sensor-placement study (`03`, step 6). Use same seeds so profiles are comparable.
6. `simdata/dataset_card.py` — write `dataset_card.md`: counts per class, magnitude/location histograms, noise settings, assumptions, known limitations.
7. CLI: `python -m simdata.generate --config config/scenarios.yaml --profile major_only --out data/processed/`

## Quality checks (`tests/test_simdata.py`)
- Class balance as configured (±2 %).
- No duplicate scenario_ids; no group appears in >1 split.
- Theft rows: `true_km` and `magnitude_mw` present; non-theft rows: NaN.
- Sanity: mean upstream `p_from` for theft rows > matching normal rows at same group.
- Reproducible: same seed → identical parquet hash.
- Plot (save PNG): measured-vs-expected residual distribution per class; they must visibly separate for large thefts and overlap for tiny ones (that's expected and honest).

## Pitfalls to avoid
- DO NOT subtract theft from a meter column.
- DO NOT let rows from the same group sit in train and test.
- DO NOT tune noise until results look good; keep it in `sensors.yaml` and report it.
