# 01 — WORKFLOW: pandapower Grid Model (the "nodes")

**Goal:** build the Sharavathi-origin 220 kV network as a runnable pandapower model with virtual tap buses.
**Reads:** `config/topology.yaml`, `config/sensors.yaml` **Writes:** `grid/` package, `data/raw/baseline_states.parquet`
**Depends on:** `08_DATA_CONTRACTS.md` only.

## Tasks
1. `grid/config.py` — load/validate YAML with pydantic. Refuse to run if a param lacks `source`.
2. `grid/build.py` — `build_net(topology, load_scale=1.0, theft=None) -> pp.pandapowerNet`
   - Create HV buses (220 kV) for each substation.
   - Create `pp.create_ext_grid` at SHARAVATHI (vm_pu from config) as slack.
   - Subdivide each line into segments of `tap_spacing_km`; create one **tap bus** per boundary (id `TAP_<km>`), plus line segments with `pp.create_line_from_parameters` (r, x, c, max_i_ka, `parallel=circuits`).
   - Add accounted loads from config at their buses (scaled by `load_scale`).
   - If `theft={km, mw, pf}` → `pp.create_load(net, tap_bus, p_mw=mw, q_mvar=mw*tan(acos(pf)), name="THEFT")`. Snap km to nearest tap.
   - Return net with `net.tap_map` (dict tap_id→km) and `net.section_map` (line segment→km range).
3. `grid/flow.py` — `run_pf(net) -> dict`: `pp.runpp(net, algorithm="nr", init="dc")`; return vm, va, line p/q from/to, loading %, losses; raise `PFNonConvergence` cleanly.
4. `grid/sensors.py` — `measure(results, profile, noise, rng, faults=None)`:
   - Keep only metered buses/lines of the profile.
   - Add Gaussian noise (sigma from `sensors.yaml`).
   - Faults: `bias(+x%)`, `drift(rate/step)`, `missing(prob)`, `stuck`.
5. `grid/estimator.py` — WLS state estimation with `pandapower.estimation` (`estimate`) using the metered measurements; return chi-square J-value + normalized residuals. Handle unobservable case (return `None`, flag it).
6. `grid/baseline.py` — sweep `load_scale ∈ [0.5, 1.15]` (e.g., 200 points) and save the *no-theft* expected states to `data/raw/baseline_states.parquet`. This is the "expected physical behavior" reference.
7. `grid/viz_check.py` — matplotlib/plotly quick plot: voltage profile vs km, line loading, losses vs load.
8. CLI: `python -m grid.build --check` prints convergence, total loss MW (%), max loading, min voltage.

## Physics sanity targets
- No-theft losses roughly 1–4 % of delivered power at peak; voltage 0.95–1.05 pu everywhere.
- Adding a 20 MW theft at km 150 must raise losses and drop downstream voltage / change flows on upstream segments.

## Acceptance tests (`tests/test_grid.py`)
- PF converges for 100 random load scales.
- Power balance: Σgen = Σload + Σloss (tol 1e-6 MW).
- Theft load at tap X changes `res_line.p_from_mw` of the upstream metered section by ≈ magnitude (± loss).
- Snap logic: km=47 → nearest tap 50.
- Missing-sensor profile still returns valid dict with NaNs.

## Notes
- All parameter values are **assumed** until verified with KPTCL/SLDC — surface this in the API's `assumptions` list.
- Keep functions pure (no globals) so `02` can call them in parallel (joblib).
