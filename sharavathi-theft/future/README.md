# future/ — Mitigation & load-balancing extension points (stubs only)

Nothing in this package performs grid control. It exists so mitigation can be added later
**without refactoring detection or localization**. Spec: `07_FUTURE_MITIGATION.md`.

| File | Interface | Stub returned today |
|---|---|---|
| `mitigation.py` | `MitigationPlanner.plan(net, anomaly) -> MitigationPlan` | `NotImplementedPlanner` → `status="not_implemented"` |
| `forecasting.py` | `DemandForecaster.forecast(history, horizon_steps) -> DemandForecast` | `NotImplementedForecaster` |
| `overload.py` | `OverloadPredictor.predict(net, horizon_steps) -> OverloadForecast` | `NotImplementedOverloadPredictor` |

Each module has a `get_*()` factory. To enable a real implementation later, return it from
the factory (e.g. based on `config/feature_flags.yaml`). No router or UI change is needed.

## Wiring (already in place)
- **Flag:** `config/feature_flags.yaml → mitigation_enabled: false`
- **API** (`backend/app/routers/mitigation.py`):
  - `GET /api/mitigation/strategies` returns `load_balance`, `redispatch`, `shed`, all `enabled: false`
  - `POST /api/mitigation/plan` returns **501** `{status: "not_implemented", planned_actions: [], message}` while the flag is off.
    When the flag is on, it calls `get_planner().plan(net, anomaly)` and returns the plan.
- **UI** (`frontend/src/pages/Mitigation.tsx`): the tab is always visible. With the flag off it shows
  a 🔒 "Coming soon" badge, a disabled strategy picker and an empty "Plan & impact" panel.

## Planned modules (not built)
| Module | Idea | Tooling |
|---|---|---|
| Demand forecasting | short-term load forecast per bus | LSTM / XGBoost on synthetic profiles |
| Overload prediction | predict line loading > limit N steps ahead | classifier on PF time series |
| Redispatch | re-solve with generation change | `pp.runopp` / `pp.rundcopp` (needs cost + limits) |
| Load management | feasible shedding / shifting suggestions | constrained optimization (SciPy / OR-Tools) |
| Impact simulation | apply plan to net, re-run PF, compare | reuse `grid.flow` |

## Rules
1. Detection/localization code (`grid/`, `simdata/`, `ml/`) must **not** import from `future/`
   (enforced by `tests/test_future.py`).
2. Any future automated control stays **recommendation-only**. Plans carry
   `requires_human_approval=True`, and there are no SCADA or real-grid connections.
3. Tests only assert the stub contract: flag off → 501 + `not_implemented`.
