# 07 — WORKFLOW: Future Work Hooks (Load Balancing & Mitigation) — STUBS ONLY

**Goal:** leave clean extension points so mitigation can be added later WITHOUT refactoring detection/localization. Nothing here performs real control. Not part of current deliverable logic.
**Depends on:** `01` (grid), `04` (routers), `05` (Mitigation page), `08` (stub schemas).

## Package `future/`
```
future/
├─ __init__.py
├─ mitigation.py     abstract interfaces + NotImplemented planner
├─ forecasting.py    stub: demand forecast interface
├─ overload.py       stub: overload predictor interface
└─ README.md
```

### `future/mitigation.py`
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class MitigationAction:
    kind: str            # "redispatch" | "shed" | "reroute" | "inspect"
    target: str          # bus/line id
    delta_mw: float
    rationale: str

@dataclass
class MitigationPlan:
    status: str          # "not_implemented" | "ok"
    actions: list[MitigationAction]
    predicted_effect: dict   # e.g. {"max_loading_pct": ..., "min_vm_pu": ...}

class MitigationPlanner(ABC):
    @abstractmethod
    def plan(self, net, anomaly: dict) -> MitigationPlan: ...

class NotImplementedPlanner(MitigationPlanner):
    def plan(self, net, anomaly):
        return MitigationPlan("not_implemented", [], {})

def get_planner() -> MitigationPlanner:
    # later: read feature_flags.yaml, return OPFPlanner()
    return NotImplementedPlanner()
```

## Planned future modules (describe in README, don't build now)
| Module | Idea | Tooling |
|---|---|---|
| Demand forecasting | short-term load forecast per bus | LSTM / XGBoost on synthetic profiles |
| Overload prediction | predict line loading > limit N steps ahead | classifier on PF time series |
| Redispatch | re-solve with generation change | `pp.runopp` / `pp.rundcopp` (needs cost + limits) |
| Load management | feasible shedding/shifting suggestions | constrained optimization (SciPy / OR-Tools) |
| Impact simulation | apply plan to net, re-run PF, compare | reuse `grid.flow` |

## Wiring (already in 04/05)
- Backend: `POST /api/mitigation/plan` calls `get_planner().plan(net, anomaly)`; returns 501 while flag off.
- Frontend: Mitigation tab renders `MitigationResponse`; shows "Coming soon" when `mitigation_enabled=false`.
- Flag: `config/feature_flags.yaml → mitigation_enabled`.

## Rules
1. Detection/localization code must NOT import from `future/`.
2. Any future automated control must stay recommendation-only (human approval); no real SCADA/grid connections.
3. Add tests only asserting the stub contract: flag off → 501 + `not_implemented`.

## Acceptance (now)
- Stub package importable; `/api/mitigation/*` returns documented placeholder; UI tab visible and disabled; zero effect on other workflows.
