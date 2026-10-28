"""
future/mitigation.py — Mitigation planner interfaces (07_FUTURE_MITIGATION). STUB ONLY.

Nothing here performs real grid control. get_planner() returns a planner whose plan()
always reports status "not_implemented"; a real planner (e.g. OPF-based) can be
returned later without changing the API router.

Rules (07): detection/localization code never imports future/; any future planner stays
recommendation-only — plans carry requires_human_approval=True and nothing here may talk
to SCADA or a real grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MitigationAction:
    kind: str            # "redispatch" | "shed" | "reroute" | "inspect"
    target: str          # bus/line id
    delta_mw: float
    rationale: str


@dataclass
class MitigationPlan:
    status: str          # "not_implemented" | "ok"
    actions: list[MitigationAction] = field(default_factory=list)
    predicted_effect: dict = field(default_factory=dict)   # e.g. {"max_loading_pct": ..., "min_vm_pu": ...}
    requires_human_approval: bool = True                    # recommendation-only, always


class MitigationPlanner(ABC):
    @abstractmethod
    def plan(self, net, anomaly: dict) -> MitigationPlan: ...


class NotImplementedPlanner(MitigationPlanner):
    def plan(self, net, anomaly):
        return MitigationPlan("not_implemented", [], {})


STRATEGIES = [
    {"id": "load_balance", "label": "Load balancing", "enabled": False},
    {"id": "redispatch", "label": "Generation redispatch", "enabled": False},
    {"id": "shed", "label": "Controlled load shedding", "enabled": False},
]


def get_planner() -> MitigationPlanner:
    # later: read feature_flags.yaml and return a real planner (e.g. OPFPlanner())
    return NotImplementedPlanner()
