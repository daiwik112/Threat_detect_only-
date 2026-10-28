"""
tests/test_future.py — WF6 (07_FUTURE_MITIGATION) stub contract.

Only the stub contract is asserted (07 rule 3): package importable, planners return
"not_implemented", no detection/localization module imports future/, and the API answers
501 + not_implemented while mitigation_enabled is false.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_package_importable_and_stubs_not_implemented():
    from future.forecasting import get_forecaster
    from future.mitigation import STRATEGIES, MitigationPlan, get_planner
    from future.overload import get_overload_predictor

    plan = get_planner().plan(None, {"scenario_id": "x", "strategy": "shed"})
    assert isinstance(plan, MitigationPlan)
    assert plan.status == "not_implemented" and plan.actions == [] and plan.predicted_effect == {}
    assert plan.requires_human_approval is True            # 07 rule 2: recommendation-only
    assert get_forecaster().forecast({"PEENYA": [180.0]}, 4).status == "not_implemented"
    assert get_overload_predictor().predict(None, 4).status == "not_implemented"
    assert {s["id"] for s in STRATEGIES} == {"load_balance", "redispatch", "shed"}
    assert not any(s["enabled"] for s in STRATEGIES)


@pytest.mark.parametrize("pkg", ["grid", "simdata", "ml"])
def test_detection_code_does_not_import_future(pkg):
    """07 rule 1: detection/localization must not depend on future/."""
    offenders = []
    for py in (ROOT / pkg).rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(n == "future" or n.startswith("future.") for n in names):
                offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, offenders


def test_flag_off_by_default():
    from grid.config import load_feature_flags
    assert load_feature_flags().mitigation_enabled is False


def test_api_stub_contract(monkeypatch):
    """Flag off → 501 + not_implemented; flag on → planner is called with a real net."""
    import sys
    sys.path.insert(0, str(ROOT / "backend"))
    from fastapi.testclient import TestClient
    from app.main import app
    from app.routers import mitigation as router

    with TestClient(app) as c:
        r = c.post("/api/mitigation/plan", json={"scenario_id": "x", "strategy": "load_balance"})
        assert r.status_code == 501
        assert r.json() == {"status": "not_implemented", "planned_actions": [],
                            "message": "Load balancing & mitigation planned for a future release"}
        s = c.get("/api/mitigation/strategies").json()
        assert s["mitigation_enabled"] is False and len(s["strategies"]) == 3

        seen = {}
        from future import mitigation as fm

        class Spy(fm.NotImplementedPlanner):
            def plan(self, net, anomaly):
                seen["n_lines"], seen["anomaly"] = len(net.line), anomaly
                return super().plan(net, anomaly)

        monkeypatch.setattr(router, "load_flags", lambda: {"mitigation_enabled": True, "gnn_enabled": True})
        monkeypatch.setattr(fm, "get_planner", lambda: Spy())
        r = c.post("/api/mitigation/plan", json={"scenario_id": "x", "strategy": "shed"})
        assert r.status_code == 200 and r.json()["status"] == "not_implemented"
        assert seen == {"n_lines": 38, "anomaly": {"scenario_id": "x", "strategy": "shed"}}
        assert c.post("/api/mitigation/plan", json={"scenario_id": "x", "strategy": "nope"}).status_code == 422
