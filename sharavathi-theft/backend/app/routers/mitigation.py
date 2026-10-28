"""Mitigation STUB routes (07). Real planning is not implemented."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.deps import load_flags
from app.events import bus
from app.schemas import MitigationRequest, MitigationResponse, StrategiesResponse

router = APIRouter(prefix="/api/mitigation", tags=["mitigation (stub)"])
_MSG = "Load balancing & mitigation planned for a future release"


@router.get("/strategies", response_model=StrategiesResponse)
def strategies():
    from future.mitigation import STRATEGIES
    return {"strategies": STRATEGIES, "mitigation_enabled": load_flags()["mitigation_enabled"]}


@router.post("/plan", response_model=MitigationResponse,
             responses={501: {"model": MitigationResponse}})
def plan(req: MitigationRequest):
    from future.mitigation import get_planner
    if not load_flags()["mitigation_enabled"]:
        bus.publish("mitigation", "warn", f"Mitigation plan requested ({req.strategy}) — blocked: not implemented (flag off)")
        return JSONResponse(status_code=501, content={
            "status": "not_implemented", "planned_actions": [], "message": _MSG})
    import copy
    from app.services.simulation_service import _template
    net = copy.deepcopy(_template())          # planners get their own copy of the base network
    p = get_planner().plan(net, {"scenario_id": req.scenario_id, "strategy": req.strategy})
    return {"status": p.status, "planned_actions": [a.__dict__ for a in p.actions],
            "message": _MSG if p.status == "not_implemented" else "ok"}
