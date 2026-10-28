from typing import Optional

from fastapi import APIRouter, Query

from app.schemas import ErrorResponse, ScenarioDetail, ScenarioListResponse
from app.services import analysis_service as svc

router = APIRouter(prefix="/api", tags=["scenarios"])


@router.get("/scenarios", response_model=ScenarioListResponse)
def get_scenarios(label: Optional[str] = None, tap: Optional[str] = None,
                  split: Optional[str] = None, limit: int = Query(50, ge=1, le=500),
                  offset: int = Query(0, ge=0)):
    return svc.list_scenarios(label, tap, split, limit, offset)


@router.get("/scenarios/{scenario_id}", response_model=ScenarioDetail,
            responses={404: {"model": ErrorResponse}})
def get_scenario(scenario_id: str):
    return svc.scenario_detail(scenario_id)
