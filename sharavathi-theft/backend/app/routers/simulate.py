from fastapi import APIRouter

from app.schemas import ErrorResponse, SimulateRequest, SimulateResponse
from app.services.simulation_service import simulate

router = APIRouter(prefix="/api", tags=["simulate"])


@router.post("/simulate", response_model=SimulateResponse,
             responses={422: {"model": ErrorResponse}})
def post_simulate(req: SimulateRequest):
    """Build net (+ theft as a real pandapower load / fault) → AC PF → noisy metering."""
    return simulate(req)
