from fastapi import APIRouter

from app.deps import state
from app.schemas import NetworkResponse
from app.services.simulation_service import build_network

router = APIRouter(prefix="/api", tags=["network"])


@router.get("/network", response_model=NetworkResponse)
def get_network():
    if state.network is None:
        state.network = build_network()
    return state.network
