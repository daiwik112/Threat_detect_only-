import json

from fastapi import APIRouter

from app.deps import APIError, settings
from app.schemas import ErrorResponse

router = APIRouter(prefix="/api", tags=["metrics"])
_404 = {404: {"model": ErrorResponse}}


def _json(name: str, hint: str):
    p = settings.model_dir / name
    if not p.exists():
        raise APIError(404, "not_ready", f"{name} not found — {hint}")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/metrics", responses=_404)
def get_metrics():
    """data/models/metrics.json — MetricsResponse keys at top level (+ detail keys)."""
    return _json("metrics.json", "run `make train`")


@router.get("/eval-arrays", responses=_404)
def get_eval_arrays():
    """Raw arrays for charts (estimated vs true km, ROC)."""
    return _json("eval_arrays.json", "run `make train`")


@router.get("/sensor-study", responses=_404)
def get_sensor_study():
    return _json("sensor_study.json", "run `make study-data && make sensor-study`")


@router.get("/gnn-results", responses=_404)
def get_gnn_results():
    return _json("gnn_results.json", "run `make train-gnn`")


@router.get("/dataset-card", responses=_404)
def get_dataset_card():
    if not settings.dataset_card.exists():
        raise APIError(404, "not_ready", "dataset_card.md not found — run `make data`")
    return {"markdown": settings.dataset_card.read_text(encoding="utf-8")}
