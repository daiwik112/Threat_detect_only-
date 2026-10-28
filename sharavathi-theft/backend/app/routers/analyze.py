from fastapi import APIRouter

from app.schemas import (AnalyzeRequest, AnalyzeResponse, BatchAnalyzeRequest, BatchAnalyzeResponse,
                         DetectResponse, ErrorResponse, LocalizeRequest, LocalizeResponse,
                         MeasurementsRequest, SampleRequest, SimilarCasesRequest, SimilarCasesResponse)
from app.services import analysis_service as svc

router = APIRouter(prefix="/api", tags=["analyze"])
_err = {422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


@router.post("/detect", response_model=DetectResponse, responses=_err)
def post_detect(req: MeasurementsRequest):
    return svc.detect(req.measurements)


@router.post("/localize", response_model=LocalizeResponse, responses=_err)
def post_localize(req: LocalizeRequest):
    return svc.localize(req.measurements, req.method)


@router.post("/analyze", response_model=AnalyzeResponse, responses=_err)
def post_analyze(req: AnalyzeRequest):
    """Simulate (or take `measurements` / dataset `scenario_id`) → detect → localize."""
    return svc.analyze(req)


@router.post("/batch-analyze", response_model=BatchAnalyzeResponse, responses=_err)
def post_batch_analyze(req: BatchAnalyzeRequest):
    """Up to 50 analyses in one call — used by the magnitude sweep and profile comparison."""
    return svc.batch_analyze(req.requests, explain=req.explain)


@router.get("/sample-requests", response_model=list[SampleRequest])
def get_sample_requests():
    """Preset scenarios for the demo buttons."""
    return svc.sample_requests()


@router.get("/sample-request", response_model=SampleRequest, responses={404: {"model": ErrorResponse}})
def get_sample_request(kind: str = "theft"):
    return svc.sample_request(kind)


@router.post("/similar-cases", response_model=SimilarCasesResponse, responses=_err)
def post_similar_cases(req: SimilarCasesRequest):
    """Nearest labelled dataset scenarios in the vector DB (physics-feature embedding)."""
    return svc.similar_cases(req.measurements, req.k, req.label)
