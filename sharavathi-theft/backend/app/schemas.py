"""
backend/app/schemas.py — Pydantic models. Names match 08_DATA_CONTRACTS.md §4.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

SensorProfile = Literal["major_only", "major_plus_mid", "dense"]
Measurements = dict[str, Optional[float]]


# ---------------------------------------------------------------- network
class Bus(BaseModel):
    id: str
    km: float
    role: str
    metered: bool
    lat: Optional[float] = None
    lon: Optional[float] = None


class Line(BaseModel):
    id: str
    from_bus: str
    to_bus: str
    km_len: float
    loading_pct: Optional[float] = None        # base case (load_scale 1.0)


class Tap(BaseModel):
    id: str
    km: float
    line_id: str


class Shunt(BaseModel):
    bus: str
    q_mvar_cap: float


class Section(BaseModel):
    id: str
    from_bus: str
    to_bus: str
    km_from: float
    km_to: float


class NetworkResponse(BaseModel):
    buses: list[Bus]
    lines: list[Line]
    taps: list[Tap]
    shunts: list[Shunt]
    sections: list[Section]                    # metered sections (major_only)
    assumptions: list[str]
    disclaimer: str


# ---------------------------------------------------------------- simulate
class TheftSpec(BaseModel):
    km: float = Field(..., description="km from SHARAVATHI; clamped to [0, 361], snapped to nearest tap")
    magnitude_mw: float = Field(..., description="unmetered load MW; clamped to [0, 60]")
    pf: float = Field(0.95, ge=0.5, le=1.0)


class FaultSpec(BaseModel):
    type: Literal["bias", "drift", "missing", "line_outage"]
    channel: Optional[str] = Field(None, description="m_<BUS>_<vm|p|q> for sensor faults, LINE<i> for outage")
    param: Optional[float] = Field(None, description="bias %, drift %/step, or outage line index")


class SimulateRequest(BaseModel):
    load_scale: float = Field(1.0, description="system demand factor; clamped to [0.5, 1.15]")
    theft: Optional[TheftSpec] = None
    fault: Optional[FaultSpec] = None
    sensor_profile: SensorProfile = "major_only"
    noise_mult: float = Field(1.0, ge=0.0, le=10.0, description="multiplier on sensors.yaml noise")
    seed: int = 42


class SimulateResponse(BaseModel):
    scenario_id: str
    measurements: Measurements                 # metered channels only (profile-dependent)
    expected: Measurements                     # noiseless no-theft twin, same channels
    true_label: str
    true_km: Optional[float] = None
    true_tap_id: Optional[str] = None
    true_magnitude_mw: float = 0.0
    sensor_profile: str
    warnings: list[str] = []
    disclaimer: str


# ---------------------------------------------------------------- detect / localize
class MeasurementsRequest(BaseModel):
    measurements: Measurements


class LocalizeRequest(MeasurementsRequest):
    method: Optional[Literal["xgb_reg", "section_rank", "gnn"]] = None


class Residual(BaseModel):
    section: str
    r_p_mw: Optional[float]
    r_q_mvar: Optional[float]
    z: Optional[float]


class Contributor(BaseModel):
    feature: str
    shap: float


class LikelyCause(BaseModel):
    extraction: float
    sensor_error: float
    other: float


class DetectResponse(BaseModel):
    status: Literal["normal", "suspicious"]
    anomaly_score: float
    residuals: list[Residual]
    top_contributors: list[Contributor]
    likely_cause: LikelyCause
    threshold_z: Optional[float] = None        # baseline per-section alarm threshold (for charts)


class RankedSection(BaseModel):
    section: str
    prob: float


class RankedTap(BaseModel):
    tap_id: str
    km: float
    prob: float


class LocalizeResponse(BaseModel):
    method: Literal["xgb_reg", "section_rank", "gnn"]
    est_km: float
    interval_km: list[float]
    ranked_sections: list[RankedSection]
    ranked_taps: list[RankedTap]


# ---------------------------------------------------------------- analyze
class AnalyzeRequest(SimulateRequest):
    """Simulate-then-analyze, OR analyze given measurements, OR a dataset scenario_id."""
    measurements: Optional[Measurements] = None
    scenario_id: Optional[str] = None
    localize_method: Optional[Literal["xgb_reg", "section_rank", "gnn"]] = None


class AnalyzeResponse(BaseModel):
    request_id: str
    latency_ms: float
    simulate: Optional[SimulateResponse]
    detect: DetectResponse
    localize: LocalizeResponse
    recommendations: list[str]
    disclaimer: str


# ---------------------------------------------------------------- misc
class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str]
    dataset_hash: Optional[str]
    models_loaded: dict[str, bool]
    selected_localizer: Optional[str]


class FeaturesResponse(BaseModel):
    mitigation_enabled: bool
    gnn_enabled: bool
    gnn_available: bool
    rag_enabled: bool = True


# ---------------------------------------------------------------- RAG advisor (09)
class RagSource(BaseModel):
    doc_id: str
    title: str
    source_type: str
    n_chunks: int


class RagStatus(BaseModel):
    enabled: bool
    llm_available: bool
    model: Optional[str]
    n_docs: int
    n_chunks: int
    sources: list[RagSource]
    built_at: Optional[str]


class RagCitation(BaseModel):
    n: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    source_type: str
    score: float
    cited: bool
    snippet: str


class AdviceResponse(BaseModel):
    mode: Literal["llm", "extractive"]
    answer: str
    citations: list[RagCitation]
    query: str
    model: Optional[str]
    requires_human_approval: bool
    warnings: list[str]
    disclaimer: str


class AdviseAnalysis(BaseModel):
    detect: DetectResponse
    localize: Optional[LocalizeResponse] = None


class AdviseRequest(BaseModel):
    analysis: AdviseAnalysis
    question: Optional[str] = Field(None, max_length=2000)
    k: int = Field(6, ge=1, le=12)


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(6, ge=1, le=12)


class ScenarioSummary(BaseModel):
    scenario_id: str
    label: str
    split: Optional[str]
    true_km: Optional[float]
    true_tap_id: Optional[str]
    magnitude_mw: float
    load_scale: float


class ScenarioListResponse(BaseModel):
    total: int
    items: list[ScenarioSummary]


class ScenarioDetail(ScenarioSummary):
    measurements: Measurements


class MitigationRequest(BaseModel):
    scenario_id: str
    strategy: Literal["load_balance", "redispatch", "shed"]


class MitigationResponse(BaseModel):
    status: str
    planned_actions: list[Any]
    message: str


class Strategy(BaseModel):
    id: str
    label: str
    enabled: bool


class StrategiesResponse(BaseModel):
    strategies: list[Strategy]
    mitigation_enabled: bool


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


# ---------------------------------------------------------------- 06 additions
class BatchAnalyzeRequest(BaseModel):
    requests: list[AnalyzeRequest] = Field(..., min_length=1, max_length=50)
    explain: bool = False          # SHAP + recommendations per item (slower)


class BatchItem(BaseModel):
    index: int
    result: Optional[AnalyzeResponse]
    error: Optional[ErrorBody]


class BatchAnalyzeResponse(BaseModel):
    request_id: str
    latency_ms: float
    n_ok: int
    results: list[BatchItem]


class SampleRequest(BaseModel):
    kind: str
    label: str
    request: SimulateRequest


class SimilarCasesRequest(BaseModel):
    measurements: Measurements
    k: int = Field(5, ge=1, le=25)
    label: Optional[str] = None


class SimilarCase(BaseModel):
    scenario_id: str
    similarity: float
    label: str
    true_km: Optional[float]
    magnitude_mw: float
    load_scale: float


class SimilarCasesResponse(BaseModel):
    items: list[SimilarCase]
    n_indexed: int
    dim: int
    embedding: str
    latency_ms: float
