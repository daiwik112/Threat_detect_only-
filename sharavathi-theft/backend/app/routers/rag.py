"""RAG mitigation advisor routes (09_RAG_MODULE). Recommendation-only; no control actions."""

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.deps import APIError, load_flags
from app.events import bus
from app.schemas import AdviceResponse, AdviseRequest, ErrorResponse, RagQueryRequest, RagStatus

router = APIRouter(prefix="/api/rag", tags=["rag advisor"])
_err = {503: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


def _advisor():
    if not load_flags()["rag_enabled"]:
        raise APIError(503, "rag_disabled", "RAG advisor is disabled (feature_flags.yaml: rag_enabled: false).")
    from rag.advisor import get_advisor
    try:
        return get_advisor()
    except Exception as e:                       # corpus/index problems
        raise APIError(503, "rag_unavailable", f"RAG index could not be loaded: {e}") from e


@router.get("/status", response_model=RagStatus, responses=_err)
async def status():
    if not load_flags()["rag_enabled"]:
        return {"enabled": False, "llm_available": False, "model": None, "n_docs": 0, "n_chunks": 0,
                "sources": [], "built_at": None}
    adv = await run_in_threadpool(_advisor)
    return {"enabled": True, **adv.status()}


@router.post("/advise", response_model=AdviceResponse, responses=_err)
async def advise(req: AdviseRequest):
    """Cited guidance for a detect/localize result (e.g. the Scenario Lab output)."""
    adv = await run_in_threadpool(_advisor)
    out = await run_in_threadpool(adv.advise, req.analysis.model_dump(), req.question, req.k)
    _emit("Advisor consulted for current analysis", req.question, out)
    return out


@router.post("/query", response_model=AdviceResponse, responses=_err)
async def query(req: RagQueryRequest):
    """Free-text question answered from the knowledge base."""
    adv = await run_in_threadpool(_advisor)
    out = await run_in_threadpool(adv.ask, req.question.strip(), req.k)
    _emit("Advisor question", req.question, out)
    return out


def _emit(what: str, question, out: dict) -> None:
    src = out["citations"][0]["section"] if out["citations"] else "—"
    q = f' "{question[:80]}"' if question else ""
    bus.publish("rag", "warn" if out["warnings"] else "info",
                f"{what}{q} → {out['mode']} answer, {len(out['citations'])} sources (top: {src})",
                {"mode": out["mode"], "warnings": out["warnings"]})
