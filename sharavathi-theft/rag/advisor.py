"""
rag/advisor.py — The RAG mitigation advisor (recommendation-only).

    from rag.advisor import get_advisor
    adv = get_advisor()
    adv.advise({"detect": {...}, "localize": {...}}, question=None)   -> AdviceResponse dict
    adv.ask("How do I tell a sensor fault from extraction?")          -> AdviceResponse dict
"""

from __future__ import annotations

import re
from functools import lru_cache

from rag.generator import (ExtractiveGenerator, LLMGenerator, LLMUnavailable, credentials_present,
                           sdk_installed, validate_citations)
from rag.ingest import load_config, load_or_build
from rag.retriever import Hit, Retriever

GUIDANCE_SOURCES = {"playbook", "external"}
DISCLAIMER = ("Advisory output from a simulated research tool. Recommendations require review and "
              "approval by qualified personnel; no control action is taken by this system.")

CAUSE_TAGS = {
    "extraction": {"extraction", "field", "patrol", "evidence"},
    "sensor_error": {"sensor", "metering", "data-quality"},
    "other": {"outage", "network", "protection"},
}
CAUSE_TEXT = {
    "extraction": ("suspected unauthorized extraction pre-dispatch desk review checks planning the patrol "
                   "section estimated kilometre interval candidate taps on site"),
    "sensor_error": "sensor metering fault triage steps KCL redundant measurements calibration bias drift",
    "other": "line outage breaker protection restoration network event",
}
_CONTROL = re.compile(
    r"\b(open|close|trip|operate|switch\s+(off|on)|disconnect|de-?energi[sz]e|isolate|shed|"
    r"redispatch\s+now|send\s+(a\s+)?command|scada|turn\s+off|cut\s+(the\s+)?power)\b", re.I)
REFUSAL_NOTE = ("Note: this advisor cannot issue, schedule or execute control actions (breakers, "
                "switching, dispatch or shedding). It only provides cited recommendations for "
                "qualified operators to review under the utility's procedures.")
_SECTION_NAMES = {"SHARAVATHI": "Sharavathi", "SHIMOGA": "Shimoga", "SIRA_KBCROSS": "Sira/KB-Cross",
                  "NELAMANGALA": "Nelamangala", "PEENYA": "Peenya"}


def _sec(name: str) -> str:
    return " → ".join(_SECTION_NAMES.get(b, b) for b in name.split("-"))


def situation(analysis: dict) -> tuple[str, str | None]:
    """One-paragraph description of the analysis + the dominant cause key."""
    det = analysis.get("detect") or {}
    loc = analysis.get("localize") or {}
    if not det:
        return "No analysis supplied.", None
    lc = det.get("likely_cause") or {}
    cause = max(lc, key=lc.get) if lc else None
    if det.get("status") != "suspicious":
        return (f"Status normal (anomaly score {det.get('anomaly_score', 0):.2f}); no anomaly above "
                f"the alarm threshold.", None)
    parts = [f"Status suspicious (anomaly score {det.get('anomaly_score', 0):.2f})."]
    if cause:
        label = {"extraction": "suspected unauthorized extraction", "sensor_error": "a metering/sensor fault",
                 "other": "a network event such as a line outage"}[cause]
        parts.append(f"Evidence is most consistent with {label} ({lc[cause] * 100:.0f}% of anomaly evidence).")
    if cause == "extraction" and loc:
        secs = loc.get("ranked_sections") or []
        if secs:
            parts.append(f"Most likely section {_sec(secs[0]['section'])} ({secs[0]['prob'] * 100:.0f}%).")
        lo, hi = loc.get("interval_km", [None, None])
        if lo is not None:
            parts.append(f"Estimated location km {loc.get('est_km', 0):.0f}, 80% interval km {lo:.0f}–{hi:.0f}.")
        unexplained = sum((r.get("r_p_mw") or 0) for r in det.get("residuals", []))
        if unexplained > 0.5:
            parts.append(f"About {unexplained:.0f} MW unexplained.")
    return " ".join(parts), cause


class Advisor:
    def __init__(self, index: dict | None = None, cfg: dict | None = None, llm_client=None):
        self.cfg = cfg or load_config()
        self.index = index or load_or_build(self.cfg)
        self.retriever = Retriever(self.index)
        self.extractive = ExtractiveGenerator(self.index["word"])
        self._llm_client = llm_client          # injected in tests
        self.llm = LLMGenerator(self.cfg.get("model", "claude-opus-5-5"),
                                self.cfg.get("effort", "medium"), client=llm_client)

    @property
    def llm_available(self) -> bool:
        if not self.cfg.get("llm_enabled", True):
            return False
        return self._llm_client is not None or (sdk_installed() and credentials_present())

    def status(self) -> dict:
        return {"llm_available": self.llm_available,
                "model": self.cfg.get("model") if self.llm_available else None,
                "n_docs": len(self.retriever.documents), "n_chunks": len(self.retriever.chunks),
                "sources": self.retriever.documents, "built_at": self.index.get("built_at"),
                "index_hash": self.index.get("hash")}

    # ------------------------------------------------------------------
    def _run(self, query: str, context: str, question: str, boost: set[str] | None, k: int,
             source_types: set[str] | None = None) -> dict:
        control = bool(_CONTROL.search(question or ""))
        if control:                                   # steer toward the governance guidance
            boost = (boost or set()) | {"governance", "safety"}
            source_types = GUIDANCE_SOURCES
        hits = self.retriever.search(query, k=k, boost_tags=boost, source_types=source_types,
                                     max_per_doc=3 if source_types else 2)
        g = self.retriever.best_with_tag(query, "governance")
        if control and g:                             # control requests: governance leads, by rule
            hits = [g] + [h for h in hits if h.chunk["id"] != g.chunk["id"]][: k - 1]
        elif g and not any("governance" in h.chunk["tags"] for h in hits):   # always carry the approval rule
            hits = hits[: k - 1] + [g] if len(hits) >= k else hits + [g]

        warnings: list[str] = []
        mode, model, answer = "extractive", None, None
        if self.llm_available:
            try:
                answer = self.llm.generate(context, question, hits)
                mode, model = "llm", self.llm.model
            except LLMUnavailable as e:
                warnings.append(f"LLM unavailable ({e}); showing extractive answer.")
        if answer is None:
            extra = ["", "Approval: these are recommendations for qualified personnel to review; "
                         "nothing is executed automatically."]
            answer = self.extractive.generate(context, question, hits, extra)
        if control and REFUSAL_NOTE not in answer:
            answer = f"{REFUSAL_NOTE}\n\n{answer}"
        answer = re.sub(r"(?i)theft (is )?confirmed", "suspected unauthorized extraction", answer)

        answer, cited, cite_warn = validate_citations(answer, len(hits))
        warnings += cite_warn
        return {
            "mode": mode, "answer": answer, "query": query, "model": model,
            "citations": [self._citation(n, h, n in cited) for n, h in enumerate(hits, 1)],
            "requires_human_approval": True, "warnings": warnings, "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _citation(n: int, h: Hit, cited: bool) -> dict:
        c = h.chunk
        return {"n": n, "chunk_id": c["id"], "doc_id": c["doc_id"], "title": c["title"],
                "section": c["section"], "source_type": c["source_type"], "score": round(h.score, 4),
                "cited": cited, "snippet": c["text"][:600]}

    def advise(self, analysis: dict, question: str | None = None, k: int | None = None) -> dict:
        context, cause = situation(analysis)
        q = question or "What should investigators and operators check and do next?"
        query = " ".join(filter(None, [q, context, CAUSE_TEXT.get(cause or "", "")]))
        # situational advice draws on guidance documents only (not the project's spec files)
        return self._run(query, context, q, CAUSE_TAGS.get(cause or ""), k or self.cfg.get("top_k", 6),
                         source_types=GUIDANCE_SOURCES)

    def ask(self, question: str, k: int | None = None) -> dict:
        context = "General question (no specific analysis attached)."
        return self._run(question, context, question, None, k or self.cfg.get("top_k", 6))


@lru_cache(maxsize=1)
def get_advisor() -> Advisor:
    return Advisor()
