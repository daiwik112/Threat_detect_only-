"""
rag/ — Retrieval-augmented mitigation advisor (09_RAG_MODULE.md). Recommendation-only.

ingest     collect corpus → heading-aware chunks → TF-IDF index (data/models/rag_index.joblib)
retriever  hybrid word/char TF-IDF search with cause-based tag boosts
generator  Claude (LLM mode) or deterministic extractive answers; citation validation
advisor    advise(analysis, question) / ask(question); guardrails; get_advisor()

Detection/localization code (grid/, simdata/, ml/) never imports this package.
"""
