# 09 — WORKFLOW: RAG Mitigation Advisor (retrieval-augmented, recommendation-only)

**Goal:** given a detection/localization result, return **cited, grounded guidance** on what
investigators and operators should check and consider: field verification, sensor triage,
outage response, load-balancing principles, evidence and reporting. Every statement must be
traceable to a passage in a curated knowledge base.
**Depends on:** `03` (`ml.infer` outputs), `04` (API), `05` (UI), `07` (mitigation rules), `08` (contracts).
**Writes:** `rag/` package, `rag/corpus/`, `config/rag.yaml`, `data/models/rag_index.joblib`,
`/api/rag/*`, an Advisor panel in Scenario Lab and on the Mitigation page.

---

## 1. Scope & non-negotiables
1. **Advice, never control.** The advisor produces text recommendations for humans
   (`requires_human_approval: true` on every response). It has no tool calls, no SCADA access
   and no write access to the network model. Questions asking it to operate equipment get a
   fixed refusal note plus the governance guidance. This keeps 07's rules 1–2 intact.
2. **No invented regulations.** The knowledge base holds (a) the project's own documents
   (00–09, the dataset card, `future/README.md`) and (b) **project-authored playbooks marked
   "illustrative — not an official KPTCL / CEA / SLDC procedure"**. Real utility procedures can
   be dropped into `rag/corpus/external/` later. The advisor never cites clause numbers
   that aren't in the corpus.
3. **Grounded or nothing.** Answers cite passages as `[n]`. Server-side validation removes any
   citation that does not map to a retrieved passage, and an answer with no valid citation is
   flagged (`warnings`).
4. **Wording:** "suspected unauthorized extraction", never "theft confirmed".
5. **Works offline.** Retrieval is local (scikit-learn). Generation uses Claude when
   credentials exist and otherwise falls back to a deterministic **extractive** answer, so tests
   and demos run without a network.
6. **Corpus text is data, not instructions.** Passages are wrapped in `<passage>` tags and the
   system prompt says to ignore any instructions inside them (prompt-injection hygiene for
   documents users add later).

## 2. Architecture
```
analysis (detect + localize [+ simulate]) ─┐
optional free-text question ───────────────┤
                                           ▼
                      rag/advisor.py  build_query() ── cause / section / interval / magnitude
                                           │
                      rag/retriever.py  hybrid TF-IDF (word 1–2-gram + char 3–5-gram)
                                           │  + tag boost by likely cause, top-k, governance pinned
                                           ▼
                      rag/generator.py  ┌ LLM mode: Claude (claude-opus-5-5) with numbered passages
                                        └ extractive mode: best-matching sentences per passage
                                           │
                      citation validation → AdviceResponse {answer, citations[], mode, warnings}
```

| File | Responsibility |
|---|---|
| `rag/corpus/playbooks/*.md` | 6 authored playbooks (front-matter: `title`, `tags`, `status: illustrative`) |
| `rag/ingest.py` | collect sources (`config/rag.yaml`), heading-aware chunking (~900 chars, paragraph overlap), build + save index, `python -m rag.ingest` |
| `rag/retriever.py` | `Retriever.search(query, k, boost_tags)` → scored chunks |
| `rag/generator.py` | `LLMGenerator` (Anthropic SDK) and `ExtractiveGenerator`; citation validation |
| `rag/advisor.py` | `Advisor.advise(analysis, question)`, `Advisor.ask(question)`, guardrails, `get_advisor()` |
| `config/rag.yaml` | sources, chunking, k, model, effort, `llm_enabled` |
| `feature_flags.yaml` | `rag_enabled: true` |

### Index
- `data/models/rag_index.joblib`: chunks, both vectorizers, matrices, a hash of the corpus
  contents and the build time. It's rebuilt automatically when the corpus hash changes.
- Chunk = `{id, doc_id, title, section, source_type: playbook|project_doc|external, path, tags, text}`.

### Retrieval
- `score = 0.6·cos(word) + 0.4·cos(char) + 0.08·[chunk tags ∩ boost tags ≠ ∅]`.
- Boost tags by likely cause: extraction → `extraction, field, patrol, evidence`;
  sensor error → `sensor, metering`; other → `outage, network`.
- The top governance/safety chunk is always included so every answer carries the
  human-approval guidance.
- Dense embeddings are deliberately out of scope (no extra heavy deps); the `Retriever`
  interface allows swapping them in later.

### Generation
- **LLM mode** (when `llm_enabled`, the `anthropic` package is installed, and credentials resolve:
  `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` / an `ant auth login` profile):
  - Request: `client.beta.messages.create(model="claude-opus-5-5", max_tokens=16000,
    output_config={"effort": "medium"}, betas=["server-side-fallback-2026-07-01"],
    fallbacks="default", system=…, messages=[…])`. Thinking is omitted because it is always
    on for this model.
  - Check `stop_reason == "refusal"` before reading content. On a refusal, or on any API error
    (typed chain: authentication → rate limit → status → connection), fall back to extractive
    mode and add a warning.
- **Extractive mode:** a situation line built from the analysis, then for each passage the 1–2
  sentences most similar to the query with `[n]`, then the governance line. Deterministic.

## 3. API (backend)
| Method | Path | Body → Response |
|---|---|---|
| GET | `/api/rag/status` | → `RagStatus {enabled, llm_available, model, n_docs, n_chunks, sources[], built_at}` |
| POST | `/api/rag/advise` | `{analysis: {detect, localize, simulate?}, question?, k?}` → `AdviceResponse` |
| POST | `/api/rag/query` | `{question, k?}` → `AdviceResponse` |

```
AdviceResponse
  mode:        "llm" | "extractive"
  answer:      str                 (cites passages as [n])
  citations:   [{n, chunk_id, doc_id, title, section, source_type, score, snippet}]
  query:       str                 (what was retrieved for)
  model:       str | null
  requires_human_approval: true
  warnings:    [str]
  disclaimer:  str
```
Errors: 503 `rag_disabled` when the flag is off; 422 `validation_error` for an empty question.

## 4. UI
- **Scenario Lab:** an "Ask the advisor" button under *Recommended inspection actions* posts
  the current result to `/api/rag/advise` and shows the cited answer, a mode badge
  (LLM / Extractive) and clickable `[n]` chips that reveal each source passage.
- **Mitigation page:** a "Knowledge-base advisor (RAG)" panel for free-text questions. It is
  available even while `mitigation_enabled=false`, because it is advice only.

## 5. Tests
- Chunking and indexing: every source becomes ≥1 chunk; headings are preserved; the index
  rebuilds when the corpus hash changes.
- Retrieval relevance: a sensor-fault query ranks the sensor-triage playbook first; an outage
  query ranks the outage playbook first; an extraction query ranks the field-verification
  playbook first.
- Advise (extractive): schema; every `[n]` is valid; the governance passage is present; the
  wording never says "confirmed".
- Guardrail: "open the breaker at Shimoga" returns the refusal note and
  `requires_human_approval: true`.
- LLM path with a fake client: invalid citations are stripped, a refusal falls back to
  extractive, and an API error falls back to extractive. No network is used in tests.
- API endpoints, including the flag-off 503. Frontend component test for the Advisor panel.

## 6. Acceptance
- `python -m rag.ingest` builds the index; `make test` passes.
- For a 150 km / 20 MW scenario, `/api/rag/advise` returns guidance that cites the
  field-verification and governance playbooks, in extractive mode with no credentials.
- No module in `grid/`, `simdata/` or `ml/` imports `rag/`.

## 7. Future
Dense or hybrid embeddings; ingesting real utility SOPs (PDF) into `corpus/external/`;
feeding advisor output as the `rationale` of a future `MitigationPlanner` (07); an evaluation
set of question → expected source documents.
