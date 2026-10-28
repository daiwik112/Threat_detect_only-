/**
 * RAG mitigation advisor panel (09). Shows a cited answer; [n] chips open the source passage.
 * With `analysis`, posts to /api/rag/advise; without, it is a free-text Q&A over /api/rag/query.
 */
import { useState, type ReactNode } from "react";
import { useAdvise, useRagQuery, useRagStatus } from "../api/hooks";
import type { AdviceResponse, DetectResponse, LocalizeResponse, RagCitation } from "../api/types";
import { logUser } from "../lib/activity";
import { Card, ErrorState } from "./ui";

function renderAnswer(text: string, onCite: (n: number) => void, active: number | null): ReactNode {
  return text.split("\n").map((line, i) => {
    if (!line.trim()) return <div key={i} className="h-2" />;
    const parts = line.split(/(\[\d+\])/g).map((part, j) => {
      const m = part.match(/^\[(\d+)\]$/);
      if (!m) return <span key={j}>{part}</span>;
      const n = Number(m[1]);
      return (
        <button key={j} type="button" onClick={() => onCite(n)} aria-label={`Show source ${n}`} aria-pressed={active === n}
                className={`mx-0.5 rounded px-1 text-[11px] font-semibold align-super mono ${active === n ? "bg-accent text-white" : "bg-surface-3 text-accent hover:bg-accent hover:text-white"}`}>
          {n}
        </button>
      );
    });
    const heading = /^(Assessment|Guidance from|Approval|Note):/.test(line);
    return <p key={i} className={`text-sm leading-relaxed text-ink-1 ${heading ? "font-semibold" : ""}`}>{parts}</p>;
  });
}

const SOURCE_LABEL: Record<string, string> = { playbook: "playbook · illustrative", external: "utility document", project_doc: "project spec" };

function Sources({ citations, active, setActive }: { citations: RagCitation[]; active: number | null; setActive: (n: number | null) => void }) {
  return (
    <ol className="mt-3 space-y-1.5" aria-label="Sources">
      {citations.map((c) => (
        <li key={c.n} className={`rounded-md border p-2 text-xs ${active === c.n ? "border-accent bg-surface-2" : "border-line"}`}>
          <button type="button" className="flex w-full items-start gap-2 text-left" aria-expanded={active === c.n}
                  onClick={() => setActive(active === c.n ? null : c.n)}>
            <span className="mono font-semibold text-accent">[{c.n}]</span>
            <span className="flex-1">
              <span className="font-medium text-ink-1">{c.section}</span>
              <span className="ml-2 text-ink-3">{SOURCE_LABEL[c.source_type] ?? c.source_type}{c.cited ? "" : " · retrieved, not cited"}</span>
            </span>
            <span className="mono text-ink-3">{c.score.toFixed(2)}</span>
          </button>
          {active === c.n && <p className="mt-2 whitespace-pre-line border-t border-line pt-2 text-ink-2" data-testid="source-snippet">{c.snippet}</p>}
        </li>
      ))}
    </ol>
  );
}

export function AdviceView({ advice }: { advice: AdviceResponse }) {
  const [active, setActive] = useState<number | null>(null);
  return (
    <div data-testid="advice">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px]">
        <span className="rounded-full border border-line bg-surface-2 px-2 py-0.5 font-semibold text-ink-2">
          {advice.mode === "llm" ? `LLM · ${advice.model}` : "Extractive (offline)"}
        </span>
        <span className="rounded-full border border-warning/50 bg-warning/10 px-2 py-0.5 text-ink-1">◆ Requires human approval</span>
      </div>
      {advice.warnings.length > 0 && (
        <div role="status" className="mb-2 rounded-md border border-warning/50 bg-warning/10 p-2 text-xs">
          {advice.warnings.map((w) => <div key={w}>◆ {w}</div>)}
        </div>
      )}
      <div className="space-y-1">{renderAnswer(advice.answer, (n) => setActive(active === n ? null : n), active)}</div>
      <Sources citations={advice.citations} active={active} setActive={setActive} />
      <p className="mt-2 text-[11px] text-ink-3">{advice.disclaimer}</p>
    </div>
  );
}

export function AdvisorPanel({ analysis, title = "Mitigation advisor (RAG)" }: {
  analysis?: { detect: DetectResponse; localize: LocalizeResponse } | null; title?: string;
}) {
  const status = useRagStatus();
  const advise = useAdvise();
  const ask = useRagQuery();
  const [question, setQuestion] = useState("");
  const mut = analysis ? advise : ask;
  const disabled = status.data && !status.data.enabled;

  const submit = () => {
    logUser(analysis ? "Asked the RAG advisor about the current result" : `Asked the RAG advisor: "${question.trim().slice(0, 80)}"`);
    if (analysis) advise.mutate({ analysis, question: question.trim() || null, k: 6 });
    else if (question.trim().length >= 3) ask.mutate({ question: question.trim(), k: 6 });
  };

  return (
    <Card title={title}
      subtitle={status.data?.enabled
        ? <>Cited guidance from {status.data.n_docs} documents ({status.data.n_chunks} passages) · {status.data.llm_available ? `LLM: ${status.data.model}` : "offline extractive mode"}</>
        : "Retrieval-augmented, recommendation-only advice"}>
      {disabled ? <p className="text-sm text-ink-2">The RAG advisor is disabled (<span className="mono">rag_enabled: false</span>).</p> : (
        <>
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); submit(); }}>
            <label htmlFor={analysis ? "advisor-q-lab" : "advisor-q"} className="sr-only">Question for the advisor</label>
            <input id={analysis ? "advisor-q-lab" : "advisor-q"} type="text" value={question} onChange={(e) => setQuestion(e.target.value)}
                   placeholder={analysis ? "Optional follow-up, e.g. what should the patrol look for?" : "Ask, e.g. how do I tell a sensor fault from extraction?"}
                   className="flex-1 rounded-md border border-line bg-surface-2 px-3 py-1.5 text-sm text-ink-1" />
            <button className="btn btn-primary" type="submit" disabled={mut.isPending || (!analysis && question.trim().length < 3)} data-testid="ask-advisor">
              {mut.isPending ? "Retrieving…" : analysis ? "Ask the advisor" : "Ask"}
            </button>
          </form>
          <div className="mt-3">
            {mut.isError && <ErrorState error={mut.error} />}
            {mut.data && <AdviceView advice={mut.data} />}
          </div>
        </>
      )}
    </Card>
  );
}
