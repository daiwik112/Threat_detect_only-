import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AdvisorPanel } from "../components/AdvisorPanel";
import { analyze } from "./fixtures";

afterEach(() => vi.unstubAllGlobals());

const status = { enabled: true, llm_available: false, model: null, n_docs: 18, n_chunks: 164, sources: [], built_at: "x" };
const advice = {
  mode: "extractive", model: null, query: "q", requires_human_approval: true, warnings: [],
  disclaimer: "Advisory output from a simulated research tool.",
  answer: "Assessment: Status suspicious.\n\nGuidance from the knowledge base:\n1. Patrol the section with the highest probability first. [1]\n2. Treat the interval as a search area. [2]",
  citations: [
    { n: 1, chunk_id: "field_verification#2", doc_id: "field_verification", title: "Field verification", section: "Field verification › Planning the patrol", source_type: "playbook", score: 0.41, cited: true, snippet: "Patrol the section with the highest section probability first." },
    { n: 2, chunk_id: "safety_governance#1", doc_id: "safety_governance", title: "Safety", section: "Safety › Know the model's limits", source_type: "playbook", score: 0.2, cited: true, snippet: "Treat the interval and the ranked taps as a search area." },
  ],
};

function setup(analysis?: boolean) {
  const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
    const body = url.includes("/api/rag/status") ? status : advice;
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={qc}>
    <AdvisorPanel analysis={analysis ? { detect: analyze.detect, localize: analyze.localize } : undefined} />
  </QueryClientProvider>);
  return fetchMock;
}

describe("AdvisorPanel (RAG)", () => {
  it("advise mode: posts the analysis, renders cited answer, citation chip reveals the source", async () => {
    const fetchMock = setup(true);
    expect(await screen.findByText(/18 documents/)).toBeInTheDocument();
    await userEvent.click(screen.getByTestId("ask-advisor"));
    expect(await screen.findByTestId("advice")).toHaveTextContent("Requires human approval");
    const call = fetchMock.mock.calls.find(([u]) => String(u).includes("/api/rag/advise"))!;
    const body = JSON.parse((call[1] as RequestInit).body as string);
    expect(body.analysis.detect.status).toBe("suspicious");
    expect(screen.queryByTestId("source-snippet")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Show source 1" }));
    expect(screen.getByTestId("source-snippet")).toHaveTextContent("highest section probability");
    expect(screen.getByText("Extractive (offline)")).toBeInTheDocument();
  });

  it("question mode: needs ≥3 characters, then queries /api/rag/query", async () => {
    const fetchMock = setup(false);
    const btn = await screen.findByTestId("ask-advisor");
    expect(btn).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Question for the advisor"), "sensor drift triage");
    await userEvent.click(btn);
    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u).includes("/api/rag/query"))).toBe(true));
    expect(await screen.findByTestId("advice")).toBeInTheDocument();
  });
});
