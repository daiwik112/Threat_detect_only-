import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { analyze, network } from "./fixtures";

// Plotly cannot render in jsdom — replace the chart surface with a stub.
vi.mock("plotly.js-dist-min", () => ({ default: { downloadImage: vi.fn() } }));
vi.mock("react-plotly.js/factory", () => ({ default: () => () => <div data-testid="plot" /> }));
// React Flow needs layout APIs jsdom lacks; a stub keeps the page test focused on data flow.
vi.mock("../components/NetworkMap", () => ({
  NetworkMap: ({ overlay }: { overlay?: { estimate?: unknown } }) => (
    <div data-testid="network-map">{overlay?.estimate ? <span data-testid="localization-marker" /> : null}</div>
  ),
  LoadingLegend: () => null,
}));

import { StatusBadge } from "../components/ui";
import { DetectionCard } from "../components/charts";
import ScenarioLab from "../pages/ScenarioLab";

function wrap(ui: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}><MemoryRouter>{ui}</MemoryRouter></QueryClientProvider>);
}

afterEach(() => vi.unstubAllGlobals());

describe("StatusBadge", () => {
  it("pairs status colour with an icon and a text label", () => {
    render(<StatusBadge status="suspicious" />);
    const b = screen.getByTestId("status-badge");
    expect(b).toHaveTextContent("Suspicious");
    expect(b).toHaveTextContent("▲");
  });
});

describe("DetectionCard", () => {
  it("shows the score and never claims theft is confirmed", () => {
    render(<DetectionCard detect={analyze.detect} />);
    expect(screen.getByText("1.000")).toBeInTheDocument();
    expect(screen.getByText(/suspected, not confirmed/i)).toBeInTheDocument();
  });
});

describe("ScenarioLab", () => {
  it("Run → Suspicious badge + marker; Reveal truth shows the km error", async () => {
    const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
      const body = url.includes("/api/network") ? network
        : url.includes("/api/analyze") ? analyze
        : url.includes("/api/scenarios") ? { total: 0, items: [] }
        : url.includes("/api/sample-requests") ? [] : {};
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetchMock);
    wrap(<ScenarioLab />);
    await userEvent.click(screen.getByTestId("run-button"));
    await waitFor(() => expect(screen.getByTestId("status-badge")).toHaveAttribute("data-status", "suspicious"));
    expect(screen.getByTestId("localization-marker")).toBeInTheDocument();
    const call = fetchMock.mock.calls.find(([u]) => String(u).includes("/api/analyze"))!;
    expect(JSON.parse((call[1] as RequestInit).body as string)).toMatchObject({ theft: { km: 150, magnitude_mw: 20 } });
    await userEvent.click(screen.getByTestId("reveal-truth"));
    expect(screen.getByTestId("truth-panel")).toHaveTextContent("error 5.0 km");
    expect(screen.getByTestId("truth-panel")).toHaveTextContent("inside the 80% interval");
  });

  it("shows an error state when the API is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    wrap(<ScenarioLab />);
    await waitFor(() => expect(screen.getAllByRole("alert")[0]).toHaveTextContent(/network_error/));
  });
});
