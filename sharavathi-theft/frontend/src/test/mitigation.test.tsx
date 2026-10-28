import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Mitigation from "../pages/Mitigation";

afterEach(() => vi.unstubAllGlobals());

function mockApi(mitigation_enabled: boolean) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const body = url.includes("/api/features")
      ? { mitigation_enabled, gnn_enabled: true, gnn_available: false }
      : { mitigation_enabled, strategies: [
          { id: "load_balance", label: "Load balancing", enabled: false },
          { id: "redispatch", label: "Generation redispatch", enabled: false },
          { id: "shed", label: "Controlled load shedding", enabled: false }] };
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={qc}><MemoryRouter><Mitigation /></MemoryRouter></QueryClientProvider>);
}

describe("Mitigation tab (07 stub)", () => {
  it("flag off: Coming soon badge, disabled strategy picker, empty plan panel", async () => {
    mockApi(false);
    expect(await screen.findByText(/Coming soon/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("radio", { name: /Load balancing/ })).toBeDisabled());
    expect(screen.getByRole("button", { name: "Plan" })).toBeDisabled();
    expect(screen.getByLabelText("Empty plan panel")).toBeInTheDocument();
  });

  it("flag on: controls become active", async () => {
    mockApi(true);
    await waitFor(() => expect(screen.getByRole("button", { name: "Plan" })).toBeEnabled());
    expect(screen.queryByText(/Coming soon/)).not.toBeInTheDocument();
  });
});
