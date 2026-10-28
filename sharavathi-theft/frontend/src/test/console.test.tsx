import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { ConsoleView } from "../components/ConsoleDock";
import { filterEvents, useConsoleStore, type AppEvent } from "../store/useConsoleStore";

const ev = (seq: number, level: AppEvent["level"], category: string, message: string, data = {}): AppEvent =>
  ({ seq, ts: 1_700_000_000 + seq, level, category, message, data, source: category === "user" ? "client" : "server", request_id: "r" + seq });

const EVENTS = [
  ev(1, "debug", "api", "GET /api/health → 200"),
  ev(2, "info", "user", "Ran analysis: 20 MW extraction at km 150"),
  ev(3, "info", "simulation", "Injected 20.0 MW unmetered load at km 150 (TAP_150)"),
  ev(4, "alert", "detection", "SUSPECTED UNAUTHORIZED EXTRACTION near km 145", { km: 145 }),
  ev(5, "warn", "mitigation", "Mitigation plan requested — blocked"),
];

beforeEach(() => useConsoleStore.setState({
  events: [], unreadAlerts: 0, open: false, minLevel: "info", hidden: new Set(), search: "", paused: false, clearedAt: 0, connected: true,
}));

describe("console store", () => {
  it("de-duplicates by seq, caps, and counts unread alerts while closed", () => {
    const s = useConsoleStore.getState();
    s.addEvents(EVENTS);
    s.addEvents(EVENTS.slice(2));                       // reconnect backlog overlap
    expect(useConsoleStore.getState().events).toHaveLength(5);
    expect(useConsoleStore.getState().unreadAlerts).toBe(1);
    s.setOpen(true);
    expect(useConsoleStore.getState().unreadAlerts).toBe(0);
  });

  it("filters by level, category, search and clear", () => {
    const base = { minLevel: "info" as const, hidden: new Set<string>(), search: "", clearedAt: 0 };
    expect(filterEvents(EVENTS, base).map((e) => e.seq)).toEqual([2, 3, 4, 5]);       // debug hidden
    expect(filterEvents(EVENTS, { ...base, minLevel: "alert" }).map((e) => e.seq)).toEqual([4]);
    expect(filterEvents(EVENTS, { ...base, hidden: new Set(["user", "simulation"]) }).map((e) => e.seq)).toEqual([4, 5]);
    expect(filterEvents(EVENTS, { ...base, search: "km 150" }).map((e) => e.seq)).toEqual([2, 3]);
    expect(filterEvents(EVENTS, { ...base, clearedAt: 4 }).map((e) => e.seq)).toEqual([5]);
  });
});

describe("ConsoleView", () => {
  it("renders the timeline, highlights alerts and expands details with a Lab link", async () => {
    useConsoleStore.getState().addEvents(EVENTS);
    render(<MemoryRouter><ConsoleView /></MemoryRouter>);
    const rows = screen.getAllByTestId("console-row");
    expect(rows).toHaveLength(4);
    const alert = rows.find((r) => r.dataset.level === "alert")!;
    expect(alert).toHaveTextContent("SUSPECTED UNAUTHORIZED EXTRACTION near km 145");
    await userEvent.click(within(alert).getByRole("button"));
    expect(within(alert).getByRole("button", { name: /Open km 145 in Scenario Lab/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "user" }));             // hide user events
    expect(screen.getAllByTestId("console-row")).toHaveLength(3);
    await userEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.queryAllByTestId("console-row")).toHaveLength(0);
  });
});
