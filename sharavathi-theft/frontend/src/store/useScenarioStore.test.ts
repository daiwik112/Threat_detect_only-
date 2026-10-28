import { beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_REQUEST, fromApiRequest, snapKm, toApiRequest, useScenarioStore } from "./useScenarioStore";
import type { AnalyzeResponse } from "../api/types";

describe("snapKm", () => {
  it("snaps to the nearest 10-km tap", () => {
    expect(snapKm(153)).toBe(150);
    expect(snapKm(156)).toBe(160);
  });
  it("clamps to the tap range and skips substations at 110/320 km", () => {
    expect(snapKm(0)).toBe(10);
    expect(snapKm(361)).toBe(360);
    expect(snapKm(108)).toBe(100);
    expect(snapKm(112)).toBe(120);
    expect(snapKm(321)).toBe(330);
  });
});

describe("toApiRequest", () => {
  it("maps theft + no fault", () => {
    const r = toApiRequest(DEFAULT_REQUEST);
    expect(r.theft).toEqual({ km: 150, magnitude_mw: 20, pf: 0.95 });
    expect(r.fault).toBeNull();
    expect(r.sensor_profile).toBe("major_only");
  });
  it("drops theft when off or zero MW and maps faults", () => {
    expect(toApiRequest({ ...DEFAULT_REQUEST, theftOn: false }).theft).toBeNull();
    expect(toApiRequest({ ...DEFAULT_REQUEST, theftMw: 0 }).theft).toBeNull();
    expect(toApiRequest({ ...DEFAULT_REQUEST, fault: "bias", faultChannel: "m_PEENYA_q", faultParam: 5 }).fault)
      .toEqual({ type: "bias", channel: "m_PEENYA_q", param: 5 });
    expect(toApiRequest({ ...DEFAULT_REQUEST, fault: "line_outage", faultParam: 12 }).fault)
      .toEqual({ type: "line_outage", param: 12, channel: null });
  });
});

describe("store", () => {
  beforeEach(() => useScenarioStore.setState({ request: DEFAULT_REQUEST, history: [], revealTruth: true }));
  it("setTheftKm snaps and enables theft", () => {
    useScenarioStore.getState().setRequest({ theftOn: false });
    useScenarioStore.getState().setTheftKm(244);
    expect(useScenarioStore.getState().request).toMatchObject({ theftKm: 240, theftOn: true });
  });
  it("history keeps the last 20 runs and hides the truth for a new run", () => {
    for (let i = 0; i < 25; i++) useScenarioStore.getState().pushHistory(DEFAULT_REQUEST, { request_id: String(i) } as AnalyzeResponse);
    const s = useScenarioStore.getState();
    expect(s.history).toHaveLength(20);
    expect(s.history[0].result.request_id).toBe("24");
    expect(s.history[0].request).toEqual(DEFAULT_REQUEST);
    expect(s.revealTruth).toBe(false);
  });
});

describe("06 additions", () => {
  beforeEach(() => useScenarioStore.setState({ request: DEFAULT_REQUEST }));
  it("nudgeKm moves ±10 km and hops over substations", () => {
    useScenarioStore.getState().nudgeKm(10);
    expect(useScenarioStore.getState().request.theftKm).toBe(160);
    useScenarioStore.setState({ request: { ...DEFAULT_REQUEST, theftKm: 100 } });
    useScenarioStore.getState().nudgeKm(10);
    expect(useScenarioStore.getState().request.theftKm).toBe(120);
  });
  it("fromApiRequest inverts toApiRequest", () => {
    const r = { ...DEFAULT_REQUEST, fault: "bias" as const, faultChannel: "m_PEENYA_q", faultParam: 5, theftKm: 240 };
    expect(fromApiRequest(toApiRequest(r))).toEqual(r);
    const outage = fromApiRequest({ load_scale: 0.8, fault: { type: "line_outage", param: 20 } });
    expect(outage).toMatchObject({ theftOn: false, fault: "line_outage", faultParam: 20, loadScale: 0.8 });
  });
});
