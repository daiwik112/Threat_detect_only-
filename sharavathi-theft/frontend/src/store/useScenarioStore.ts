/**
 * UI-only state (Zustand). Server data stays in TanStack Query; the store keeps only the
 * request being edited, UI toggles, and a short local run history for export/compare.
 */
import { create } from "zustand";
import type { AnalyzeRequest, AnalyzeResponse, SensorProfile } from "../api/types";
import type { ThemeMode } from "../theme";

export type FaultKind = "none" | "bias" | "drift" | "missing" | "line_outage";

export interface ScenarioRequest {
  loadScale: number;
  theftOn: boolean;
  theftKm: number;
  theftMw: number;
  fault: FaultKind;
  faultChannel: string;
  faultParam: number;
  sensorProfile: SensorProfile;
  noiseMult: number;
  seed: number;
}

export const TAP_SPACING_KM = 10;
export const CORRIDOR_KM = 361;
export const DEFAULT_REQUEST: ScenarioRequest = {
  loadScale: 1.0, theftOn: true, theftKm: 150, theftMw: 20,
  fault: "none", faultChannel: "m_SHIMOGA_p", faultParam: 8,
  sensorProfile: "major_only", noiseMult: 1.0, seed: 42,
};

/** Snap km to the nearest 10-km tap (taps exist at 10…360 except substations at 110/320). */
export function snapKm(km: number): number {
  const s = Math.round(Math.min(Math.max(km, TAP_SPACING_KM), 360) / TAP_SPACING_KM) * TAP_SPACING_KM;
  if (s === 110) return km < 110 ? 100 : 120;
  if (s === 320) return km < 320 ? 310 : 330;
  return s;
}

export function toApiRequest(r: ScenarioRequest): AnalyzeRequest {
  return {
    load_scale: r.loadScale,
    theft: r.theftOn && r.theftMw > 0 ? { km: r.theftKm, magnitude_mw: r.theftMw, pf: 0.95 } : null,
    fault:
      r.fault === "none"
        ? null
        : r.fault === "line_outage"
          ? { type: "line_outage", param: r.faultParam, channel: null }
          : { type: r.fault, channel: r.faultChannel, param: r.faultParam },
    sensor_profile: r.sensorProfile,
    noise_mult: r.noiseMult,
    seed: r.seed,
  };
}

/** Inverse of toApiRequest — used by presets (GET /api/sample-requests). */
export function fromApiRequest(a: Partial<AnalyzeRequest>, base: ScenarioRequest = DEFAULT_REQUEST): ScenarioRequest {
  const f = a.fault ?? null;
  return {
    ...base,
    loadScale: a.load_scale ?? 1,
    theftOn: !!a.theft,
    theftKm: a.theft ? snapKm(a.theft.km) : base.theftKm,
    theftMw: a.theft ? a.theft.magnitude_mw : base.theftMw,
    fault: f ? (f.type as FaultKind) : "none",
    faultChannel: f?.channel && f.type !== "line_outage" ? f.channel : base.faultChannel,
    faultParam: f?.param ?? base.faultParam,
    sensorProfile: a.sensor_profile ?? "major_only",
    noiseMult: a.noise_mult ?? 1,
    seed: a.seed ?? 42,
  };
}

export interface HistoryEntry { at: number; request: ScenarioRequest; result: AnalyzeResponse }

interface State {
  request: ScenarioRequest;
  revealTruth: boolean;
  liveMode: boolean;
  focusedTap: string | null;
  history: HistoryEntry[];
  theme: ThemeMode;
  setRequest: (patch: Partial<ScenarioRequest>) => void;
  setTheftKm: (km: number) => void;
  nudgeKm: (deltaKm: number) => void;
  resetRequest: () => void;
  setRevealTruth: (v: boolean) => void;
  setLiveMode: (v: boolean) => void;
  focusTap: (id: string | null) => void;
  pushHistory: (request: ScenarioRequest, r: AnalyzeResponse) => void;
  toggleTheme: () => void;
}

export const useScenarioStore = create<State>((set) => ({
  request: DEFAULT_REQUEST,
  revealTruth: false,
  liveMode: false,
  focusedTap: null,
  history: [],
  theme: (typeof localStorage !== "undefined" && (localStorage.getItem("theme") as ThemeMode)) || "dark",
  setRequest: (patch) => set((s) => ({ request: { ...s.request, ...patch } })),
  setTheftKm: (km) => set((s) => ({ request: { ...s.request, theftKm: snapKm(km), theftOn: true } })),
  nudgeKm: (d) => set((s) => {
    let km = snapKm(s.request.theftKm + d);
    if (km === s.request.theftKm) km = snapKm(s.request.theftKm + 2 * d);   // hop over substations
    return { request: { ...s.request, theftKm: km, theftOn: true } };
  }),
  resetRequest: () => set({ request: DEFAULT_REQUEST }),
  setRevealTruth: (v) => set({ revealTruth: v }),
  setLiveMode: (v) => set({ liveMode: v }),
  focusTap: (id) => set({ focusedTap: id }),
  pushHistory: (request, r) => set((s) => ({
    history: [{ at: Date.now(), request, result: r }, ...s.history].slice(0, 20), revealTruth: false,
  })),
  toggleTheme: () =>
    set((s) => {
      const theme: ThemeMode = s.theme === "dark" ? "light" : "dark";
      localStorage.setItem("theme", theme);
      document.documentElement.dataset.theme = theme;
      return { theme };
    }),
}));
