/** Activity console state: the live event log (from /ws/events) + UI toggles. */
import { create } from "zustand";

export type Level = "debug" | "info" | "warn" | "alert" | "error";
export const LEVELS: Level[] = ["debug", "info", "warn", "alert", "error"];
export const CATEGORIES = ["system", "api", "simulation", "detection", "localization", "rag", "replay",
  "batch", "mitigation", "user"] as const;

export interface AppEvent {
  seq: number; ts: number; level: Level; category: string; message: string;
  data: Record<string, unknown>; source: "server" | "client"; request_id: string | null;
}

const MAX = 2000;

interface ConsoleState {
  events: AppEvent[];
  connected: boolean;
  open: boolean;
  unreadAlerts: number;
  minLevel: Level;
  hidden: Set<string>;            // categories filtered out
  search: string;
  paused: boolean;
  clearedAt: number;              // events with seq ≤ this are hidden after "Clear"
  dockHeight: number;
  setDockHeight: (h: number) => void;
  addEvents: (evs: AppEvent[]) => void;
  setConnected: (v: boolean) => void;
  setOpen: (v: boolean) => void;
  toggleOpen: () => void;
  setMinLevel: (l: Level) => void;
  toggleCategory: (c: string) => void;
  setSearch: (s: string) => void;
  setPaused: (v: boolean) => void;
  clear: () => void;
}

export const useConsoleStore = create<ConsoleState>((set) => ({
  events: [], connected: false, open: false, unreadAlerts: 0,
  minLevel: "info", hidden: new Set(), search: "", paused: false, clearedAt: 0,
  dockHeight: 300, setDockHeight: (dockHeight) => set({ dockHeight }),
  addEvents: (evs) => set((s) => {
    const last = s.events.length ? s.events[s.events.length - 1].seq : 0;
    const fresh = evs.filter((e) => e.seq > last);             // backlog after reconnect is de-duplicated
    if (!fresh.length) return s;
    const alerts = s.open ? 0 : fresh.filter((e) => e.level === "alert" || e.level === "error").length;
    return { events: [...s.events, ...fresh].slice(-MAX), unreadAlerts: s.unreadAlerts + alerts };
  }),
  setConnected: (v) => set({ connected: v }),
  setOpen: (v) => set({ open: v, ...(v ? { unreadAlerts: 0 } : {}) }),
  toggleOpen: () => set((s) => ({ open: !s.open, ...(!s.open ? { unreadAlerts: 0 } : {}) })),
  setMinLevel: (l) => set({ minLevel: l }),
  toggleCategory: (c) => set((s) => {
    const hidden = new Set(s.hidden);
    hidden.has(c) ? hidden.delete(c) : hidden.add(c);
    return { hidden };
  }),
  setSearch: (search) => set({ search }),
  setPaused: (paused) => set({ paused }),
  clear: () => set((s) => ({ clearedAt: s.events.length ? s.events[s.events.length - 1].seq : 0, unreadAlerts: 0 })),
}));

export function filterEvents(evs: AppEvent[], s: Pick<ConsoleState, "minLevel" | "hidden" | "search" | "clearedAt">): AppEvent[] {
  const floor = LEVELS.indexOf(s.minLevel);
  const q = s.search.trim().toLowerCase();
  return evs.filter((e) => e.seq > s.clearedAt && LEVELS.indexOf(e.level) >= floor && !s.hidden.has(e.category)
    && (!q || e.message.toLowerCase().includes(q) || e.category.includes(q) || (e.request_id ?? "").includes(q)));
}
