/**
 * Activity console: a live timeline of what the app is doing — user actions, API calls,
 * simulations, detections (suspected extraction = ALERT), RAG questions, replay alerts, errors.
 * ConsoleView is used by the bottom dock (toggle with ` or the sidebar button) and the /console page.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CATEGORIES, LEVELS, filterEvents, useConsoleStore, type AppEvent, type Level } from "../store/useConsoleStore";
import { useScenarioStore } from "../store/useScenarioStore";

const LEVEL_STYLE: Record<Level, { icon: string; cls: string }> = {
  debug: { icon: "·", cls: "text-ink-3" },
  info: { icon: "●", cls: "text-accent" },
  warn: { icon: "◆", cls: "text-warning" },
  alert: { icon: "▲", cls: "text-[var(--status-critical-text)]" },
  error: { icon: "✖", cls: "text-[var(--status-critical-text)]" },
};

const time = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false }) +
  "." + String(Math.floor((ts % 1) * 1000)).padStart(3, "0");

function Row({ e, expanded, onToggle }: { e: AppEvent; expanded: boolean; onToggle: () => void }) {
  const navigate = useNavigate();
  const setTheftKm = useScenarioStore((s) => s.setTheftKm);
  const st = LEVEL_STYLE[e.level] ?? LEVEL_STYLE.info;
  const km = typeof e.data.km === "number" ? (e.data.km as number) : null;
  const strong = e.level === "alert" || e.level === "error";
  return (
    <li className={`border-b border-line/50 ${strong ? "bg-critical/10" : ""}`} data-testid="console-row" data-level={e.level} data-category={e.category}>
      <button type="button" onClick={onToggle} aria-expanded={expanded}
              className="grid w-full grid-cols-[6.5rem_1.2rem_6.5rem_1fr] items-start gap-2 px-2 py-1 text-left font-mono text-[12px] hover:bg-surface-2">
        <span className="text-ink-3">{time(e.ts)}</span>
        <span className={st.cls} aria-label={e.level}>{st.icon}</span>
        <span className={`truncate ${e.source === "client" ? "text-good" : "text-ink-2"}`}>{e.category}</span>
        <span className={`whitespace-pre-wrap break-words ${strong ? "font-semibold text-ink-1" : "text-ink-1"}`}>{e.message}</span>
      </button>
      {expanded && (
        <div className="space-y-1 px-2 pb-2 pl-[15.5rem] font-mono text-[11px] text-ink-2">
          <div>seq {e.seq} · {e.source} · level {e.level}{e.request_id ? ` · request ${e.request_id}` : ""}</div>
          {Object.keys(e.data).length > 0 && <pre className="overflow-x-auto rounded bg-surface-2 p-2">{JSON.stringify(e.data, null, 2)}</pre>}
          {km != null && (
            <button className="btn !px-2 !py-0.5 text-xs" onClick={() => { setTheftKm(km); navigate("/lab"); }}>
              Open km {Math.round(km)} in Scenario Lab →
            </button>
          )}
        </div>
      )}
    </li>
  );
}

export function ConsoleView({ height }: { height?: number | string }) {
  const s = useConsoleStore();
  const shown = useMemo(() => filterEvents(s.events, s), [s.events, s.minLevel, s.hidden, s.search, s.clearedAt]);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [follow, setFollow] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (follow && listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [shown.length, follow]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const e of s.events) if (e.seq > s.clearedAt) c[e.level] = (c[e.level] ?? 0) + 1;
    return c;
  }, [s.events, s.clearedAt]);

  const exportJson = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(shown, null, 2)], { type: "application/json" }));
    a.download = `activity-log-${new Date().toISOString().slice(0, 19)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="flex h-full min-h-0 flex-col" style={height ? { height } : undefined}>
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-2 py-1.5 text-xs">
        <span className="flex items-center gap-1" aria-live="polite">
          <span className={s.connected ? "text-good" : "text-critical"} aria-hidden>●</span>{s.connected ? "live" : "reconnecting…"}
        </span>
        <label className="sr-only" htmlFor="console-level">Minimum level</label>
        <select id="console-level" value={s.minLevel} onChange={(e) => s.setMinLevel(e.target.value as Level)} className="!py-0.5 text-xs">
          {LEVELS.map((l) => <option key={l} value={l}>{l}+ ({LEVELS.slice(LEVELS.indexOf(l)).reduce((n, x) => n + (counts[x] ?? 0), 0)})</option>)}
        </select>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Categories">
          {CATEGORIES.map((c) => (
            <button key={c} type="button" aria-pressed={!s.hidden.has(c)} onClick={() => s.toggleCategory(c)}
                    className={`rounded px-1.5 py-0.5 font-mono ${s.hidden.has(c) ? "text-ink-3 line-through" : "bg-surface-3 text-ink-1"}`}>{c}</button>
          ))}
        </div>
        <input type="search" placeholder="search…" value={s.search} onChange={(e) => s.setSearch(e.target.value)}
               aria-label="Search console" className="w-36 rounded border border-line bg-surface-2 px-2 py-0.5 text-xs" />
        <div className="ml-auto flex gap-1">
          <button className="btn !px-2 !py-0.5 text-xs" aria-pressed={s.paused} onClick={() => s.setPaused(!s.paused)}>{s.paused ? "▶ Resume" : "❚❚ Pause"}</button>
          <button className="btn !px-2 !py-0.5 text-xs" aria-pressed={follow} onClick={() => setFollow(!follow)}>{follow ? "⇣ Following" : "⇣ Follow"}</button>
          <button className="btn !px-2 !py-0.5 text-xs" onClick={s.clear}>Clear</button>
          <button className="btn !px-2 !py-0.5 text-xs" onClick={exportJson}>⬇ JSON</button>
        </div>
      </div>
      <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto"
           onScroll={(e) => { const el = e.currentTarget; if (el.scrollHeight - el.scrollTop - el.clientHeight > 40) setFollow(false); }}>
        {shown.length === 0
          ? <p className="p-4 text-center text-xs text-ink-3">{s.events.length ? "No events match the filters." : "Waiting for activity… run a scenario or navigate the app."}</p>
          : <ol aria-label="Activity log">{shown.map((e) => <Row key={e.seq} e={e} expanded={expanded === e.seq} onToggle={() => setExpanded(expanded === e.seq ? null : e.seq)} />)}</ol>}
      </div>
      <div className="border-t border-line px-2 py-1 text-[10px] text-ink-3">
        {shown.length} shown · {s.events.length} received · ▲ alert = suspected extraction (needs field verification) · <span className="text-good">green category</span> = your actions
      </div>
    </div>
  );
}

export function ConsoleDock() {
  const { open, toggleOpen, setOpen, dockHeight: h, setDockHeight: setH } = useConsoleStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "`" && !["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        toggleOpen();
      }
      if (e.key === "Escape" && useConsoleStore.getState().open) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleOpen, setOpen]);

  if (!open) return null;
  return (
    <section aria-label="Activity console" data-testid="console-dock"
             className="fixed inset-x-0 bottom-0 z-40 flex flex-col border-t-2 border-accent bg-surface-1 shadow-2xl" style={{ height: h }}>
      <div className="flex cursor-row-resize items-center justify-between bg-surface-2 px-3 py-1"
           onMouseDown={(e) => {
             const y0 = e.clientY, h0 = h;
             const move = (m: MouseEvent) => setH(Math.min(Math.max(h0 + y0 - m.clientY, 140), window.innerHeight - 80));
             const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
             window.addEventListener("mousemove", move);
             window.addEventListener("mouseup", up);
           }}>
        <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-2">Activity console <kbd className="mono ml-1 font-normal">`</kbd></span>
        <button className="btn !px-2 !py-0 text-xs" onClick={() => setOpen(false)} aria-label="Close console">✕</button>
      </div>
      <div className="min-h-0 flex-1"><ConsoleView /></div>
    </section>
  );
}
