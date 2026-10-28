import { lazy, Suspense, useEffect } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useFeatures, useHealth } from "./api/hooks";
import { ConsoleDock } from "./components/ConsoleDock";
import { DisclaimerBanner } from "./components/DisclaimerBanner";
import { logUser, useEventStream } from "./lib/activity";
import { useConsoleStore } from "./store/useConsoleStore";
import { Skeleton, Toaster } from "./components/ui";
import { useScenarioStore } from "./store/useScenarioStore";

const Overview = lazy(() => import("./pages/Overview"));
const ScenarioLab = lazy(() => import("./pages/ScenarioLab"));
const Replay = lazy(() => import("./pages/Replay"));
const Metrics = lazy(() => import("./pages/Metrics"));
const SensorStudy = lazy(() => import("./pages/SensorStudy"));
const Mitigation = lazy(() => import("./pages/Mitigation"));
const ConsolePage = lazy(() => import("./pages/Console"));

const NAV = [
  { to: "/", label: "Overview", icon: "⌁" },
  { to: "/lab", label: "Scenario Lab", icon: "◎" },
  { to: "/replay", label: "Replay", icon: "▷" },
  { to: "/metrics", label: "Metrics", icon: "▤" },
  { to: "/sensors", label: "Sensor Study", icon: "◉" },
  { to: "/console", label: "Console", icon: "≡" },
];

const PAGE_NAMES: Record<string, string> = {
  "/": "Overview", "/lab": "Scenario Lab", "/replay": "Replay", "/metrics": "Metrics",
  "/sensors": "Sensor Study", "/console": "Console", "/mitigation": "Mitigation",
};

export default function App() {
  const features = useFeatures();
  const health = useHealth();
  const { theme, toggleTheme } = useScenarioStore();
  const mitigationOn = !!features.data?.mitigation_enabled;
  const apiOk = health.data?.status === "ok";
  const { unreadAlerts, connected, toggleOpen, open, dockHeight } = useConsoleStore();
  const location = useLocation();
  useEventStream();
  useEffect(() => { logUser(`Opened ${PAGE_NAMES[location.pathname] ?? location.pathname} page`, { path: location.pathname }); }, [location.pathname]);

  return (
    <div className="flex h-full flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 btn">Skip to content</a>
      <DisclaimerBanner />
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <nav aria-label="Main" className="flex w-full shrink-0 flex-col border-b border-line bg-surface-1 p-3 md:w-56 md:border-b-0 md:border-r">
          <div className="mb-5 px-2">
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-3">220 kV · simulated</div>
            <div className="text-base font-semibold leading-tight">Sharavathi Corridor<br />Anomaly Control Room</div>
          </div>
          <ul className="flex flex-wrap gap-1 md:block md:space-y-0.5">
            {NAV.map((n) => (
              <li key={n.to}>
                <NavLink to={n.to} end={n.to === "/"}
                  className={({ isActive }) => `flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm ${isActive ? "bg-surface-3 font-semibold text-ink-1" : "text-ink-2 hover:bg-surface-2 hover:text-ink-1"}`}>
                  <span aria-hidden className="w-4 text-center text-accent">{n.icon}</span>{n.label}
                </NavLink>
              </li>
            ))}
            <li>
              <NavLink to="/mitigation"
                className={({ isActive }) => `flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm ${isActive ? "bg-surface-3 font-semibold text-ink-1" : "text-ink-3 hover:bg-surface-2"}`}>
                <span aria-hidden className="w-4 text-center">{mitigationOn ? "⚙" : "🔒"}</span>Mitigation
                {!mitigationOn && <span className="ml-auto rounded bg-surface-3 px-1.5 text-[10px] text-ink-2">soon</span>}
              </NavLink>
            </li>
          </ul>
          <div className="mt-3 flex flex-wrap items-center gap-2 px-2 text-xs text-ink-3 md:mt-auto md:block md:space-y-2">
            <div className="flex items-center gap-1.5" aria-live="polite">
              <span aria-hidden className={apiOk ? "text-good" : "text-critical"}>●</span>
              API {apiOk ? "online" : health.isError ? "offline" : health.data?.status ?? "…"}
            </div>
            {health.data && <div className="mono text-[10px]">{health.data.model_version} · {health.data.dataset_hash}</div>}
            <button className="btn w-full justify-between" onClick={toggleOpen} aria-pressed={open} data-testid="console-toggle"
                    aria-label={`Activity console${unreadAlerts ? `, ${unreadAlerts} new alerts` : ""}`}>
              <span><span aria-hidden className={connected ? "text-good" : "text-ink-3"}>●</span> Console <kbd className="mono text-[10px] opacity-70">`</kbd></span>
              {unreadAlerts > 0 && <span className="rounded-full bg-critical px-1.5 text-[10px] font-bold text-white" data-testid="alert-badge">{unreadAlerts}</span>}
            </button>
            <button className="btn w-full" onClick={() => { toggleTheme(); logUser(`Switched to ${theme === "dark" ? "light" : "dark"} theme`); }} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
              {theme === "dark" ? "☀ Light" : "☾ Dark"}
            </button>
          </div>
        </nav>
        <main id="main" className="min-w-0 flex-1 overflow-y-auto p-3 md:p-5" style={open ? { paddingBottom: dockHeight + 16 } : undefined}>
          <Suspense fallback={<Skeleton h={400} />}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/lab" element={<ScenarioLab />} />
              <Route path="/replay" element={<Replay />} />
              <Route path="/metrics" element={<Metrics />} />
              <Route path="/sensors" element={<SensorStudy />} />
              <Route path="/mitigation" element={<Mitigation />} />
              <Route path="/console" element={<ConsolePage />} />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </Suspense>
        </main>
      </div>
      <ConsoleDock />
      <Toaster />
    </div>
  );
}
