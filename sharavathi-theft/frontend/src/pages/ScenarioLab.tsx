import { useCallback, useEffect, useRef, useState } from "react";
import { useAnalyze, useNetwork, useSampleRequests, useScenarios } from "../api/hooks";
import type { AnalyzeResponse } from "../api/types";
import { AdvisorPanel } from "../components/AdvisorPanel";
import { CauseBar, DetectionCard, LocalizationChart, MeasuredVsExpected, ResidualChart, ShapBar } from "../components/charts";
import { FaultInjector, LoadNoiseSliders, MeasurementTable, SeedInput, SensorToggle, TheftSlider } from "../components/controls";
import { NetworkMap, LoadingLegend } from "../components/NetworkMap";
import { Plotly } from "../components/Plot";
import { CompareProfiles, DetectabilitySweep, RunHistory, SimilarCases } from "../components/tools";
import { Card, ErrorState, Skeleton, num, useToasts } from "../components/ui";
import { logUser } from "../lib/activity";
import { LABEL_NAMES } from "../lib/labels";
import { fromApiRequest, toApiRequest, useScenarioStore, type HistoryEntry } from "../store/useScenarioStore";

const extractionLikely = (d: AnalyzeResponse["detect"]) =>
  d.likely_cause.extraction >= Math.max(d.likely_cause.sensor_error, d.likely_cause.other);

const TOOLS = [
  { id: "sweep", label: "Detectability sweep" },
  { id: "compare", label: "Compare profiles" },
  { id: "similar", label: "Similar past cases" },
  { id: "history", label: "Run history" },
] as const;

function download(name: string, text: string, type = "application/json") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

export default function ScenarioLab() {
  const net = useNetwork();
  const analyze = useAnalyze();
  const push = useToasts((s) => s.push);
  const { request, setRequest, setTheftKm, nudgeKm, resetRequest, revealTruth, setRevealTruth, pushHistory, focusedTap, focusTap,
          liveMode, setLiveMode } = useScenarioStore();
  const presets = useSampleRequests();
  const presetList = Array.isArray(presets.data) ? presets.data : [];
  const [tool, setTool] = useState<(typeof TOOLS)[number]["id"]>("sweep");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [caseLabel, setCaseLabel] = useState("theft");
  const cases = useScenarios(caseLabel, 30);
  const locRef = useRef<HTMLDivElement>(null);

  const done = useCallback((r: AnalyzeResponse) => {
    setResult(r);
    pushHistory(useScenarioStore.getState().request, r);
    focusTap(null);
  }, [pushHistory, focusTap]);
  const fail = useCallback((e: Error) => { if (e.name !== "AbortError") push(`Analyze failed: ${e.message}`); }, [push]);

  const run = useCallback(() => {
    const r = request;
    logUser(`Ran analysis: ${r.theftOn ? `${r.theftMw} MW extraction at km ${r.theftKm}` : r.fault !== "none" ? `${r.fault} fault` : "no injection"}`
      + ` (load ×${r.loadScale.toFixed(2)}, ${r.sensorProfile}, noise ×${r.noiseMult}${liveMode ? ", live mode" : ""})`, { request: r });
    analyze.mutate(toApiRequest(request), { onSuccess: done, onError: fail });
  }, [analyze, request, done, fail, liveMode]);

  // Live mode: re-run 300 ms after the last control change (the API client aborts the in-flight call)
  const firstLive = useRef(true);
  useEffect(() => {
    if (!liveMode) { firstLive.current = true; return; }
    if (firstLive.current) { firstLive.current = false; run(); return; }
    const id = setTimeout(run, 300);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request, liveMode]);

  const restore = (h: HistoryEntry) => {
    setRequest(h.request); setResult(h.result); setRevealTruth(false);
    logUser(`Restored run from ${new Date(h.at).toLocaleTimeString()}`, { request_id: h.result.request_id });
  };
  const toggleTruth = (v: boolean) => {
    setRevealTruth(v);
    if (v && result?.simulate) {
      const s = result.simulate;
      logUser(`Revealed truth: ${s.true_label}${s.true_km != null ? ` at km ${s.true_km}, error ${Math.abs(result.localize.est_km - s.true_km).toFixed(1)} km` : ""}`);
    }
  };

  const runCase = (scenario_id: string) => {
    logUser(`Replaying held-out dataset case ${scenario_id.slice(0, 8)}`);
    analyze.mutate({ scenario_id, sensor_profile: "major_only", load_scale: 1, noise_mult: 1, seed: 0 }, { onSuccess: done, onError: fail });
  };

  // keyboard shortcuts: R = run, T = reveal truth (ignored while typing in a field)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) && (e.target as HTMLInputElement).type !== "range") return;
      if (e.key === "r" && !e.ctrlKey && !e.metaKey) { e.preventDefault(); run(); }
      if (e.key === "t" && result) toggleTruth(!revealTruth);
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        if ((e.target as HTMLInputElement).type === "range") return;    // sliders keep their own arrows
        e.preventDefault();
        nudgeKm(e.key === "ArrowRight" ? 10 : -10);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run, result, revealTruth, setRevealTruth, nudgeKm]);

  const sim = result?.simulate ?? null;
  const loc = result?.localize;
  const truthKm = revealTruth ? sim?.true_km ?? null : null;
  const focusKm = focusedTap ? Number(focusedTap.replace("TAP_", "")) : null;

  const exportPng = () => {
    const gd = locRef.current?.querySelector(".js-plotly-plot") as HTMLElement | null;
    if (gd) Plotly.downloadImage(gd, { format: "png", filename: `localization-${result?.request_id}`, width: 1200, height: 500 });
  };

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[19rem_1fr]">
      {/* ---------------- controls ---------------- */}
      <aside className="space-y-4" aria-label="Scenario controls">
        <Card title="Scenario">
          <div className="space-y-4">
            <div>
              <label htmlFor="preset" className="field-label">Preset</label>
              <select id="preset" className="mt-1 w-full" defaultValue="" disabled={!presets.data}
                      onChange={(e) => {
                        const p = presetList.find((x) => x.kind === e.target.value);
                        if (p) { setRequest(fromApiRequest(p.request, request)); logUser(`Loaded preset "${p.label}"`); }
                        e.target.value = "";
                      }}>
                <option value="">{presets.data ? "Load a preset…" : "Loading presets…"}</option>
                {presetList.map((p) => <option key={p.kind} value={p.kind}>{p.label}</option>)}
              </select>
            </div>
            <TheftSlider />
            <hr className="border-line" />
            <LoadNoiseSliders />
            <SensorToggle />
            <FaultInjector />
            <SeedInput />
            <div className="flex gap-2 pt-1">
              <button className="btn btn-primary flex-1" onClick={run} disabled={analyze.isPending} data-testid="run-button"
                      aria-keyshortcuts="r">
                {analyze.isPending ? "Running…" : "▶ Run"} <kbd className="mono text-[10px] opacity-70">R</kbd>
              </button>
              <button className="btn" onClick={resetRequest}>Reset</button>
            </div>
            <label className="flex items-center justify-between gap-2 text-sm">
              <span>Live mode <span className="text-[11px] text-ink-3">(re-run on every change)</span></span>
              <input type="checkbox" className="h-4 w-4 accent-[var(--accent)]" checked={liveMode} data-testid="live-mode"
                     onChange={(e) => { setLiveMode(e.target.checked); logUser(`Live mode ${e.target.checked ? "on" : "off"}`); }} aria-label="Live mode" />
            </label>
            <p className="text-[11px] text-ink-3">Click a tap on the map to place the injection · keys <kbd className="mono">R</kbd> run, <kbd className="mono">T</kbd> truth, <kbd className="mono">←</kbd>/<kbd className="mono">→</kbd> km ±10.</p>
          </div>
        </Card>
        <Card title="Replay a held-out dataset case">
          <div className="space-y-2">
            <select aria-label="Case class" className="w-full" value={caseLabel} onChange={(e) => setCaseLabel(e.target.value)}>
              {Object.entries(LABEL_NAMES).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            {cases.isError ? <ErrorState error={cases.error} /> : (
              <select aria-label="Dataset case" className="w-full" defaultValue="" disabled={!cases.data}
                      onChange={(e) => e.target.value && runCase(e.target.value)}>
                <option value="">{cases.data ? `Pick one of ${cases.data.total} test cases…` : "Loading…"}</option>
                {cases.data?.items.map((c) => (
                  <option key={c.scenario_id} value={c.scenario_id}>
                    {c.true_km != null ? `km ${c.true_km} · ${c.magnitude_mw.toFixed(1)} MW` : c.scenario_id.slice(0, 8)} · load ×{c.load_scale.toFixed(2)}
                  </option>
                ))}
              </select>
            )}
          </div>
        </Card>
      </aside>

      {/* ---------------- results ---------------- */}
      <div className="min-w-0 space-y-4">
        <Card title="Corridor" actions={net.data && <LoadingLegend network={net.data} />}>
          {net.isPending ? <Skeleton h={230} /> : net.isError ? <ErrorState error={net.error} hint="Start the backend: make api" /> : (
            <NetworkMap network={net.data} onPickKm={(km) => { setTheftKm(km); logUser(`Placed injection on map at km ${km}`); }} readings={sim?.measurements}
              overlay={{
                selectedKm: focusKm ?? (request.theftOn ? request.theftKm : null),
                // pin a location only for a suspected extraction — not for sensor faults / network events
                estimate: loc && result?.detect.status === "suspicious" && extractionLikely(result.detect)
                  ? { km: loc.est_km, lo: loc.interval_km[0], hi: loc.interval_km[1] } : null,
                truthKm,
              }} />
          )}
        </Card>

        {analyze.isError && !result && <ErrorState error={analyze.error} />}
        {analyze.isPending && !result && (
          <div className="grid gap-4 md:grid-cols-2"><Skeleton h={150} /><Skeleton h={150} /><Skeleton h={280} /><Skeleton h={280} /></div>
        )}
        {!result && !analyze.isPending && !analyze.isError && (
          <div className="card grid place-items-center p-10 text-center text-sm text-ink-2">
            Set a scenario and press <span className="mx-1 font-semibold text-ink-1">Run</span> (or key <kbd className="mono">R</kbd>).
          </div>
        )}

        {result && (
          <div className={`space-y-4 ${analyze.isPending ? "opacity-60 transition-opacity" : ""}`} aria-busy={analyze.isPending}>
            {sim?.warnings?.length ? (
              <div role="status" className="rounded-md border border-warning/50 bg-warning/10 p-2 text-xs text-ink-1">
                {sim.warnings.map((w) => <div key={w}>◆ {w}</div>)}
              </div>
            ) : null}
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <DetectionCard detect={result.detect} latencyMs={result.latency_ms} localize={result.localize} />
              <CauseBar detect={result.detect} />
            </div>

            <Card title="Recommended inspection actions" subtitle="Rule-based from section, interval and cause confidence">
              <ol className="list-decimal space-y-1 pl-5 text-sm text-ink-1">
                {result.recommendations.map((r) => <li key={r}>{r}</li>)}
              </ol>
            </Card>

            <AdvisorPanel key={result.request_id} analysis={{ detect: result.detect, localize: result.localize }} />

            {loc && (
              <>
                <LocalizationChart ref={locRef} localize={loc} truthKm={truthKm} onPickTap={focusTap}
                  title={result.detect.status === "suspicious" ? "Localization" : "Localization (shown for reference — status is normal)"} />
                <div className="card flex flex-wrap items-center gap-3 p-3">
                  <button className="btn" onClick={() => toggleTruth(!revealTruth)} aria-pressed={revealTruth} aria-keyshortcuts="t" data-testid="reveal-truth">
                    {revealTruth ? "Hide truth" : "Reveal truth"} <kbd className="mono text-[10px] opacity-70">T</kbd>
                  </button>
                  {revealTruth && sim && (
                    <div className="text-sm" data-testid="truth-panel">
                      Truth: <strong>{LABEL_NAMES[sim.true_label] ?? sim.true_label}</strong>
                      {sim.true_km != null && (
                        <>
                          {" "}at <span className="mono">{sim.true_km} km</span>, <span className="mono">{sim.true_magnitude_mw?.toFixed(1)} MW</span>
                          {" · "}error <span className="mono font-semibold">{num(Math.abs(loc.est_km - sim.true_km), 1)} km</span>
                          {" · "}{sim.true_km >= loc.interval_km[0] && sim.true_km <= loc.interval_km[1] ? "inside" : "outside"} the 80% interval
                        </>
                      )}
                    </div>
                  )}
                  {revealTruth && !sim && <span className="text-sm text-ink-3">No ground truth for raw measurements.</span>}
                  <div className="ml-auto flex gap-2">
                    <button className="btn" onClick={() => download(`analysis-${result.request_id}.json`, JSON.stringify(result, null, 2))}>⬇ JSON</button>
                    <button className="btn" onClick={exportPng}>⬇ PNG</button>
                  </div>
                </div>
              </>
            )}

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <ResidualChart detect={result.detect} />
              <ShapBar detect={result.detect} />
            </div>

            {sim && Object.keys(sim.expected).length > 0 && <MeasuredVsExpected measurements={sim.measurements} expected={sim.expected} />}
            <section className="card p-4" aria-label="Investigation tools">
              <div role="tablist" aria-label="Investigation tools" className="mb-3 flex flex-wrap gap-1">
                {TOOLS.map((t) => (
                  <button key={t.id} role="tab" aria-selected={tool === t.id} onClick={() => { setTool(t.id); logUser(`Opened tool: ${t.label}`); }}
                          className={`rounded-md px-3 py-1.5 text-sm ${tool === t.id ? "bg-surface-3 font-semibold text-ink-1" : "text-ink-2 hover:bg-surface-2"}`}>
                    {t.label}
                  </button>
                ))}
              </div>
              <div role="tabpanel">
                {tool === "sweep" && <DetectabilitySweep />}
                {tool === "compare" && <CompareProfiles />}
                {tool === "similar" && <SimilarCases key={result.request_id} measurements={sim?.measurements ?? null} />}
                {tool === "history" && <RunHistory onRestore={restore} />}
              </div>
            </section>

            {sim && (
              <details className="card p-4">
                <summary className="card-title cursor-pointer">All metered channels ({Object.keys(sim.measurements).length})</summary>
                <div className="mt-3"><MeasurementTable measurements={sim.measurements} expected={sim.expected} /></div>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
