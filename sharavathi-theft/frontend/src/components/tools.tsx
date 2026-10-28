/**
 * 06 investigation tools for Scenario Lab: detectability sweep, sensor-profile comparison,
 * run history (restore + diff) and similar past cases (vector DB). Each chart keeps one axis.
 */
import type { Data } from "plotly.js";
import { useMemo, useState } from "react";
import { useBatchAnalyze, useSimilarCases } from "../api/hooks";
import type { AnalyzeResponse, Measurements, SensorProfile } from "../api/types";
import { logUser } from "../lib/activity";
import { LABEL_NAMES } from "../lib/labels";
import { toApiRequest, useScenarioStore, type HistoryEntry, type ScenarioRequest } from "../store/useScenarioStore";
import { LocalizationChart } from "./charts";
import { Plot, useTheme } from "./Plot";
import { ChartCard, DataTable, ErrorState, Skeleton, StatusBadge, num, pct } from "./ui";

/* ------------------------------------------------------------------ sweep */
export const SWEEP_MW = [1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40];
export const SWEEP_SEEDS = [1, 2, 3];

export interface SweepPoint { mw: number; n: number; detected: number; medianErr: number | null }

export function summariseSweep(results: (AnalyzeResponse | null)[], mws: number[], seeds: number[]): SweepPoint[] {
  return mws.map((mw, i) => {
    const runs = seeds.map((_, j) => results[i * seeds.length + j]).filter(Boolean) as AnalyzeResponse[];
    const det = runs.filter((r) => r.detect.status === "suspicious");
    const errs = det.map((r) => Math.abs(r.localize.est_km - (r.simulate?.true_km ?? NaN))).filter(Number.isFinite).sort((a, b) => a - b);
    return { mw, n: runs.length, detected: runs.length ? det.length / runs.length : 0,
             medianErr: errs.length ? errs[Math.floor((errs.length - 1) / 2)] : null };
  });
}

export function DetectabilitySweep() {
  const t = useTheme();
  const request = useScenarioStore((s) => s.request);
  const batch = useBatchAnalyze();
  const [ranFor, setRanFor] = useState<ScenarioRequest | null>(null);
  const run = () => {
    const reqs = SWEEP_MW.flatMap((mw) => SWEEP_SEEDS.map((seed) =>
      toApiRequest({ ...request, theftOn: true, theftMw: mw, fault: "none", seed })));
    setRanFor(request);
    logUser(`Started detectability sweep at km ${request.theftKm} (${reqs.length} simulations)`);
    batch.mutate(reqs);
  };
  const pts = useMemo(() => batch.data ? summariseSweep(batch.data.results.map((r) => r.result ?? null), SWEEP_MW, SWEEP_SEEDS) : [], [batch.data]);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <button className="btn btn-primary" onClick={run} disabled={batch.isPending} data-testid="run-sweep">
          {batch.isPending ? "Sweeping…" : "▶ Run detectability sweep"}
        </button>
        <span className="text-xs text-ink-3">
          {SWEEP_MW.length} magnitudes × {SWEEP_SEEDS.length} noise seeds = {SWEEP_MW.length * SWEEP_SEEDS.length} simulations at km {request.theftKm}, load ×{request.loadScale.toFixed(2)}, {request.sensorProfile}
        </span>
      </div>
      {batch.isError && <ErrorState error={batch.error} />}
      {batch.isPending && <Skeleton h={260} />}
      {batch.data && ranFor && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2" data-testid="sweep-result">
          <ChartCard title="Detection rate vs magnitude" subtitle={`km ${ranFor.theftKm} · ${batch.data.n_ok}/${batch.data.results.length} runs converged · ${(batch.data.latency_ms / 1000).toFixed(1)} s`}
            table={{ columns: ["MW", "runs", "detected", "median error km"], rows: pts.map((p) => [p.mw, p.n, pct(p.detected, 0), num(p.medianErr)]) }}>
            <Plot ariaLabel="Detection rate versus extraction magnitude" height={240} filename="detectability" data={[{
              type: "scatter", mode: "lines+markers", x: pts.map((p) => p.mw), y: pts.map((p) => p.detected), name: "detection rate",
              line: { color: t.series[0], width: 2 }, marker: { size: 8, color: t.series[0], line: { color: t.surface, width: 2 } },
              hovertemplate: "%{x} MW<br>detected %{y:.0%}<extra></extra>",
            } as Data]} layout={{ xaxis: { title: { text: "extraction magnitude (MW)" }, type: "log" }, yaxis: { tickformat: ".0%", range: [0, 1.05] } }} />
          </ChartCard>
          <ChartCard title="Localization error vs magnitude" subtitle="Median |estimated − true| km over detected runs">
            <Plot ariaLabel="Median localization error versus extraction magnitude" height={240} filename="sweep-error" data={[{
              type: "scatter", mode: "lines+markers", x: pts.map((p) => p.mw), y: pts.map((p) => p.medianErr), name: "median error",
              line: { color: t.series[0], width: 2 }, marker: { size: 8, color: t.series[0], line: { color: t.surface, width: 2 } },
              hovertemplate: "%{x} MW<br>median error %{y:.1f} km<extra></extra>",
            } as Data]} layout={{ xaxis: { title: { text: "extraction magnitude (MW)" }, type: "log" }, yaxis: { title: { text: "km" }, rangemode: "tozero" } }} />
          </ChartCard>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ compare profiles */
const PROFILES: { id: SensorProfile; label: string }[] = [
  { id: "major_only", label: "Major only" }, { id: "major_plus_mid", label: "Major + mid" }, { id: "dense", label: "Dense" },
];

export function CompareProfiles() {
  const request = useScenarioStore((s) => s.request);
  const batch = useBatchAnalyze();
  const run = () => {
    logUser("Compared sensor profiles (major-only / major+mid / dense)");
    batch.mutate(PROFILES.map((p) => toApiRequest({ ...request, sensorProfile: p.id })));
  };
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <button className="btn btn-primary" onClick={run} disabled={batch.isPending} data-testid="run-compare">
          {batch.isPending ? "Comparing…" : "▶ Compare sensor profiles"}
        </button>
        <span className="text-xs text-ink-3">Same scenario and noise seed ({request.seed}) under each metering profile. Models were trained on major-only; extra channels are simulated but ignored by them.</span>
      </div>
      {batch.isError && <ErrorState error={batch.error} />}
      {batch.isPending && <Skeleton h={260} />}
      {batch.data && (
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3" data-testid="compare-result">
          {batch.data.results.map((item, i) => (
            <div key={PROFILES[i].id} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold">{PROFILES[i].label}</span>
                {item.result && <StatusBadge status={item.result.detect.status} />}
              </div>
              {item.error ? <ErrorState error={item.error} /> : item.result && (
                <>
                  <div className="text-xs text-ink-2">score <span className="mono">{item.result.detect.anomaly_score.toFixed(3)}</span>
                    {" · "}{Object.keys(item.result.simulate?.measurements ?? {}).length} metered channels
                    {item.result.simulate?.true_km != null && <> · error <span className="mono">{num(Math.abs(item.result.localize.est_km - item.result.simulate.true_km))} km</span></>}</div>
                  <LocalizationChart localize={item.result.localize} truthKm={item.result.simulate?.true_km} height={200} title={`${PROFILES[i].label} — localization`} />
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ run history */
function row(h: HistoryEntry) {
  const r = h.result, s = r.simulate;
  return {
    time: new Date(h.at).toLocaleTimeString(),
    scenario: h.request.theftOn ? `${h.request.theftMw} MW @ km ${h.request.theftKm}` : h.request.fault !== "none" ? h.request.fault : "normal",
    status: r.detect.status, score: r.detect.anomaly_score.toFixed(3),
    est: `${r.localize.est_km.toFixed(0)} [${r.localize.interval_km.map((v) => v.toFixed(0)).join("–")}]`,
    err: s?.true_km != null ? num(Math.abs(r.localize.est_km - s.true_km)) : "—",
    cause: Object.entries(r.detect.likely_cause).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—",
  };
}

export function RunHistory({ onRestore }: { onRestore: (h: HistoryEntry) => void }) {
  const history = useScenarioStore((s) => s.history);
  const [sel, setSel] = useState<number[]>([]);
  if (!history.length) return <p className="text-sm text-ink-3">No runs yet — each Run is kept here (last 20).</p>;
  const toggle = (i: number) => setSel((s) => s.includes(i) ? s.filter((x) => x !== i) : [...s.slice(-1), i]);
  const [a, b] = sel.map((i) => history[i]);
  return (
    <div className="space-y-3">
      <div className="max-h-72 overflow-auto">
        <table className="w-full text-left text-xs" data-testid="history-table">
          <thead className="sticky top-0 bg-surface-1 text-ink-2">
            <tr>{["Diff", "Time", "Scenario", "Status", "Score", "Est km [80%]", "Error km", "Cause", ""].map((c) => <th key={c} className="border-b border-line py-1.5 pr-3 font-semibold">{c}</th>)}</tr>
          </thead>
          <tbody>
            {history.map((h, i) => {
              const r = row(h);
              return (
                <tr key={h.at + h.result.request_id} className="border-b border-line/60">
                  <td className="py-1 pr-3"><input type="checkbox" aria-label={`Select run ${i + 1} for diff`} checked={sel.includes(i)} onChange={() => toggle(i)} /></td>
                  <td className="mono pr-3">{r.time}</td><td className="pr-3">{r.scenario}</td>
                  <td className="pr-3"><StatusBadge status={r.status} /></td><td className="mono pr-3">{r.score}</td>
                  <td className="mono pr-3">{r.est}</td><td className="mono pr-3">{r.err}</td><td className="pr-3">{r.cause}</td>
                  <td><button className="btn !px-2 !py-0.5 text-xs" onClick={() => onRestore(h)}>Restore</button></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {a && b ? (
        <div data-testid="history-diff">
          <div className="card-title mb-1">Diff</div>
          <DataTable columns={["Field", "Run A", "Run B"]} rows={(Object.keys(row(a)) as (keyof ReturnType<typeof row>)[]).map((k) => {
            const ra = row(a), rb = row(b);
            return [k, ra[k], rb[k] === ra[k] ? rb[k] : `▶ ${rb[k]}`];
          })} />
        </div>
      ) : <p className="text-xs text-ink-3">Tick two runs to diff them.</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ similar cases */
export function SimilarCases({ measurements }: { measurements: Measurements | null }) {
  const sim = useSimilarCases();
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <button className="btn btn-primary" disabled={!measurements || sim.isPending} data-testid="run-similar"
                onClick={() => { if (!measurements) return; logUser("Searched vector DB for similar past cases"); sim.mutate({ measurements, k: 8 }); }}>
          {sim.isPending ? "Searching…" : "▶ Find similar past cases"}
        </button>
        <span className="text-xs text-ink-3">Nearest labelled scenarios in the vector database (41-d physics-feature embedding, cosine similarity). Reference context only — not a classifier.</span>
      </div>
      {!measurements && <p className="text-sm text-ink-3">Run a scenario first.</p>}
      {sim.isError && <ErrorState error={sim.error} />}
      {sim.data && (
        <div data-testid="similar-result">
          <DataTable columns={["Similarity", "Label", "True km", "MW", "Load ×", "Scenario"]}
            rows={sim.data.items.map((c) => [c.similarity.toFixed(3), LABEL_NAMES[c.label] ?? c.label, c.true_km ?? "—", c.magnitude_mw || "—", c.load_scale, c.scenario_id.slice(0, 8)])} />
          <p className="mt-1 text-[11px] text-ink-3">{sim.data.n_indexed.toLocaleString()} train+val scenarios indexed · {sim.data.latency_ms.toFixed(0)} ms</p>
        </div>
      )}
    </div>
  );
}
