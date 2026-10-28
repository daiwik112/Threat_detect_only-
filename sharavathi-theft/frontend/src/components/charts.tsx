/**
 * Result charts for one analysis. Conventions (dataviz skill): one axis per chart, thin
 * marks, status colours only with icon+label, text in ink tokens, every chart has a
 * table view via ChartCard, hover tooltips on every mark.
 */
import type { Annotation, Data, Shape } from "plotly.js";
import { forwardRef } from "react";
import type { DetectResponse, LocalizeResponse, Measurements } from "../api/types";
import { busName, channelLabel, featureLabel, sectionName } from "../lib/labels";
import { Plot, useTheme } from "./Plot";
import { ChartCard, StatusBadge, num, pct } from "./ui";

/* ------------------------------------------------------------------ detection */
/** One-line plain-English summary ("Suspected unauthorized extraction near km 145 (119–170)"). */
export function summaryLine(detect: DetectResponse, localize?: LocalizeResponse | null): string {
  if (detect.status !== "suspicious") return "Normal operation — no anomaly above the alarm threshold.";
  const lc = detect.likely_cause;
  const main = (Object.keys(lc) as (keyof typeof lc)[]).reduce((a, b) => (lc[a] >= lc[b] ? a : b));
  if (main === "extraction" && localize) {
    const [lo, hi] = localize.interval_km;
    return `Suspected unauthorized extraction near km ${localize.est_km.toFixed(0)} (80% interval ${lo.toFixed(0)}–${hi.toFixed(0)} km).`;
  }
  return main === "sensor_error" ? "Anomaly most consistent with a metering / sensor fault — verify meters before any patrol."
    : "Anomaly most consistent with a network event such as a line outage.";
}

export function DetectionCard({ detect, latencyMs, localize }: { detect: DetectResponse; latencyMs?: number; localize?: LocalizeResponse | null }) {
  const t = useTheme();
  const s = detect.anomaly_score;
  return (
    <section className="card flex items-center gap-5 p-4" aria-label="Detection result">
      <div className="flex-1">
        <h2 className="card-title">Detection</h2>
        <div className="mt-2"><StatusBadge status={detect.status} large /></div>
        <p className="mt-2 text-sm font-medium text-ink-1" data-testid="status-summary">{summaryLine(detect, localize)}</p>
        <p className="mt-1 text-xs text-ink-3">
          {detect.status === "suspicious"
            ? "Unexplained anomaly — suspected, not confirmed. Requires field verification."
            : "No anomaly above the alarm threshold (score < 0.5)."}
        </p>
        {latencyMs != null && <p className="mt-1 mono text-[10px] text-ink-3">analysed in {latencyMs.toFixed(0)} ms</p>}
      </div>
      <div className="w-40 text-center" aria-label={`Anomaly score ${s.toFixed(3)}`}>
        <svg viewBox="0 0 120 70" className="w-full" aria-hidden>
          <path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke={t.grid} strokeWidth="10" strokeLinecap="round" />
          <path d="M10 60 A50 50 0 0 1 110 60" fill="none" strokeWidth="10" strokeLinecap="round"
                stroke={detect.status === "suspicious" ? t.status.critical : t.status.good}
                strokeDasharray={`${Math.PI * 50 * s} 400`} />
          <line x1="60" y1="8" x2="60" y2="16" stroke={t.text2} strokeWidth="1.5" />
        </svg>
        <div className="-mt-6 text-2xl font-semibold mono">{s.toFixed(3)}</div>
        <div className="text-[10px] text-ink-3">anomaly score · threshold 0.5</div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ residuals */
export function ResidualChart({ detect }: { detect: DetectResponse }) {
  const t = useTheme();
  const r = detect.residuals;
  const thr = detect.threshold_z ?? null;
  const names = r.map((x) => sectionName(x.section).replace(" → ", " →<br>"));
  const z = r.map((x) => x.z ?? 0);
  const data: Data[] = [{
    type: "bar", x: names, y: z, name: "|r_P| / σ",
    marker: { color: z.map((v) => (thr && v > thr ? t.status.critical : t.series[0])), line: { color: t.surface, width: 2 } },
    customdata: r.map((x) => [x.r_p_mw ?? NaN, x.r_q_mvar ?? NaN]),
    hovertemplate: "%{x}<br>z = %{y:.1f}<br>P residual %{customdata[0]:.2f} MW<br>Q residual %{customdata[1]:.2f} Mvar<extra></extra>",
    text: z.map((v) => v.toFixed(1)), textposition: "outside", textfont: { color: t.text2 },
  }];
  return (
    <ChartCard title="Section power residuals" subtitle="Unexplained active power per metered section, in σ of normal operation"
      table={{ columns: ["Section", "r_P (MW)", "r_Q (Mvar)", "z"], rows: r.map((x) => [sectionName(x.section), num(x.r_p_mw, 2), num(x.r_q_mvar, 2), num(x.z, 2)]) }}>
      <Plot ariaLabel="Residual z-score per section" height={250} data={data} filename="residuals" layout={{
        yaxis: { title: { text: "z (σ)" }, rangemode: "tozero" }, showlegend: false,
        shapes: thr ? [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: thr, y1: thr, line: { color: t.text2, dash: "dash", width: 1.5 } }] : [],
        annotations: thr ? [{ xref: "paper", x: 1, y: thr, text: `alarm threshold z=${thr.toFixed(1)}`, showarrow: false, yanchor: "bottom", xanchor: "right", font: { color: t.text2, size: 11 } }] : [],
      }} />
    </ChartCard>
  );
}

/* ------------------------------------------------------------------ localization */
export const LocalizationChart = forwardRef<HTMLDivElement, {
  localize: LocalizeResponse; truthKm?: number | null; onPickTap?: (tapId: string) => void; height?: number; title?: string;
}>(function LocalizationChart({ localize, truthKm, onPickTap, height = 280, title = "Localization" }, ref) {
  const t = useTheme();
  const taps = [...localize.ranked_taps].sort((a, b) => a.km - b.km);
  const [lo, hi] = localize.interval_km;
  const ymax = Math.max(...taps.map((x) => x.prob), 0.05) * 1.25;
  const data: Data[] = [{
    type: "bar", x: taps.map((x) => x.km), y: taps.map((x) => x.prob), width: 6, name: "Tap probability",
    marker: { color: t.series[0], line: { color: t.surface, width: 1 } },
    customdata: taps.map((x) => x.tap_id),
    hovertemplate: "%{customdata} · km %{x}<br>P = %{y:.1%}<extra></extra>",
  }];
  const shapes: Partial<Shape>[] = [
    { type: "rect", x0: lo, x1: hi, yref: "paper", y0: 0, y1: 1, fillcolor: t.status.critical, opacity: 0.1, line: { width: 0 }, layer: "below" },
    { type: "line", x0: localize.est_km, x1: localize.est_km, yref: "paper", y0: 0, y1: 1, line: { color: t.status.critical, width: 2 } },
  ];
  const ann: Partial<Annotation>[] = [
    { x: localize.est_km, yref: "paper", y: 1, text: `est. ${localize.est_km.toFixed(0)} km`, showarrow: false, yanchor: "bottom", font: { color: t.text, size: 11 } },
    { x: hi, yref: "paper", y: 0.97, text: `80% interval ${lo.toFixed(0)}–${hi.toFixed(0)} km`, showarrow: false, xanchor: hi > 300 ? "right" : "left", xshift: hi > 300 ? -4 : 4, yanchor: "top", font: { color: t.text2, size: 10 } },
  ];
  if (truthKm != null) {
    shapes.push({ type: "line", x0: truthKm, x1: truthKm, yref: "paper", y0: 0, y1: 1, line: { color: t.text, width: 2, dash: "dot" } });
    const right = truthKm >= localize.est_km;
    ann.push({ x: truthKm, yref: "paper", y: 0.82, text: `true ${truthKm} km`, showarrow: false, xanchor: right ? "left" : "right", xshift: right ? 4 : -4, bgcolor: t.surface, font: { color: t.text, size: 11 } });
  }
  return (
    <ChartCard title={title}
      subtitle={<>Method <span className="mono">{localize.method}</span> · bars = top-10 candidate taps · band = 80% interval</>}
      table={{ columns: ["Tap", "km", "Probability"], rows: localize.ranked_taps.map((x) => [x.tap_id, x.km, pct(x.prob)]) }}>
      <div ref={ref}>
        <Plot ariaLabel="Theft location probability along the corridor" height={height} data={data} filename="localization" layout={{
          xaxis: { title: { text: "km from Sharavathi" }, range: [0, 361], dtick: 50 },
          yaxis: { title: { text: "probability" }, tickformat: ".0%", range: [0, ymax] },
          shapes, annotations: ann, showlegend: false,
        }} />
      </div>
      {onPickTap && (
        <div className="mt-2 flex flex-wrap gap-1.5" aria-label="Ranked taps">
          {localize.ranked_taps.slice(0, 5).map((tp, i) => (
            <button key={tp.tap_id} className="btn !px-2 !py-0.5 text-xs" onClick={() => onPickTap(tp.tap_id)}>
              #{i + 1} {tp.tap_id} <span className="mono text-ink-3">{pct(tp.prob, 0)}</span>
            </button>
          ))}
        </div>
      )}
      <div className="mt-3 grid grid-cols-1 gap-1 text-xs sm:grid-cols-3">
        {localize.ranked_sections.map((s) => (
          <div key={s.section} className="flex items-center justify-between rounded bg-surface-2 px-2 py-1">
            <span className="text-ink-2">{sectionName(s.section)}</span><span className="mono">{pct(s.prob, 0)}</span>
          </div>
        ))}
      </div>
    </ChartCard>
  );
});

/* ------------------------------------------------------------------ likely cause */
const CAUSES = [
  { key: "extraction", label: "Extraction" },
  { key: "sensor_error", label: "Sensor error" },
  { key: "other", label: "Other event" },
] as const;

export function CauseBar({ detect }: { detect: DetectResponse }) {
  const t = useTheme();
  const lc = detect.likely_cause;
  const data: Data[] = CAUSES.map((c, i) => ({
    type: "bar", orientation: "h", y: ["cause"], x: [lc[c.key]], name: c.label,
    marker: { color: t.series[i], line: { color: t.surface, width: 2 } },
    text: lc[c.key] >= 0.08 ? [`${c.label} ${(lc[c.key] * 100).toFixed(0)}%`] : [""],
    textposition: "inside", insidetextanchor: "middle", textfont: { color: "#fff", size: 11 },
    hovertemplate: `${c.label}: %{x:.1%}<extra></extra>`,
  }));
  return (
    <ChartCard title="Likely cause" subtitle="If anomalous, how the evidence splits (6-class model, anomaly classes renormalised)"
      table={{ columns: ["Cause", "Share"], rows: CAUSES.map((c) => [c.label, pct(lc[c.key])]) }}>
      <Plot ariaLabel="Likely cause split" height={100} data={data} filename="likely-cause" layout={{
        barmode: "stack", xaxis: { range: [0, 1], visible: false }, yaxis: { visible: false },
        margin: { l: 4, r: 4, t: 4, b: 4 }, legend: { orientation: "h", y: -0.25, x: 0, traceorder: "normal" },
      }} />
    </ChartCard>
  );
}

/* ------------------------------------------------------------------ SHAP */
export function ShapBar({ detect }: { detect: DetectResponse }) {
  const t = useTheme();
  const top = [...detect.top_contributors].reverse();
  if (!top.length) return null;
  const data: Data[] = [{
    type: "bar", orientation: "h", y: top.map((c) => featureLabel(c.feature)), x: top.map((c) => c.shap),
    marker: { color: top.map((c) => (c.shap >= 0 ? t.status.critical : t.series[0])), line: { color: t.surface, width: 2 } },
    customdata: top.map((c) => c.feature),
    hovertemplate: "%{y}<br><span style='font-family:monospace'>%{customdata}</span><br>SHAP %{x:+.2f} (log-odds)<extra></extra>",
  }];
  return (
    <ChartCard title="Top contributors (SHAP)"
      subtitle={<><span className="text-[var(--status-critical-text)]">▶ pushes toward anomaly</span> · <span className="text-accent">◀ toward normal</span></>}
      table={{ columns: ["Feature", "SHAP (log-odds)"], rows: detect.top_contributors.map((c) => [featureLabel(c.feature), c.shap.toFixed(3)]) }}>
      <Plot ariaLabel="SHAP feature contributions" height={220} data={data} filename="shap" layout={{
        xaxis: { title: { text: "SHAP (log-odds)" }, zeroline: true }, yaxis: { automargin: true },
        margin: { l: 8, r: 16, t: 8, b: 40 }, showlegend: false,
      }} />
    </ChartCard>
  );
}

/* ------------------------------------------------------------------ measured vs expected */
const QUANT = [
  { k: "vm", title: "Voltage", unit: "pu" },
  { k: "p", title: "Active power", unit: "MW" },
  { k: "q", title: "Reactive power", unit: "Mvar" },
] as const;

export function MeasuredVsExpected({ measurements, expected }: { measurements: Measurements; expected: Measurements }) {
  const t = useTheme();
  const buses = Array.from(new Set(Object.keys(measurements).map((k) => k.match(/^m_(?!LINE)(.+)_(vm|va|p|q)$/)?.[1]).filter(Boolean))) as string[];
  const rows = Object.keys(measurements).map((k) => [channelLabel(k), num(measurements[k], 3), num(expected[k], 3),
    measurements[k] != null && expected[k] != null ? (measurements[k]! - expected[k]!).toFixed(3) : "—"]);
  return (
    <ChartCard title="Measured vs expected" subtitle="Metered bus channels · expected = theft-free, noiseless twin at the same load"
      table={{ columns: ["Channel", "Measured", "Expected", "Δ"], rows }}>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
        {QUANT.map((q) => {
          const m = buses.map((b) => measurements[`m_${b}_${q.k}`] ?? null);
          const e = buses.map((b) => expected[`m_${b}_${q.k}`] ?? null);
          const data: Data[] = [
            q.k === "vm"
              // voltage: dot plot on its own range — bars from 0 would hide a 0.01 pu deviation
              ? { type: "scatter", mode: "markers", x: buses.map(busName), y: m, name: "Measured",
                  marker: { color: t.series[0], size: 11, line: { color: t.surface, width: 2 } },
                  hovertemplate: `%{x}<br>measured %{y:.4f} ${q.unit}<extra></extra>` }
              : { type: "bar", x: buses.map(busName), y: m, name: "Measured", marker: { color: t.series[0] },
                  hovertemplate: `%{x}<br>measured %{y:.3f} ${q.unit}<extra></extra>` },
            { type: "scatter", mode: "markers", x: buses.map(busName), y: e, name: "Expected",
              marker: { symbol: "line-ew-open", size: 26, color: t.text, line: { width: 2.5, color: t.text } },
              hovertemplate: `%{x}<br>expected %{y:.3f} ${q.unit}<extra></extra>` },
          ];
          return (
            <div key={q.k}>
              <div className="mb-1 text-xs text-ink-2">{q.title} ({q.unit})</div>
              <Plot ariaLabel={`${q.title} measured vs expected`} height={200} data={data} filename={`measured-${q.k}`}
                layout={{ margin: { l: 44, r: 8, t: 24, b: 50 }, legend: { orientation: "h", y: 1.18, x: 0 },
                          yaxis: q.k === "vm" ? { tickformat: ".3f", autorange: true } : {} }} />
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}
