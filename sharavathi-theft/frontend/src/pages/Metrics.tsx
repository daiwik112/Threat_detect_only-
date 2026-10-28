import type { Data } from "plotly.js";
import { useEvalArrays, useMetrics } from "../api/hooks";
import type { DetMetrics, LocMetrics } from "../api/types";
import { Plot, useTheme } from "../components/Plot";
import { Card, ChartCard, DataTable, ErrorState, Skeleton, Stat, num, pct } from "../components/ui";
import { LABEL_NAMES } from "../lib/labels";

const LOC_NAMES: Record<string, string> = { physics_baseline: "Physics baseline", xgb_reg: "XGBoost regression", section_rank: "XGBoost tap ranking", gnn: "GNN (GraphSAGE)" };
const LOC_SERIES: Record<string, number> = { xgb_reg: 0, physics_baseline: 1, gnn: 2 };

export default function Metrics() {
  const t = useTheme();
  const q = useMetrics();
  const arr = useEvalArrays();
  if (q.isPending) return <Skeleton h={400} />;
  if (q.isError) return <ErrorState error={q.error} hint="Train models first: make train" />;
  const m = q.data;
  const det = m.detection_detail.test;
  const cm = m.classification.confusion;
  const labels = m.classification.labels.map((l) => LABEL_NAMES[l] ?? l);
  const rowPct = cm.map((r) => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / s); });

  type Arr = { true_km: number[]; est_km: number[]; magnitude_mw: number[] };
  const methods = ["xgb_reg", "gnn"].filter((k) => arr.data?.[`${k}_holdout`]);
  const errVsMag: Data[] = methods.map((k) => {
    const a = arr.data![`${k}_holdout`] as Arr;
    return { type: "scatter", mode: "markers", name: LOC_NAMES[k], x: a.magnitude_mw, y: a.est_km.map((e, i) => Math.abs(e - a.true_km[i])),
      marker: { size: 6, color: t.series[LOC_SERIES[k]], opacity: 0.55, line: { color: t.surface, width: 1 } },
      hovertemplate: "%{x:.1f} MW<br>error %{y:.1f} km<extra>" + LOC_NAMES[k] + "</extra>" };
  });
  const estVsTrue: Data[] = methods.map((k) => {
    const a = arr.data![`${k}_holdout`] as Arr;
    return { type: "box", name: LOC_NAMES[k], x: a.true_km, y: a.est_km.map((e, i) => e - a.true_km[i]),
      marker: { color: t.series[LOC_SERIES[k]] }, line: { width: 1.5 }, boxpoints: false,
      hovertemplate: "true km %{x}<br>signed error %{y:.1f} km<extra></extra>" } as Data;
  });

  const detRows = (name: string, d?: DetMetrics) => d ? [name, pct(d.precision), pct(d.recall), pct(d.f1), pct(d.fpr), num(d.roc_auc, 3), pct(d.recall_at_fpr_1pct)] : [name, "—", "—", "—", "—", "—", "—"];
  const locRow = (k: string, l?: LocMetrics) => l ? [LOC_NAMES[k] ?? k, num(l.mae_km), num(l.median_km), pct(l.within_10km, 0), pct(l.within_25km, 0), pct(l.within_50km, 0), pct(l.top3_within_10km, 0)] : null;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Model metrics</h1>
        <p className="text-sm text-ink-2">Held-out data only: group-held-out <span className="mono">test</span> and leave-location-out <span className="mono">test_loc_holdout</span> (taps at 40/50/130/140/250/260 km never seen in training).</p>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Precision" value={pct(m.detection.precision)} sub={`baseline ${pct(m.baseline.precision)}`} />
        <Stat label="Recall" value={pct(m.detection.recall)} sub={`baseline ${pct(m.baseline.recall)}`} tone="good" />
        <Stat label="F1" value={pct(m.detection.f1)} sub={`baseline ${pct(m.baseline.f1)}`} tone="good" />
        <Stat label="FPR" value={pct(m.detection.fpr)} sub={`baseline ${pct(m.baseline.fpr)}`} tone={(m.detection.fpr ?? 0) > (m.baseline.fpr ?? 0) ? "bad" : "good"} />
        <Stat label="MAE km (unseen locations)" value={num(m.localization.mae_km)} sub={`median ${num(m.localization.median_km)} · baseline ${num(m.baseline.mae_km)}`} tone="good" />
      </div>
      <Card title="Honest claim"><p className="text-sm">{m.honest_claim}</p></Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="AI vs baseline — detection (test)">
          <DataTable columns={["Model", "Precision", "Recall", "F1", "FPR", "ROC-AUC", "Recall @1% FPR"]}
            rows={[detRows("XGBoost", det.xgb), detRows("Residual baseline", det.baseline), ...(det.gnn ? [detRows("GNN", det.gnn)] : []),
                   detRows("XGBoost — theft vs normal", det.xgb_theft_vs_normal), detRows("Baseline — theft vs normal", det.baseline_theft_vs_normal)]} />
        </Card>
        <Card title="AI vs baseline — localization (unseen locations)">
          <DataTable columns={["Method", "MAE km", "Median km", "≤10 km", "≤25 km", "≤50 km", "Top-3 ≤10 km"]}
            rows={Object.entries(m.localization_detail.holdout).map(([k, l]) => locRow(k, l)).filter(Boolean) as (string | null)[][]} />
          <p className="mt-2 text-xs text-ink-3">Shipped localizer: <span className="mono">{m.selected_localizer}</span>
            {m.gnn_decision && <> · GNN {m.gnn_decision.keep ? "kept (improves on XGBoost)" : "not kept — did not beat XGBoost (negative result documented)"}</>}</p>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ChartCard title="Confusion matrix (test, row-normalised)" subtitle="Rows = true class, columns = predicted"
          table={{ columns: ["True \\ Pred", ...labels], rows: cm.map((r, i) => [labels[i], ...r]) }}>
          <Plot ariaLabel="Confusion matrix heatmap" height={340} filename="confusion" data={[{
            type: "heatmap", z: rowPct, x: labels, y: labels, colorscale: t.seq.map((c, i) => [i / (t.seq.length - 1), c]) as [number, string][],
            zmin: 0, zmax: 1, xgap: 2, ygap: 2, customdata: cm as unknown as number[][],
            text: rowPct.map((r) => r.map((v) => (v >= 0.005 ? `${(v * 100).toFixed(0)}%` : ""))) as unknown as string[],
            texttemplate: "%{text}", textfont: { color: t.text, size: 11 },
            hovertemplate: "true %{y}<br>pred %{x}<br>%{z:.1%} (%{customdata} cases)<extra></extra>",
            colorbar: { tickformat: ".0%", tickfont: { color: t.text2 }, outlinewidth: 0, thickness: 10 },
          }]} layout={{ yaxis: { autorange: "reversed", showgrid: false }, xaxis: { showgrid: false, tickangle: -30 }, margin: { l: 130, r: 8, t: 8, b: 90 } }} />
        </ChartCard>
        <ChartCard title="Per-class recall (test)"
          table={{ columns: ["Class", "Recall"], rows: Object.entries(m.classification.per_class_recall).map(([k, v]) => [LABEL_NAMES[k] ?? k, pct(v)]) }}>
          <Plot ariaLabel="Per-class recall" height={340} filename="per-class-recall" data={[{
            type: "bar", orientation: "h", y: Object.keys(m.classification.per_class_recall).map((k) => LABEL_NAMES[k] ?? k),
            x: Object.values(m.classification.per_class_recall), marker: { color: t.series[0] },
            text: Object.values(m.classification.per_class_recall).map((v) => pct(v, 0)), textposition: "outside", textfont: { color: t.text2 },
            hovertemplate: "%{y}: %{x:.1%}<extra></extra>",
          }]} layout={{ xaxis: { range: [0, 1.12], tickformat: ".0%" }, yaxis: { autorange: "reversed" }, margin: { l: 140, r: 16, t: 8, b: 36 } }} />
        </ChartCard>
      </div>

      {arr.isError ? <ErrorState error={arr.error} /> : methods.length > 0 && (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <ChartCard title="Localization error vs magnitude (unseen locations)"
            table={{ columns: ["Bucket (MW)", "n", "MAE km", "Median km", "≤10 km"], rows: Object.entries(m.localization_detail.holdout[m.selected_localizer ?? "xgb_reg"]?.by_magnitude ?? {}).map(([k, b]) => [k.replace("_", "–"), b.n, num(b.mae_km), num(b.median_km), "—"]) }}>
            <Plot ariaLabel="Localization error versus theft magnitude" height={300} data={errVsMag} filename="error-vs-magnitude" layout={{
              xaxis: { title: { text: "extraction magnitude (MW)" } }, yaxis: { title: { text: "|error| (km)" }, rangemode: "tozero" },
            }} />
          </ChartCard>
          <ChartCard title="Signed error by true location (unseen locations)" subtitle="Each box = one held-out tap">
            <Plot ariaLabel="Signed localization error by true km" height={300} data={estVsTrue} filename="error-vs-true-km" layout={{
              boxmode: "group", xaxis: { title: { text: "true km" } }, yaxis: { title: { text: "estimated − true (km)" }, zeroline: true },
            }} />
          </ChartCard>
        </div>
      )}

      {arr.data?.roc && (
        <ChartCard title="ROC — test split" subtitle="Log-scale FPR highlights the low-false-alarm region that matters operationally">
          <Plot ariaLabel="ROC curves" height={300} filename="roc" data={Object.entries(arr.data.roc).map(([k, r]) => ({
            type: "scatter", mode: "lines", name: k === "xgb" ? "XGBoost" : k === "baseline" ? "Residual baseline" : "GNN",
            x: r.fpr.map((v) => Math.max(v, 1e-3)), y: r.tpr, line: { width: 2, color: t.series[k === "xgb" ? 0 : k === "baseline" ? 1 : 2] },
            hovertemplate: "FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
          }))} layout={{ xaxis: { type: "log", title: { text: "false-positive rate" }, range: [-3, 0] }, yaxis: { title: { text: "true-positive rate" }, range: [0, 1.02] } }} />
        </ChartCard>
      )}

      <Card title="Robustness (test, theft vs normal; features recomputed after perturbation)">
        <DataTable columns={["Condition", "XGB P(theft) AUC", "XGB P(theft) recall @1% FPR", "Baseline AUC", "Baseline recall @1% FPR", "Loc. MAE km"]}
          rows={Object.entries(m.robustness).map(([k, v]) => {
            const r = v as Record<string, Record<string, number> | number>;
            const g = (a: string, b: string) => (r[a] as Record<string, number> | undefined)?.[b];
            return [k.replace(/_/g, " "), num(g("xgb_p_theft", "roc_auc"), 3), pct(g("xgb_p_theft", "recall_at_fpr_1pct")),
                    num(g("baseline", "roc_auc"), 3), pct(g("baseline", "recall_at_fpr_1pct")), num(r.xgb_reg_mae_km as number)];
          })} />
      </Card>
    </div>
  );
}
