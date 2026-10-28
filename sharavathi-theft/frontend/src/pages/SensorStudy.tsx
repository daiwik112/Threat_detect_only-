import type { Data } from "plotly.js";
import { useSensorStudy } from "../api/hooks";
import { ApiError } from "../api/client";
import { Plot, useTheme } from "../components/Plot";
import { Card, ChartCard, DataTable, ErrorState, Skeleton, num, pct } from "../components/ui";

const PROFILES = ["major_only", "major_plus_mid", "dense"];
const PROFILE_NAMES: Record<string, string> = { major_only: "Major only", major_plus_mid: "Major + mid", dense: "Dense" };

export default function SensorStudy() {
  const t = useTheme();
  const q = useSensorStudy();
  if (q.isPending) return <Skeleton h={400} />;
  if (q.isError) {
    const notReady = (q.error as ApiError).status === 404;
    return (
      <div className="space-y-3">
        <h1 className="text-xl font-semibold">Sensor placement study</h1>
        {notReady
          ? <Card title="Not ready yet"><p className="text-sm text-ink-2">sensor_study.json has not been produced. Generate the 12 study datasets and run the study:
              <code className="mono ml-1">make study-data && make sensor-study</code>, then reload.</p></Card>
          : <ErrorState error={q.error} />}
      </div>
    );
  }
  const s = q.data;
  const noises = Array.from(new Set(s.configs.map((c) => c.noise_factor))).sort((a, b) => a - b);
  // one series per profile, fixed colour per profile; x = noise level (grouped bars)
  const grouped = (key: "detection_recall" | "mae_km" | "top3_tap_acc"): Data[] =>
    PROFILES.filter((p) => s.configs.some((c) => c.profile === p)).map((p, i) => ({
      type: "bar", name: PROFILE_NAMES[p], x: noises.map((n) => `×${n}`),
      y: noises.map((n) => (s.configs.find((c) => c.profile === p && c.noise_factor === n)?.[key] as number | undefined) ?? null),
      marker: { color: t.series[i], line: { color: t.surface, width: 2 } },
      hovertemplate: `${PROFILE_NAMES[p]} · noise %{x}<br>%{y:.3f}<extra></extra>`,
    }));
  const ind = s.indistinguishability;
  const pm = (ind as { pair_matrix?: { km: number[]; distance: number[][] } }).pair_matrix;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Sensor placement & observability study</h1>
        <p className="text-sm text-ink-2">Models retrained from scratch for each sensor profile × noise multiplier; detection on the test split, localization on unseen locations.</p>
      </div>
      {s.missing_datasets && s.missing_datasets.length > 0 && (
        <div role="status" className="rounded-md border border-warning/50 bg-warning/10 p-2 text-xs">◆ {s.missing_datasets.length} configuration(s) not yet generated — results are partial.</div>
      )}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <ChartCard title="Detection recall (theft vs normal)">
          <Plot ariaLabel="Detection recall by profile and noise" height={260} data={grouped("detection_recall")} filename="study-recall"
                layout={{ barmode: "group", yaxis: { tickformat: ".0%", range: [0, 1.05] }, xaxis: { title: { text: "noise multiplier" } } }} />
        </ChartCard>
        <ChartCard title="Localization MAE (km, lower is better)">
          <Plot ariaLabel="Localization MAE by profile and noise" height={260} data={grouped("mae_km")} filename="study-mae"
                layout={{ barmode: "group", yaxis: { rangemode: "tozero", title: { text: "km" } }, xaxis: { title: { text: "noise multiplier" } } }} />
        </ChartCard>
        <ChartCard title="Top-3 tap within 10 km">
          <Plot ariaLabel="Top-3 tap accuracy by profile and noise" height={260} data={grouped("top3_tap_acc")} filename="study-top3"
                layout={{ barmode: "group", yaxis: { tickformat: ".0%", range: [0, 1.05] }, xaxis: { title: { text: "noise multiplier" } } }} />
        </ChartCard>
      </div>
      <Card title="All configurations">
        <DataTable columns={["Profile", "Noise", "Recall", "FPR", "Recall @1% FPR", "MAE km", "Median km", "≤10 km", "Top-3 ≤10 km"]}
          rows={s.configs.map((c) => [PROFILE_NAMES[c.profile] ?? c.profile, `×${c.noise_factor}`, pct(c.detection_recall), pct(c.fpr),
                                      pct(c.recall_at_fpr_1pct), num(c.mae_km), num(c.median_km), pct(c.within_10km, 0), pct(c.top3_tap_acc, 0)])} />
      </Card>
      {pm && (
        <ChartCard title="Location separability under major-only metering"
          subtitle={<>Standardised distance between mean feature vectors of 15–40 MW thefts at each tap pair. Darker = easier to tell apart; cells below {ind.threshold ?? 1} are effectively indistinguishable.
            {ind.unresolvable_ranges_km?.length ? <> Unresolvable ranges: {ind.unresolvable_ranges_km.map(([a, b]) => `${a}–${b} km`).join(", ")}.</> : null}</>}
          table={{ columns: ["km A", "km B", "distance", "separable"], rows: (ind.adjacent_pairs ?? []).map((p) => [p.km_a, p.km_b, p.distance.toFixed(3), p.separable ? "yes" : "NO"]) }}>
          <Plot ariaLabel="Tap pair separability heatmap" height={420} filename="separability" data={[{
            type: "heatmap", x: pm.km, y: pm.km, z: pm.distance, xgap: 1, ygap: 1,
            colorscale: t.seq.map((c, i) => [i / (t.seq.length - 1), c]) as [number, string][], zmin: 0,
            hovertemplate: "km %{x} vs km %{y}<br>distance %{z:.2f}<extra></extra>",
            colorbar: { title: { text: "distance", font: { color: t.text2 } }, tickfont: { color: t.text2 }, outlinewidth: 0, thickness: 10 },
          }]} layout={{ xaxis: { title: { text: "tap km" }, showgrid: false, range: [0, 365], constrain: "domain" }, yaxis: { title: { text: "tap km" }, showgrid: false, range: [0, 365], constrain: "domain" } }} />
        </ChartCard>
      )}
    </div>
  );
}
