import { useState } from "react";
import { Link } from "react-router-dom";
import { useHealth, useMetrics, useNetwork } from "../api/hooks";
import { AssumptionsDrawer } from "../components/AssumptionsDrawer";
import { LoadingLegend, NetworkMap } from "../components/NetworkMap";
import { Card, DataTable, ErrorState, Skeleton, Stat, num, pct } from "../components/ui";
import { busName, sectionName } from "../lib/labels";

export default function Overview() {
  const net = useNetwork();
  const health = useHealth();
  const metrics = useMetrics();
  const [drawer, setDrawer] = useState(false);
  const m = metrics.data;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Sharavathi → Peenya 220 kV corridor</h1>
          <p className="text-sm text-ink-2">361 km simulated transmission corridor · unmetered tap buses every 10 km are candidate extraction points</p>
        </div>
        <div className="flex gap-2">
          <button className="btn" onClick={() => setDrawer(true)} disabled={!net.data}>◆ Assumptions</button>
          <Link to="/lab" className="btn btn-primary">Open Scenario Lab →</Link>
        </div>
      </div>

      <Card title="Corridor diagram" actions={net.data && <LoadingLegend network={net.data} />}>
        {net.isPending ? <Skeleton h={250} /> : net.isError ? <ErrorState error={net.error} hint="Start the backend: make api" /> :
          <NetworkMap network={net.data} height={250} />}
      </Card>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Detection recall (test)" value={pct(m?.detection.recall)} sub={m && `baseline ${pct(m.baseline.recall)}`} tone="good" />
        <Stat label="False-positive rate" value={pct(m?.detection.fpr)} sub={m && `baseline ${pct(m.baseline.fpr)}`} />
        <Stat label="Recall @ 1% FPR" value={pct(m?.detection.recall_at_fpr_1pct)} sub={m && `baseline ${pct(m.baseline.recall_at_fpr_1pct)}`} tone="good" />
        <Stat label="Localization MAE (unseen locations)" value={m ? `${num(m.localization.mae_km)} km` : "—"} sub={m && `physics baseline ${num(m.baseline.mae_km)} km`} tone="good" />
        <Stat label="API" value={health.data?.status ?? (health.isError ? "offline" : "…")}
              sub={health.data ? `${health.data.model_version} · data ${health.data.dataset_hash}` : health.isError ? "run `make api`" : ""}
              tone={health.data?.status === "ok" ? "good" : "bad"} />
      </div>

      {m && <Card title="Honest claim"><p className="text-sm text-ink-1">{m.honest_claim}</p></Card>}

      {net.data && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card title="Substations">
            <DataTable columns={["Bus", "km", "Role", "Metered"]}
              rows={net.data.buses.map((b) => [busName(b.id), b.km, b.role, b.metered ? "◉ yes" : "no (dashed)"])} />
          </Card>
          <Card title="Metered sections (model resolution)">
            <DataTable columns={["Section", "From km", "To km", "Candidate taps"]}
              rows={net.data.sections.map((s) => [sectionName(s.id), s.km_from, s.km_to,
                net.data!.taps.filter((t) => t.km > s.km_from && t.km <= s.km_to).length])} />
          </Card>
        </div>
      )}
      <AssumptionsDrawer open={drawer} onClose={() => setDrawer(false)} assumptions={net.data?.assumptions ?? []} />
    </div>
  );
}
