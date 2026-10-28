import { useState } from "react";
import { useFeatures, useMitigationPlan, useStrategies } from "../api/hooks";
import { AdvisorPanel } from "../components/AdvisorPanel";
import { logUser } from "../lib/activity";
import { Card, ErrorState, Skeleton } from "../components/ui";

const PLANNED = [
  { t: "Load balancing", d: "Redistribute flows across parallel paths to relieve stressed segments after a suspected extraction." },
  { t: "Generation redispatch", d: "Adjust injections to restore voltage and loading margins (OPF-based)." },
  { t: "Overload prediction", d: "Forecast short-term demand per bus and flag segments at risk of exceeding thermal limits." },
];

export default function Mitigation() {
  const f = useFeatures();
  const s = useStrategies();
  const plan = useMitigationPlan();
  const [strategy, setStrategy] = useState("load_balance");
  const enabled = !!f.data?.mitigation_enabled;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">Mitigation & load balancing</h1>
        {!enabled && <span className="rounded-full border border-line bg-surface-2 px-2.5 py-0.5 text-xs font-semibold text-ink-2">🔒 Coming soon</span>}
      </div>
      <p className="text-sm text-ink-2">
        Future work — interfaces exist (<span className="mono">future/mitigation.py</span>, <span className="mono">/api/mitigation/*</span>) but no control
        logic is implemented. Gated by <span className="mono">mitigation_enabled</span> in <span className="mono">feature_flags.yaml</span>.
      </p>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {PLANNED.map((p) => <Card key={p.t} title={p.t}><p className="text-sm text-ink-2">{p.d}</p></Card>)}
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[20rem_1fr]">
        <Card title="Strategy">
          {s.isPending ? <Skeleton h={80} /> : s.isError ? <ErrorState error={s.error} /> : (
            <fieldset disabled={!enabled} className="space-y-2">
              <legend className="sr-only">Mitigation strategy</legend>
              {s.data.strategies.map((x) => (
                <label key={x.id} className="flex items-center gap-2 text-sm">
                  <input type="radio" name="strategy" value={x.id} checked={strategy === x.id} onChange={() => setStrategy(x.id)} />
                  {x.label} {!x.enabled && <span className="text-xs text-ink-3">(disabled)</span>}
                </label>
              ))}
              <button className="btn btn-primary mt-2 w-full" onClick={() => { logUser(`Requested mitigation plan (${strategy})`); plan.mutate({ scenario_id: "latest", strategy }); }}>Plan</button>
            </fieldset>
          )}
        </Card>
        <Card title="Plan & impact">
          {plan.data ? <pre className="mono text-xs">{JSON.stringify(plan.data, null, 2)}</pre>
            : plan.isError ? <ErrorState error={plan.error} />
            : (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3" aria-label="Empty plan panel">
                {["Planned actions", "Max loading after", "Min voltage after"].map((h) => (
                  <div key={h} className="rounded-md border border-dashed border-line p-4 text-center text-xs text-ink-3">{h}<div className="mt-2 text-lg">—</div></div>
                ))}
              </div>
            )}
        </Card>
      </div>
      <AdvisorPanel title="Knowledge-base advisor (RAG)" />
      <p className="text-xs text-ink-3">The advisor is available while mitigation is locked because it only retrieves and cites guidance; it never plans or executes control actions.</p>
    </div>
  );
}
