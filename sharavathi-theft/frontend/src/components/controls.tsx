import type { Measurements, SensorProfile } from "../api/types";
import { channelLabel } from "../lib/labels";
import { CORRIDOR_KM, useScenarioStore } from "../store/useScenarioStore";
import { DataTable, num } from "./ui";

function Slider({ id, label, value, min, max, step, unit, onChange, disabled, fmt }: {
  id: string; label: string; value: number; min: number; max: number; step: number; unit?: string;
  onChange: (v: number) => void; disabled?: boolean; fmt?: (v: number) => string;
}) {
  return (
    <div className={disabled ? "opacity-50" : ""}>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="field-label">{label}</label>
        <output htmlFor={id} className="mono text-sm text-ink-1">{fmt ? fmt(value) : value}{unit && <span className="text-ink-3"> {unit}</span>}</output>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled}
             onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function TheftSlider() {
  const { request: r, setRequest, setTheftKm } = useScenarioStore();
  return (
    <fieldset className="space-y-3">
      <legend className="sr-only">Unauthorized extraction injection</legend>
      <label className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium">Inject unmetered extraction</span>
        <input type="checkbox" className="h-4 w-4 accent-[var(--accent)]" checked={r.theftOn}
               onChange={(e) => setRequest({ theftOn: e.target.checked })} aria-label="Inject unmetered extraction" />
      </label>
      <Slider id="theft-km" label="Location (snaps to 10-km tap)" value={r.theftKm} min={0} max={CORRIDOR_KM} step={1} unit="km"
              onChange={setTheftKm} disabled={!r.theftOn} />
      <Slider id="theft-mw" label="Magnitude" value={r.theftMw} min={0} max={40} step={0.5} unit="MW"
              onChange={(v) => setRequest({ theftMw: v })} disabled={!r.theftOn} fmt={(v) => v.toFixed(1)} />
    </fieldset>
  );
}

export function LoadNoiseSliders() {
  const { request: r, setRequest } = useScenarioStore();
  return (
    <div className="space-y-3">
      <Slider id="load-scale" label="System load scale" value={r.loadScale} min={0.5} max={1.15} step={0.01}
              onChange={(v) => setRequest({ loadScale: v })} fmt={(v) => `×${v.toFixed(2)}`} />
      <Slider id="noise" label="Sensor noise multiplier" value={r.noiseMult} min={0} max={4} step={0.25}
              onChange={(v) => setRequest({ noiseMult: v })} fmt={(v) => `×${v.toFixed(2)}`} />
    </div>
  );
}

const PROFILES: { id: SensorProfile; label: string }[] = [
  { id: "major_only", label: "Major substations only" },
  { id: "major_plus_mid", label: "+ Sira/KB-Cross" },
  { id: "dense", label: "Dense (every tap)" },
];

export function SensorToggle() {
  const { request: r, setRequest } = useScenarioStore();
  return (
    <div>
      <label htmlFor="profile" className="field-label">Sensor profile</label>
      <select id="profile" className="mt-1 w-full" value={r.sensorProfile}
              onChange={(e) => setRequest({ sensorProfile: e.target.value as SensorProfile })}>
        {PROFILES.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
      </select>
      {r.sensorProfile !== "major_only" && (
        <p className="mt-1 text-[11px] text-ink-3">Models were trained on major_only; extra channels are simulated but only major_only channels feed the models.</p>
      )}
    </div>
  );
}

const FAULT_CHANNELS = ["SHARAVATHI", "SHIMOGA", "NELAMANGALA", "PEENYA"].flatMap((b) =>
  ["p", "q", "vm"].map((k) => `m_${b}_${k}`));

export function FaultInjector() {
  const { request: r, setRequest } = useScenarioStore();
  const paramLabel = { bias: "Bias (%)", drift: "Drift rate (%/step)", line_outage: "Line index (0–37)" } as Record<string, string>;
  return (
    <div className="space-y-2">
      <label htmlFor="fault" className="field-label">Fault injector</label>
      <select id="fault" className="w-full" value={r.fault}
              onChange={(e) => {
                const fault = e.target.value as typeof r.fault;
                setRequest({ fault, faultParam: fault === "line_outage" ? 20 : fault === "drift" ? 1.5 : 8 });
              }}>
        <option value="none">None</option>
        <option value="bias">Sensor bias</option>
        <option value="drift">Sensor drift</option>
        <option value="missing">Missing channel</option>
        <option value="line_outage">Line outage</option>
      </select>
      {r.fault !== "none" && r.fault !== "line_outage" && (
        <div>
          <label htmlFor="fault-ch" className="field-label">Channel</label>
          <select id="fault-ch" className="mt-1 w-full" value={r.faultChannel} onChange={(e) => setRequest({ faultChannel: e.target.value })}>
            {FAULT_CHANNELS.map((c) => <option key={c} value={c}>{channelLabel(c)}</option>)}
          </select>
        </div>
      )}
      {paramLabel[r.fault] && (
        <div className="flex items-center justify-between gap-2">
          <label htmlFor="fault-param" className="field-label">{paramLabel[r.fault]}</label>
          <input id="fault-param" type="number" className="w-24" value={r.faultParam}
                 step={r.fault === "line_outage" ? 1 : 0.5} min={r.fault === "line_outage" ? 0 : -50} max={r.fault === "line_outage" ? 37 : 50}
                 onChange={(e) => setRequest({ faultParam: Number(e.target.value) })} />
        </div>
      )}
    </div>
  );
}

export function SeedInput() {
  const { request: r, setRequest } = useScenarioStore();
  return (
    <div className="flex items-center justify-between gap-2">
      <label htmlFor="seed" className="field-label">Noise seed</label>
      <div className="flex gap-1">
        <input id="seed" type="number" className="w-24" value={r.seed} onChange={(e) => setRequest({ seed: Number(e.target.value) })} />
        <button className="btn !px-2" aria-label="Random seed" title="Random seed"
                onClick={() => setRequest({ seed: Math.floor(Math.random() * 1e6) })}>⟳</button>
      </div>
    </div>
  );
}

export function MeasurementTable({ measurements, expected }: { measurements: Measurements; expected: Measurements }) {
  return (
    <DataTable columns={["Channel", "Key", "Measured", "Expected", "Δ"]}
      rows={Object.keys(measurements).map((k) => {
        const m = measurements[k], e = expected[k];
        return [channelLabel(k), k, m == null ? "missing" : num(m, 3), num(e, 3), m != null && e != null ? (m - e).toFixed(3) : "—"];
      })} />
  );
}
