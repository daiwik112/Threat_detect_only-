import type { Data } from "plotly.js";
import { useEffect, useRef, useState } from "react";
import { wsUrl } from "../api/client";
import type { ReplayMessage } from "../api/types";
import { LocalizationChart } from "../components/charts";
import { Plot, useTheme } from "../components/Plot";
import { Card, ErrorState, StatusBadge } from "../components/ui";
import { logUser } from "../lib/activity";
import { LABEL_NAMES } from "../lib/labels";

type Conn = "idle" | "connecting" | "open" | "closed" | "error";

export default function Replay() {
  const t = useTheme();
  const ws = useRef<WebSocket | null>(null);
  const [conn, setConn] = useState<Conn>("idle");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [msgs, setMsgs] = useState<ReplayMessage[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const connect = () => {
    ws.current?.close();
    setConn("connecting"); setErr(null); setMsgs([]);
    const s = new WebSocket(wsUrl("/ws/replay"));
    s.onopen = () => { setConn("open"); setPlaying(true); s.send(JSON.stringify({ cmd: "speed", value: speed })); };
    s.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.error) { setErr(m.error.message); return; }
      setMsgs((prev) => [...prev.slice(-199), m as ReplayMessage]);
    };
    s.onerror = () => { setConn("error"); setErr("WebSocket error — is the API running on :8000?"); };
    s.onclose = () => { setConn((c) => (c === "error" ? c : "closed")); setPlaying(false); };
    ws.current = s;
  };
  useEffect(() => () => ws.current?.close(), []);

  const send = (cmd: object) => ws.current?.readyState === WebSocket.OPEN && ws.current.send(JSON.stringify(cmd));
  const toggle = () => { send({ cmd: playing ? "pause" : "play" }); setPlaying(!playing); logUser(playing ? "Paused replay" : "Resumed replay"); };
  const changeSpeed = (v: number) => { setSpeed(v); send({ cmd: "speed", value: v }); logUser(`Replay speed ×${v}`); };

  // scrubber: follow the live edge, or pin a past tick
  const [follow, setFollow] = useState(true);
  const [cursor, setCursor] = useState(0);
  const idx = follow ? msgs.length - 1 : Math.min(cursor, msgs.length - 1);
  const last = msgs[idx];
  const alerts = msgs.filter((m) => m.detect.status === "suspicious");
  const hits = alerts.filter((m) => m.truth.label !== "normal").length;
  const timeline: Data[] = [
    { type: "scatter", mode: "lines", x: msgs.map((m) => m.t), y: msgs.map((m) => m.detect.anomaly_score), name: "Anomaly score",
      line: { color: t.series[0], width: 2 }, hovertemplate: "t=%{x}<br>score %{y:.3f}<extra></extra>" },
    { type: "scatter", mode: "markers", x: alerts.map((m) => m.t), y: alerts.map((m) => m.detect.anomaly_score), name: "Alert (suspicious)",
      marker: { symbol: "triangle-up", size: 10, color: t.status.critical, line: { color: t.surface, width: 2 } },
      customdata: alerts.map((m) => LABEL_NAMES[m.truth.label] ?? m.truth.label),
      hovertemplate: "t=%{x} · ALERT<br>truth: %{customdata}<extra></extra>" },
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Live replay</h1>
          <p className="text-sm text-ink-2">Held-out test scenarios streamed over <span className="mono">/ws/replay</span>, one per tick, with live detection.</p>
        </div>
        <div className="flex items-center gap-2">
          {conn !== "open"
            ? <button className="btn btn-primary" onClick={connect}>{conn === "connecting" ? "Connecting…" : "▶ Connect & play"}</button>
            : <button className="btn btn-primary" onClick={toggle} aria-pressed={playing}>{playing ? "❚❚ Pause" : "▶ Play"}</button>}
          <label htmlFor="speed" className="field-label">Speed</label>
          <select id="speed" value={speed} onChange={(e) => changeSpeed(Number(e.target.value))}>
            {[0.5, 1, 2, 4, 8].map((v) => <option key={v} value={v}>×{v}</option>)}
          </select>
          <span className="mono text-xs text-ink-3" aria-live="polite">{conn}</span>
        </div>
      </div>
      {err && <ErrorState error={{ code: "replay", message: err }} />}

      <Card title="Alert timeline" subtitle={msgs.length ? `${msgs.length} scenarios · ${alerts.length} alerts · ${hits} on true anomalies` : "Connect to start the stream"}>
        <Plot ariaLabel="Anomaly score over time with alert markers" height={230} data={timeline} filename="replay-timeline" layout={{
          xaxis: { title: { text: "tick" } }, yaxis: { title: { text: "anomaly score" }, range: [0, 1.05] },
          shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: t.text2, dash: "dash", width: 1 } }],
        }} />
      </Card>

      {msgs.length > 0 && (
        <div className="card flex flex-wrap items-center gap-3 p-3">
          <label htmlFor="scrubber" className="field-label">Scrub</label>
          <input id="scrubber" type="range" min={0} max={Math.max(msgs.length - 1, 0)} value={idx} className="flex-1"
                 aria-valuetext={`tick ${last?.t}`} data-testid="replay-scrubber"
                 onChange={(e) => { setFollow(false); setCursor(Number(e.target.value)); }} />
          <span className="mono text-xs text-ink-2">tick {last?.t} / {msgs[msgs.length - 1].t}</span>
          <button className="btn !py-0.5 text-xs" aria-pressed={follow} onClick={() => setFollow(true)} disabled={follow}>⇥ Follow live</button>
        </div>
      )}

      {last && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[18rem_1fr]">
          <Card title={`Tick ${last.t}`}>
            <div className="space-y-2 text-sm">
              <StatusBadge status={last.detect.status} large />
              <div>Score <span className="mono">{last.detect.anomaly_score.toFixed(3)}</span></div>
              <div>Truth: <strong>{LABEL_NAMES[last.truth.label] ?? last.truth.label}</strong>
                {last.truth.km != null && <> at <span className="mono">{last.truth.km} km</span>, {last.truth.magnitude_mw.toFixed(1)} MW</>}</div>
              <div className="mono text-[10px] text-ink-3">{last.scenario_id}</div>
            </div>
          </Card>
          <LocalizationChart localize={last.localize} truthKm={last.truth.km} height={230}
            title={last.detect.status === "suspicious" ? "Localization (live)" : "Localization (live · status normal)"} />
        </div>
      )}
    </div>
  );
}
