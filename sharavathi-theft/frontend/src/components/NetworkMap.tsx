/**
 * Corridor diagram (React Flow). Buses on a km axis; metered = sensor icon, unmetered =
 * dashed; segments coloured by base-case loading (single-hue sequential ramp). Taps are
 * clickable when `onPickKm` is given. Overlays: selected theft km (ghost), estimated km
 * (pulsing) with 80 % interval band, true km (after reveal).
 */
import { ReactFlow, Background, Handle, Position, useNodesState, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo } from "react";
import type { NetworkResponse } from "../api/types";
import { busName } from "../lib/labels";
import { useTheme } from "../lib/useTheme";

const PX_PER_KM = 2.2;
const x = (km: number) => km * PX_PER_KM;

type D = Record<string, unknown>;

const hidden = { opacity: 0, width: 1, height: 1, minWidth: 0, minHeight: 0, border: 0 };
function Handles() {
  return (<>
    <Handle type="target" position={Position.Left} style={hidden} isConnectable={false} />
    <Handle type="source" position={Position.Right} style={hidden} isConnectable={false} />
  </>);
}

function BusNode({ data }: NodeProps<Node<D>>) {
  const metered = data.metered as boolean;
  return (
    <div className="relative h-3.5 w-3.5" title={(data.tooltip as string) ?? (metered ? "Metered substation" : "Unmetered junction")}>
      <Handles />
      <div className={`h-3.5 w-3.5 rounded-sm ${metered ? "bg-[var(--text-primary)]" : "border-2 border-dashed border-[var(--text-secondary)] bg-[var(--surface-1)]"}`} />
      <div className="absolute left-1/2 top-5 flex -translate-x-1/2 flex-col items-center">
      <div className="whitespace-nowrap text-[11px] font-semibold text-ink-1">
        {metered && <span aria-label="metered" className="mr-1 text-accent">◉</span>}
        {busName(data.id as string)}
      </div>
      <div className="mono whitespace-nowrap text-[10px] text-ink-3">km {data.km as number}{metered ? "" : " · unmetered"}</div>
      </div>
    </div>
  );
}

function TapNode({ data }: NodeProps<Node<D>>) {
  const sel = data.selected as boolean;
  const clickable = data.clickable as boolean;
  return (
    <div className="relative h-3 w-3"><Handles />
    {/* 24 px hit area centred on the 12 px dot (target size), dot position unchanged */}
    <button type="button" tabIndex={clickable ? 0 : -1} disabled={!clickable}
            aria-label={`Tap at km ${data.km}`} title={`${data.id} · km ${data.km}`}
            className={`nodrag group absolute -inset-1.5 flex items-center justify-center rounded-full ${clickable ? "cursor-pointer" : "cursor-default"}`}>
      <span className={`block h-3 w-3 rounded-full border ${sel ? "border-accent bg-accent" : "border-[var(--text-muted)] bg-[var(--surface-1)]"} ${clickable ? "group-hover:scale-150" : ""} transition-transform`} />
    </button>
    </div>
  );
}

function MarkerNode({ data }: NodeProps<Node<D>>) {
  const kind = data.kind as "est" | "truth" | "ghost";
  if (kind === "est")
    return (
      <div data-testid="localization-marker" className="relative" title={`Estimated km ${data.km}`}>
        <span className="pulse-ring absolute -left-2 -top-2 h-8 w-8 rounded-full border-2 border-critical" />
        <span className="block h-4 w-4 rounded-full border-2 border-[var(--surface-1)] bg-critical" />
        <span className="absolute left-1/2 top-6 -translate-x-1/2 whitespace-nowrap rounded bg-surface-2 px-1.5 text-[10px] font-semibold text-ink-1">
          est. km {Math.round(data.km as number)}
        </span>
      </div>
    );
  if (kind === "truth")
    return (
      <div className="flex flex-col items-center" style={{ marginLeft: -26 }} title={`True km ${data.km}`}>
        <span className="whitespace-nowrap text-[10px] font-semibold text-ink-1">true {data.km as number}</span>
        <span className="text-ink-1" aria-hidden>▼</span>
      </div>
    );
  return <span className="block h-5 w-5 rounded-full border-2 border-dashed border-accent" title="Selected injection point" />;
}

function BandNode({ data }: NodeProps<Node<D>>) {
  return <div className="rounded-sm bg-critical/20 ring-1 ring-critical/50" style={{ width: data.w as number, height: 10 }} title="80% interval" />;
}

function TickNode({ data }: NodeProps<Node<D>>) {
  return <div className="mono -ml-3 text-[10px] text-ink-3">{data.km as number} km</div>;
}

const nodeTypes = { bus: BusNode, tap: TapNode, marker: MarkerNode, band: BandNode, tick: TickNode };

export interface MapOverlay {
  selectedKm?: number | null;
  estimate?: { km: number; lo: number; hi: number } | null;
  truthKm?: number | null;
}

function busTooltip(id: string, km: number, metered: boolean, r?: Record<string, number | null> | null): string {
  const head = `${busName(id)} · km ${km} · ${metered ? "metered" : "unmetered"}`;
  if (!r || !metered) return head;
  const f = (k: string, d: number, u: string) => (r[k] == null ? "missing" : `${(r[k] as number).toFixed(d)} ${u}`);
  return `${head}\nV ${f(`m_${id}_vm`, 4, "pu")} · P ${f(`m_${id}_p`, 1, "MW")} · Q ${f(`m_${id}_q`, 1, "Mvar")}`;
}

export function NetworkMap({ network, overlay = {}, onPickKm, height = 230, readings }: {
  network: NetworkResponse; overlay?: MapOverlay; onPickKm?: (km: number) => void; height?: number;
  /** latest measurements → hover tooltips with V / P / Q at metered buses */
  readings?: Record<string, number | null> | null;
}) {
  const t = useTheme();
  const { nodes, edges } = useMemo(() => {
    const nodes: Node<D>[] = [];
    const Y = 60;
    for (const b of network.buses)
      nodes.push({ id: b.id, type: "bus", position: { x: x(b.km) - 7, y: Y - 7 }, data: { ...b, tooltip: busTooltip(b.id, b.km, b.metered, readings) }, draggable: false, selectable: false });
    for (const tp of network.taps)
      nodes.push({ id: tp.id, type: "tap", position: { x: x(tp.km) - 6, y: Y - 6 }, draggable: false, selectable: false,
                   data: { ...tp, clickable: !!onPickKm, selected: overlay.selectedKm === tp.km } });
    for (let km = 0; km <= 350; km += 50)
      nodes.push({ id: `tick-${km}`, type: "tick", position: { x: x(km), y: Y + 70 }, data: { km }, draggable: false, selectable: false });
    if (overlay.selectedKm != null && !overlay.estimate)
      nodes.push({ id: "ghost", type: "marker", position: { x: x(overlay.selectedKm) - 10, y: Y - 10 }, data: { kind: "ghost", km: overlay.selectedKm }, draggable: false, selectable: false, zIndex: 5 });
    if (overlay.estimate) {
      const { km, lo, hi } = overlay.estimate;
      nodes.push({ id: "band", type: "band", position: { x: x(lo), y: Y + 40 }, data: { w: Math.max(x(hi) - x(lo), 4) }, draggable: false, selectable: false });
      nodes.push({ id: "est", type: "marker", position: { x: x(km) - 8, y: Y - 40 }, data: { kind: "est", km }, draggable: false, selectable: false, zIndex: 10 });
    }
    if (overlay.truthKm != null)
      nodes.push({ id: "truth", type: "marker", position: { x: x(overlay.truthKm), y: Y - 70 }, data: { kind: "truth", km: overlay.truthKm }, draggable: false, selectable: false, zIndex: 10 });

    const max = Math.max(...network.lines.map((l) => l.loading_pct ?? 0), 1);
    const ramp = t.seq.slice(1);
    const id = (busOrTap: string) => busOrTap;
    const edges: Edge[] = network.lines.map((l) => {
      const v = (l.loading_pct ?? 0) / max;
      return {
        id: l.id, source: id(l.from_bus), target: id(l.to_bus), type: "straight",
        style: { stroke: ramp[Math.min(ramp.length - 1, Math.floor(v * ramp.length))], strokeWidth: 4 },
        focusable: false,
      };
    });
    return { nodes, edges };
  }, [network, overlay.selectedKm, overlay.estimate, overlay.truthKm, onPickKm, t, readings]);

  // Nodes are rebuilt whenever the overlay / readings change. A rebuilt node without `measured` is hidden and its
  // handles reset until React Flow re-measures it, which never happens because its DOM size is unchanged — so the
  // corridor vanished after every run. Keep the measured sizes in state and carry them onto the rebuilt nodes.
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState<Node<D>>(nodes);
  useEffect(() => {
    setRfNodes((prev) => {
      const measured = new Map(prev.map((n) => [n.id, n.measured]));
      return nodes.map((n) => ({ ...n, measured: measured.get(n.id) }));
    });
  }, [nodes, setRfNodes]);

  return (
    <div style={{ height }} className="w-full overflow-hidden rounded-md bg-surface-1" aria-label="Corridor diagram">
      <ReactFlow
        nodes={rfNodes} onNodesChange={onNodesChange} edges={edges} nodeTypes={nodeTypes}
        fitView fitViewOptions={{ padding: 0.08 }} minZoom={0.3} maxZoom={3}
        nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}
        zoomOnScroll={false} panOnScroll={false} preventScrolling={false}
        onNodeClick={(_, n) => { if (onPickKm && n.type === "tap") onPickKm((n.data as { km: number }).km); }}
        proOptions={{ hideAttribution: true }}
        colorMode={t.surface === "#1a1a19" ? "dark" : "light"}
      >
        <Background color={t.grid} gap={32} />
      </ReactFlow>
    </div>
  );
}

export function LoadingLegend({ network }: { network: NetworkResponse }) {
  const t = useTheme();
  const vals = network.lines.map((l) => l.loading_pct ?? 0);
  return (
    <div className="flex items-center gap-2 text-[11px] text-ink-2">
      <span>Segment loading (base case)</span>
      <span className="mono">{Math.min(...vals).toFixed(0)}%</span>
      <span className="h-2 w-24 rounded-sm" style={{ background: `linear-gradient(90deg, ${t.seq.slice(1).join(",")})` }} aria-hidden />
      <span className="mono">{Math.max(...vals).toFixed(0)}%</span>
      <span className="ml-3"><span className="text-accent">◉</span> metered</span>
      <span className="ml-2"><span className="inline-block h-2.5 w-2.5 border-2 border-dashed border-[var(--text-secondary)] align-middle" /> unmetered</span>
    </div>
  );
}
