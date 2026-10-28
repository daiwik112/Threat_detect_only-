import { useState, type ReactNode } from "react";
import { create } from "zustand";
import { ApiError } from "../api/client";

/* ------------------------------------------------------------------ cards */
export function Card({ title, subtitle, actions, children, className = "" }: {
  title?: string; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card p-4 ${className}`} aria-label={title}>
      {(title || actions) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <p className="mt-1 text-xs text-ink-3">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export interface TableSpec { columns: string[]; rows: (string | number | null)[][] }

/** Chart card with a Chart/Table switch — every chart has an accessible table view. */
export function ChartCard({ title, subtitle, table, children, actions, className }: {
  title: string; subtitle?: ReactNode; table?: TableSpec; children: ReactNode; actions?: ReactNode; className?: string;
}) {
  const [asTable, setAsTable] = useState(false);
  return (
    <Card title={title} subtitle={subtitle} className={className} actions={
      <div className="flex items-center gap-2">
        {actions}
        {table && (
          <button className="btn !px-2 !py-0.5 text-xs" aria-pressed={asTable}
                  onClick={() => setAsTable((v) => !v)}>
            {asTable ? "Chart" : "Table"}
          </button>
        )}
      </div>
    }>
      {asTable && table ? <DataTable {...table} /> : children}
    </Card>
  );
}

export function DataTable({ columns, rows }: TableSpec) {
  return (
    <div className="max-h-80 overflow-auto">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-surface-1">
          <tr>{columns.map((c) => <th key={c} scope="col" className="border-b border-line py-1.5 pr-3 font-semibold text-ink-2">{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/60">
              {r.map((v, j) => <td key={j} className="py-1 pr-3 mono text-ink-1">{v ?? "—"}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "good" | "bad" }) {
  return (
    <div className="card px-4 py-3">
      <div className="card-title">{label}</div>
      <div className="mt-1 text-2xl font-semibold mono">{value}</div>
      {sub && <div className={`mt-0.5 text-xs ${tone === "good" ? "text-[var(--status-good-text)]" : tone === "bad" ? "text-[var(--status-critical-text)]" : "text-ink-3"}`}>{sub}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ states */
export function Skeleton({ h = 200 }: { h?: number }) {
  return <div className="skeleton w-full" style={{ height: h }} aria-hidden />;
}

export function ErrorState({ error, hint }: { error: unknown; hint?: string }) {
  const e = error as ApiError;
  return (
    <div role="alert" className="rounded-md border border-critical/50 bg-critical/10 p-3 text-sm">
      <div className="font-semibold text-[var(--status-critical-text)]">⚠ {e?.code ?? "error"}</div>
      <div className="mt-1 text-ink-2">{e?.message ?? String(error)}</div>
      {hint && <div className="mt-1 text-xs text-ink-3">{hint}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ status */
export function StatusBadge({ status, large }: { status: "normal" | "suspicious"; large?: boolean }) {
  const s = status === "suspicious"
    ? { icon: "▲", text: "Suspicious", cls: "bg-critical/15 text-[var(--status-critical-text)] border-critical/60" }
    : { icon: "●", text: "Normal", cls: "bg-good/15 text-[var(--status-good-text)] border-good/60" };
  return (
    <span data-testid="status-badge" data-status={status}
          className={`inline-flex items-center gap-2 rounded-full border font-semibold ${s.cls} ${large ? "px-4 py-1.5 text-lg" : "px-2.5 py-0.5 text-xs"}`}>
      <span aria-hidden>{s.icon}</span>{s.text}
    </span>
  );
}

/* ------------------------------------------------------------------ toasts */
interface Toast { id: number; text: string; kind: "error" | "info" }
export const useToasts = create<{ toasts: Toast[]; push: (text: string, kind?: Toast["kind"]) => void; drop: (id: number) => void }>((set) => ({
  toasts: [],
  push: (text, kind = "error") => {
    const id = Date.now() + Math.random();
    set((s) => ({ toasts: [...s.toasts, { id, text, kind }] }));
    setTimeout(() => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })), 6000);
  },
  drop: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

export function Toaster() {
  const { toasts, drop } = useToasts();
  return (
    <div className="fixed bottom-4 right-4 z-50 flex w-96 flex-col gap-2" aria-live="assertive">
      {toasts.map((t) => (
        <div key={t.id} role="status"
             className={`card flex items-start gap-3 p-3 text-sm shadow-lg ${t.kind === "error" ? "border-critical/60" : ""}`}>
          <span aria-hidden>{t.kind === "error" ? "⚠" : "ℹ"}</span>
          <span className="flex-1 text-ink-1">{t.text}</span>
          <button className="text-ink-3 hover:text-ink-1" aria-label="Dismiss" onClick={() => drop(t.id)}>✕</button>
        </div>
      ))}
    </div>
  );
}

export const pct = (v: number | null | undefined, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 1) => (v == null ? "—" : v.toFixed(d));
