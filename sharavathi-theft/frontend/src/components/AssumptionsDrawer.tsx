import { useEffect, useRef } from "react";

export function AssumptionsDrawer({ open, onClose, assumptions }: {
  open: boolean; onClose: () => void; assumptions: string[];
}) {
  const ref = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    ref.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/40" onClick={onClose}>
      <aside role="dialog" aria-modal="true" aria-labelledby="assumptions-title"
             className="h-full w-[28rem] max-w-full overflow-y-auto border-l border-line bg-surface-1 p-5"
             onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-center justify-between">
          <h2 id="assumptions-title" className="text-base font-semibold">Modelling assumptions</h2>
          <button ref={ref} className="btn !px-2" onClick={onClose} aria-label="Close assumptions">✕</button>
        </div>
        <p className="mb-3 text-xs text-ink-3">
          Every value below is marked <code className="mono">source: assumed</code> in config and should be
          replaced with verified KPTCL / SLDC data when available.
        </p>
        <ul className="space-y-2 text-sm">
          {assumptions.map((a) => (
            <li key={a} className="flex gap-2 rounded-md bg-surface-2 p-2.5">
              <span aria-hidden className="text-warning">◆</span><span className="text-ink-1">{a}</span>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
