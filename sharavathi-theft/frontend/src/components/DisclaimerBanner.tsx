export function DisclaimerBanner() {
  return (
    <div role="note" className="flex items-center gap-2 border-b border-warning/40 bg-warning/10 px-4 py-1.5 text-xs text-ink-1">
      <span aria-hidden className="text-warning">◆</span>
      <span>
        <strong>Simulated results</strong> — suspected anomalies need field verification. The corridor is a
        proposed simplified model, not a verified KPTCL replica.
      </span>
    </div>
  );
}
