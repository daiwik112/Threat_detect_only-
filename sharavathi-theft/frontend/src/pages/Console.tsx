import { ConsoleView } from "../components/ConsoleDock";

export default function ConsolePage() {
  return (
    <div className="flex h-full min-h-[70vh] flex-col gap-3">
      <div>
        <h1 className="text-xl font-semibold">Activity console</h1>
        <p className="text-sm text-ink-2">
          Live timeline of everything happening in the app: your actions, API calls, simulations (where the extraction was injected),
          detections (▲ suspected unauthorized extraction alerts), advisor questions, replay alerts and errors. Shared by every open
          browser tab; press <kbd className="mono">`</kbd> on any page for the docked version.
        </p>
      </div>
      <div className="card min-h-0 flex-1 overflow-hidden"><ConsoleView /></div>
    </div>
  );
}
