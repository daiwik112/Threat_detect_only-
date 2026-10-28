/**
 * Activity console plumbing.
 *   logUser("Ran analysis", {...})  → POST /api/events (fire-and-forget) so user actions join the
 *                                     server timeline seen by every open console
 *   useEventStream()                 → keeps a WebSocket to /ws/events open (auto-reconnect)
 */
import { useEffect } from "react";
import { API_BASE, wsUrl } from "../api/client";
import { useConsoleStore, type AppEvent } from "../store/useConsoleStore";

export function logUser(message: string, data: Record<string, unknown> = {}): void {
  if (typeof fetch === "undefined") return;
  try {
    void fetch(`${API_BASE}/api/events`, {
      method: "POST", keepalive: true,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message.slice(0, 300), category: "user", data }),
    }).catch(() => undefined);                 // console logging must never break the app
  } catch {
    /* ignore */
  }
}

export function useEventStream(): void {
  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry = 0;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const { addEvents, setConnected } = useConsoleStore.getState();
    let batch: AppEvent[] = [];
    let flush: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      ws = new WebSocket(wsUrl("/ws/events?backlog=300"));
      ws.onopen = () => { retry = 0; setConnected(true); };
      ws.onmessage = (m) => {
        const ev = JSON.parse(m.data);
        if (ev.type === "ping" || typeof ev.seq !== "number") return;
        if (useConsoleStore.getState().paused) return;
        batch.push(ev as AppEvent);                         // coalesce bursts into one render
        if (!flush) flush = setTimeout(() => { addEvents(batch); batch = []; flush = undefined; }, 100);
      };
      ws.onclose = () => {
        setConnected(false);
        if (!stopped) timer = setTimeout(connect, Math.min(10_000, 500 * 2 ** retry++));
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => { stopped = true; clearTimeout(timer); clearTimeout(flush); ws?.close(); };
  }, []);
}
