"""
backend/app/events.py — In-memory activity event bus for the app console.

    from app.events import bus
    bus.publish("detection", "alert", "Suspected unauthorized extraction near km 145", {"km": 145})

Events: {seq, ts, level, category, message, data, request_id, source}
  level     debug | info | warn | alert | error
  category  system | api | simulation | detection | localization | rag | replay | batch |
            mitigation | user
  source    server | client

A ring buffer keeps the last MAX_EVENTS. WebSocket subscribers get a backlog and then live events.
publish() is thread-safe: sync endpoints run in FastAPI's threadpool, so delivery to asyncio
subscribers goes through loop.call_soon_threadsafe. Nothing is persisted; a restart clears the log.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from contextvars import ContextVar
from typing import Any

MAX_EVENTS = 2000
LEVELS = ("debug", "info", "warn", "alert", "error")
CATEGORIES = ("system", "api", "simulation", "detection", "localization", "rag", "replay", "batch",
              "mitigation", "user")

# request id of the HTTP request currently being handled (set by the middleware)
current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)


class EventBus:
    def __init__(self, maxlen: int = MAX_EVENTS):
        self._buf: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0
        self._subs: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, category: str, level: str, message: str, data: dict[str, Any] | None = None,
                source: str = "server", request_id: str | None = None) -> dict:
        if level not in LEVELS:
            level = "info"
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "ts": time.time(), "level": level, "category": category,
                  "message": message[:500], "data": data or {}, "source": source,
                  "request_id": request_id or current_request_id.get()}
            self._buf.append(ev)
            subs = list(self._subs)
        for q in subs:
            self._deliver(q, ev)
        return ev

    def _deliver(self, q: asyncio.Queue, ev: dict) -> None:
        def put():
            if q.full():                       # slow consumer: drop its oldest pending event
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(ev)
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running is loop:
            put()
        elif loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(put)

    def since(self, seq: int = 0, limit: int = 500, min_level: str = "debug",
              categories: set[str] | None = None) -> list[dict]:
        floor = LEVELS.index(min_level) if min_level in LEVELS else 0
        with self._lock:
            evs = [e for e in self._buf if e["seq"] > seq and LEVELS.index(e["level"]) >= floor
                   and (not categories or e["category"] in categories)]
        return evs[-limit:]

    @property
    def last_seq(self) -> int:
        return self._seq

    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


bus = EventBus()
