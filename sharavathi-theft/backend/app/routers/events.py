"""Activity console routes: event history, client (user-action) events, live WebSocket stream."""

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.events import CATEGORIES, LEVELS, bus

router = APIRouter(tags=["console"])


class Event(BaseModel):
    seq: int
    ts: float
    level: str
    category: str
    message: str
    data: dict
    source: str
    request_id: Optional[str]


class EventsResponse(BaseModel):
    events: list[Event]
    last_seq: int


class ClientEvent(BaseModel):
    """A user action reported by the browser."""
    message: str = Field(..., min_length=1, max_length=300)
    level: Literal["debug", "info", "warn", "alert", "error"] = "info"
    category: Literal["user"] = "user"
    data: dict = Field(default_factory=dict)


@router.get("/api/events", response_model=EventsResponse)
def get_events(since: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=2000),
               min_level: str = Query("debug"), category: Optional[str] = None):
    cats = set(category.split(",")) if category else None
    return {"events": bus.since(since, limit, min_level if min_level in LEVELS else "debug", cats),
            "last_seq": bus.last_seq}


@router.post("/api/events", response_model=Event, status_code=201)
def post_event(ev: ClientEvent):
    data = {k: v for k, v in list(ev.data.items())[:20]}          # bound the payload
    return bus.publish(ev.category, ev.level, ev.message, data, source="client")


@router.get("/api/events/meta")
def events_meta():
    return {"levels": list(LEVELS), "categories": list(CATEGORIES), "last_seq": bus.last_seq}


@router.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    """Sends the last `backlog` events, then streams new ones as they happen."""
    await ws.accept()
    try:
        backlog = int(ws.query_params.get("backlog", "200"))
    except ValueError:
        backlog = 200
    q = bus.subscribe()
    try:
        for ev in bus.since(0, limit=max(0, min(backlog, 2000))):
            await ws.send_json(ev)
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=20)
                await ws.send_json(ev)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping", "last_seq": bus.last_seq})   # keep-alive
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        bus.unsubscribe(q)
