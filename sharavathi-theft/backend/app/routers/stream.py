"""
WS /ws/replay — pushes one held-out test scenario per tick with live detection.

Message: {"t", "scenario_id", "measurements", "detect", "localize",
          "truth": {"label", "km", "magnitude_mw"}}
Client commands: {"cmd": "pause"} | {"cmd": "play"} | {"cmd": "speed", "value": 2}
"""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from app.deps import APIError
from app.events import bus
from app.services import analysis_service as svc

router = APIRouter(tags=["stream"])


@router.websocket("/ws/replay")
async def replay(ws: WebSocket):
    await ws.accept()
    bus.publish("replay", "info", "Replay client connected")
    ctl = {"playing": True, "speed": 1.0}

    async def reader():
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (ValueError, TypeError):
                continue
            cmd = msg.get("cmd")
            if cmd == "pause":
                ctl["playing"] = False
                bus.publish("replay", "info", "Replay paused")
            elif cmd == "play":
                ctl["playing"] = True
                bus.publish("replay", "info", "Replay resumed")
            elif cmd == "speed":
                ctl["speed"] = float(min(max(float(msg.get("value", 1)), 0.25), 10))
                bus.publish("replay", "info", f"Replay speed ×{ctl['speed']:g}")

    task = asyncio.create_task(reader())
    try:
        seq = await run_in_threadpool(svc.replay_sequence)
        t = 0
        while not task.done():                       # reader ends when the client disconnects
            if not ctl["playing"]:
                await asyncio.sleep(0.1)
                continue
            s = seq[t % len(seq)]
            det = await run_in_threadpool(svc.detect, s["measurements"])
            loc = await run_in_threadpool(svc.localize, s["measurements"])
            await ws.send_json({"t": t, "scenario_id": s["scenario_id"],
                                "measurements": s["measurements"], "detect": det, "localize": loc,
                                "truth": {"label": s["label"], "km": s["true_km"],
                                          "magnitude_mw": s["magnitude_mw"]}})
            if det["status"] == "suspicious":
                truth = s["label"] + (f" @ km {s['true_km']:g}" if s["true_km"] is not None else "")
                bus.publish("replay", "alert" if max(det["likely_cause"], key=det["likely_cause"].get) == "extraction" else "warn",
                            f"Replay tick {t}: ALERT score {det['anomaly_score']:.2f}, est. km {loc['est_km']:.0f} "
                            f"(truth: {truth})", {"tick": t, "km": loc["est_km"], "truth_label": s["label"]})
            t += 1
            await asyncio.sleep(1.0 / ctl["speed"])
    except APIError as e:
        await ws.send_json({"error": {"code": e.code, "message": e.message}})
        await ws.close(code=1011)
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        bus.publish("replay", "info", "Replay client disconnected")
