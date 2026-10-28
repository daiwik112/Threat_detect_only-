"""
backend/tests/test_api.py — WF4 API tests (httpx.AsyncClient over ASGI).

Run from backend/:  pytest tests -q
"""

from __future__ import annotations

import statistics
import time

import httpx
import numpy as np
import pytest
import pytest_asyncio

from app.deps import state
from app.main import app

pytestmark = pytest.mark.asyncio(loop_scope="module")
THEFT = {"theft": {"km": 150, "magnitude_mw": 20}, "sensor_profile": "major_only"}


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client():
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as c:
            yield c


def _err(r, status, code=None):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and {"code", "message"} <= set(body["error"])
    if code:
        assert body["error"]["code"] == code


# ---------------------------------------------------------------- meta / network
async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["dataset_hash"] and j["model_version"]
    assert r.headers["X-Request-ID"] and float(r.headers["X-Latency-ms"]) >= 0


async def test_features(client):
    j = (await client.get("/api/features")).json()
    assert j["mitigation_enabled"] is False and isinstance(j["gnn_enabled"], bool)


async def test_network(client):
    j = (await client.get("/api/network")).json()
    assert [b["id"] for b in j["buses"]] == ["SHARAVATHI", "SHIMOGA", "SIRA_KBCROSS",
                                             "NELAMANGALA", "PEENYA"]
    assert len(j["lines"]) == 38 and len(j["taps"]) == 34 and len(j["sections"]) == 3
    assert j["assumptions"] and "field verification" in j["disclaimer"]
    assert all(0 < l["loading_pct"] < 100 for l in j["lines"])


# ---------------------------------------------------------------- simulate
async def test_simulate_theft(client):
    j = (await client.post("/api/simulate", json=THEFT)).json()
    assert j["true_label"] == "theft" and j["true_km"] == 150 and j["true_tap_id"] == "TAP_150"
    assert set(j["measurements"]) == set(j["expected"])
    # only physically metered channels (no unmetered tap-segment flows)
    assert not any(k.startswith("m_LINE5_") for k in j["measurements"])
    assert "m_LINE0_p_from" in j["measurements"] and "m_LINE0_p_to" not in j["measurements"]


async def test_simulate_matches_dataset_generator():
    """Fast template path must reproduce simdata.generate.simulate_one exactly."""
    from app.services.simulation_service import _configs, _simulate_spec
    from simdata.generate import simulate_one
    from simdata.scenarios import ScenarioSpec
    topo, sensors = _configs()
    for kw in ({"label": "theft", "theft_km": 233, "theft_mw": 17.0, "theft_pf": 0.9},
               {"label": "line_outage", "outage_line_idx": 5},
               {"label": "sensor_bias", "fault_type": "bias", "fault_channels": ["m_SHIMOGA_p"],
                "fault_pct": 6.0}):
        spec = ScenarioSpec(scenario_id="x", group_id="g", load_scale=0.83, bus_jitter={}, **kw)
        a, b = _simulate_spec(spec, sensors, 11), simulate_one(spec, topo, sensors, 11)
        keys = [k for k in b if k.startswith("m_")]
        np.testing.assert_allclose([a[k] for k in keys], [b[k] for k in keys], equal_nan=True)
        assert a["true_tap_id"] == b["true_tap_id"]


async def test_simulate_clamps_and_snaps(client):
    # (60 MW at km 360 with load clamped to 1.15 does not converge → test separately)
    j = (await client.post("/api/simulate", json={"load_scale": 2.0})).json()
    assert "load_scale" in " ".join(j["warnings"])
    j = (await client.post("/api/simulate",
                           json={"load_scale": 0.6, "theft": {"km": 999, "magnitude_mw": 99}})).json()
    w = " ".join(j["warnings"])
    assert "theft.km" in w and "magnitude_mw" in w and "snapped" in w
    assert j["true_km"] == 360 and j["true_magnitude_mw"] == 60
    j = (await client.post("/api/simulate", json={"theft": {"km": 153, "magnitude_mw": 10}})).json()
    assert j["true_km"] == 150


async def test_simulate_latency(client):
    """04 req. 1: PF per request < 300 ms (two PFs per request: measured + expected twin)."""
    ts = []
    for s in range(5):
        t = time.perf_counter()
        await client.post("/api/simulate", json={**THEFT, "seed": s})
        ts.append((time.perf_counter() - t) * 1000)
    assert statistics.median(ts) < 300, ts


async def test_simulate_bad_input(client):
    _err(await client.post("/api/simulate", json={"load_scale": "abc"}), 422, "validation_error")
    _err(await client.post("/api/simulate", json={"sensor_profile": "nope"}), 422)
    _err(await client.post("/api/simulate", json={"fault": {"type": "bias", "channel": "m_FOO_p"}}),
         422, "bad_fault")
    _err(await client.post("/api/simulate", json={"fault": {"type": "line_outage", "param": 99}}),
         422, "bad_fault")


async def test_simulate_nonconvergence(client, monkeypatch):
    from grid import flow

    def boom(net):
        raise flow.PFNonConvergence("forced")
    monkeypatch.setattr(flow, "run_pf", boom)
    _err(await client.post("/api/simulate", json=THEFT), 422, "pf_not_converged")


# ---------------------------------------------------------------- detect / localize / analyze
async def test_detect_localize(client):
    meas = (await client.post("/api/simulate", json=THEFT)).json()["measurements"]
    d = (await client.post("/api/detect", json={"measurements": meas})).json()
    assert d["status"] in ("normal", "suspicious") and 0 <= d["anomaly_score"] <= 1
    assert {r["section"] for r in d["residuals"]} == {"SHARAVATHI-SHIMOGA", "SHIMOGA-NELAMANGALA",
                                                      "NELAMANGALA-PEENYA"}
    assert set(d["likely_cause"]) == {"extraction", "sensor_error", "other"}
    for m in ("xgb_reg", "section_rank"):
        l = (await client.post("/api/localize", json={"measurements": meas, "method": m})).json()
        assert l["method"] == m and l["interval_km"][0] <= l["interval_km"][1]
        assert l["ranked_taps"] and l["ranked_sections"]


async def test_detect_empty_measurements(client):
    _err(await client.post("/api/detect", json={"measurements": {}}), 422, "no_measurements")


async def test_analyze_acceptance(client):
    """04 acceptance: 150 km / 20 MW → suspicious, est_km within interval, ground truth returned."""
    j = (await client.post("/api/analyze", json=THEFT)).json()
    assert j["detect"]["status"] == "suspicious"
    lo, hi = j["localize"]["interval_km"]
    assert lo <= j["localize"]["est_km"] <= hi
    assert j["simulate"]["true_km"] == 150 and j["simulate"]["true_label"] == "theft"
    assert j["request_id"] and j["latency_ms"] > 0 and j["recommendations"]
    text = " ".join(j["recommendations"]).lower()
    assert "suspected" in text and "confirmed" not in text


async def test_analyze_normal_and_fault(client):
    j = (await client.post("/api/analyze", json={})).json()
    assert j["detect"]["status"] == "normal"
    j = (await client.post("/api/analyze", json={"fault": {"type": "bias", "param": 8}})).json()
    lc = j["detect"]["likely_cause"]
    assert lc["sensor_error"] > lc["extraction"]


async def test_analyze_from_measurements_and_scenario(client):
    meas = (await client.post("/api/simulate", json=THEFT)).json()["measurements"]
    j = (await client.post("/api/analyze", json={"measurements": meas})).json()
    assert j["simulate"] is None and j["detect"]["status"] == "suspicious"
    sid = (await client.get("/api/scenarios?label=theft&limit=1")).json()["items"][0]["scenario_id"]
    j = (await client.post("/api/analyze", json={"scenario_id": sid})).json()
    assert j["simulate"]["scenario_id"] == sid and j["simulate"]["true_label"] == "theft"


async def test_missing_models_503(client):
    saved = state.bundle
    state.bundle = None
    try:
        _err(await client.post("/api/detect", json={"measurements": {"m_PEENYA_p": 1.0}}),
             503, "models_missing")
        assert (await client.get("/api/health")).json()["status"] == "degraded"
    finally:
        state.bundle = saved


# ---------------------------------------------------------------- data endpoints
async def test_metrics(client):
    j = (await client.get("/api/metrics")).json()
    assert {"detection", "classification", "localization", "baseline"} <= set(j)


async def test_sensor_study_ready_or_404(client):
    r = await client.get("/api/sensor-study")
    if r.status_code == 404:
        _err(r, 404, "not_ready")
    else:
        assert "configs" in r.json()


async def test_scenarios(client):
    j = (await client.get("/api/scenarios?label=theft&tap=TAP_150&limit=3")).json()
    assert j["total"] > 0 and len(j["items"]) <= 3
    assert all(i["true_tap_id"] == "TAP_150" for i in j["items"])
    d = (await client.get(f"/api/scenarios/{j['items'][0]['scenario_id']}")).json()
    assert d["measurements"] and "m_LINE5_p_from" not in d["measurements"]
    _err(await client.get("/api/scenarios/does-not-exist"), 404, "scenario_not_found")
    _err(await client.get("/api/scenarios?limit=0"), 422)


async def test_dataset_card(client):
    assert "Dataset Card" in (await client.get("/api/dataset-card")).json()["markdown"]


# ---------------------------------------------------------------- mitigation stubs
async def test_mitigation(client):
    s = (await client.get("/api/mitigation/strategies")).json()
    assert {x["id"] for x in s["strategies"]} == {"load_balance", "redispatch", "shed"}
    assert not any(x["enabled"] for x in s["strategies"])
    r = await client.post("/api/mitigation/plan", json={"scenario_id": "x", "strategy": "shed"})
    assert r.status_code == 501 and r.json()["status"] == "not_implemented"


# ---------------------------------------------------------------- websocket
def test_ws_replay():
    from fastapi.testclient import TestClient
    with TestClient(app) as c, c.websocket_connect("/ws/replay") as ws:
        ws.send_json({"cmd": "speed", "value": 10})
        m = ws.receive_json()
        assert {"t", "scenario_id", "measurements", "detect", "localize", "truth"} <= set(m)
        assert ws.receive_json()["t"] == m["t"] + 1


def test_openapi_names_match_contract():
    names = set(app.openapi()["components"]["schemas"])
    assert {"NetworkResponse", "SimulateRequest", "SimulateResponse", "DetectResponse",
            "LocalizeResponse", "AnalyzeResponse", "MitigationRequest", "MitigationResponse"} <= names


# ---------------------------------------------------------------- 06: batch, samples, similar cases
async def test_batch_analyze_sweep(client):
    reqs = [{"theft": {"km": 150, "magnitude_mw": mw}, "seed": 1} for mw in (2, 10, 25)]
    reqs.append({"theft": {"km": 360, "magnitude_mw": 60}, "load_scale": 1.15})   # does not converge
    j = (await client.post("/api/batch-analyze", json={"requests": reqs})).json()
    assert len(j["results"]) == 4 and j["n_ok"] == 3 and j["request_id"] and j["latency_ms"] > 0
    ok = [r["result"] for r in j["results"] if r["result"]]
    assert ok[-1]["detect"]["status"] == "suspicious"
    assert ok[0]["detect"]["top_contributors"] == [] and ok[0]["recommendations"] == []   # explain=False
    assert j["results"][3]["error"]["code"] == "pf_not_converged"


async def test_batch_analyze_limits(client):
    _err(await client.post("/api/batch-analyze", json={"requests": [{}] * 51}), 422)
    _err(await client.post("/api/batch-analyze", json={"requests": []}), 422)


async def test_sample_requests(client):
    items = (await client.get("/api/sample-requests")).json()
    assert {i["kind"] for i in items} >= {"theft", "small_theft", "normal", "sensor_fault", "line_outage"}
    s = (await client.get("/api/sample-request?kind=sensor_fault")).json()
    assert s["request"]["fault"]["type"] == "bias"
    j = (await client.post("/api/analyze", json=s["request"])).json()
    assert j["detect"]["likely_cause"]["sensor_error"] > 0.5
    _err(await client.get("/api/sample-request?kind=nope"), 404, "unknown_kind")


async def test_similar_cases(client):
    meas = (await client.post("/api/simulate", json={"theft": {"km": 150, "magnitude_mw": 25}})).json()["measurements"]
    j = (await client.post("/api/similar-cases", json={"measurements": meas, "k": 5})).json()
    assert len(j["items"]) == 5 and j["n_indexed"] > 10000 and j["dim"] == 41
    sims = [i["similarity"] for i in j["items"]]
    assert sims == sorted(sims, reverse=True)
    assert sum(i["label"] == "theft" for i in j["items"]) >= 3
    j = (await client.post("/api/similar-cases", json={"measurements": meas, "k": 3, "label": "normal"})).json()
    assert all(i["label"] == "normal" for i in j["items"])


# ---------------------------------------------------------------- activity console (events)
async def test_events_record_simulation_and_detection(client):
    from app.events import bus
    start = bus.last_seq
    await client.post("/api/analyze", json=THEFT)
    evs = (await client.get(f"/api/events?since={start}")).json()["events"]
    cats = {(e["category"], e["level"]) for e in evs}
    assert ("simulation", "info") in cats and ("detection", "alert") in cats and ("api", "info") in cats
    alert = next(e for e in evs if e["level"] == "alert")
    assert "SUSPECTED UNAUTHORIZED EXTRACTION near km" in alert["message"] and alert["data"]["km"] > 0
    sim = next(e for e in evs if e["category"] == "simulation")
    assert "km 150" in sim["message"] and "confirmed" not in sim["message"].lower()
    # all events of one request share its request id
    api_ev = next(e for e in evs if e["category"] == "api" and e["data"]["path"] == "/api/analyze")
    assert alert["request_id"] == api_ev["request_id"]


async def test_events_filters_and_client_events(client):
    r = await client.post("/api/events", json={"message": "Clicked Run", "data": {"page": "lab"}})
    assert r.status_code == 201 and r.json()["source"] == "client" and r.json()["category"] == "user"
    _err(await client.post("/api/events", json={"message": ""}), 422)
    _err(await client.post("/api/events", json={"message": "x", "category": "detection"}), 422)  # clients can't forge
    j = (await client.get("/api/events?category=user&limit=5")).json()
    assert j["events"][-1]["message"] == "Clicked Run" and all(e["category"] == "user" for e in j["events"])
    assert all(e["level"] in ("alert", "error") for e in (await client.get("/api/events?min_level=alert")).json()["events"])


async def test_events_on_errors_and_mitigation(client):
    from app.events import bus
    start = bus.last_seq
    await client.post("/api/mitigation/plan", json={"scenario_id": "x", "strategy": "shed"})
    await client.get("/api/scenarios/nope")
    evs = (await client.get(f"/api/events?since={start}")).json()["events"]
    assert any(e["category"] == "mitigation" and "blocked" in e["message"] for e in evs)
    assert any(e["category"] == "system" and "scenario_not_found" in e["message"] for e in evs)


def test_ws_events_backlog_and_live():
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        c.post("/api/events", json={"message": "backlog marker"})
        with c.websocket_connect("/ws/events?backlog=50") as ws:
            seen = []
            for _ in range(60):
                m = ws.receive_json()
                seen.append(m.get("message"))
                if m.get("message") == "backlog marker":
                    break
            assert "backlog marker" in seen
            c.post("/api/events", json={"message": "live marker"})
            for _ in range(20):
                m = ws.receive_json()
                if m.get("message") == "live marker":
                    break
            assert m["message"] == "live marker" and m["source"] == "client"
