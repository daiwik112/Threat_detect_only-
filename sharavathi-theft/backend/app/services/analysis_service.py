"""
backend/app/services/analysis_service.py — detect / localize / analyze + dataset scenarios.

All ML calls go through ml.infer.ModelBundle (features recomputed from raw measurements).
Wording: "suspected unauthorized extraction", never "theft confirmed" (04 §Requirements 4).
"""

from __future__ import annotations

import json
import math
import time
import uuid
from functools import lru_cache

import numpy as np
import pandas as pd

from app.deps import DISCLAIMER, APIError, require_models, settings, state
from app.events import bus


def _meas(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if not k.startswith("m_"):
            continue
        out[k] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    if not out:
        raise APIError(422, "no_measurements", "measurements must contain at least one m_* key")
    return out


def detect(measurements: dict, explain: bool = True) -> dict:
    b = require_models("xgb_classifier")
    r = b.detect(_meas(measurements), explain=explain)
    r["threshold_z"] = float(b.baseline_detector.z_threshold) if b.baseline_detector else None
    return r


def localize(measurements: dict, method: str | None = None) -> dict:
    b = require_models("localizer_reg", "localizer_rank")
    if method == "gnn" and b.gnn is None:
        raise APIError(503, "gnn_unavailable",
                       "GNN model not loaded (data/models/gnn.pt missing or torch not installed).")
    return b.localize(_meas(measurements), method=method)


_NAMES = {"SHARAVATHI": "Sharavathi", "SHIMOGA": "Shimoga", "SIRA_KBCROSS": "Sira/KB-Cross",
          "NELAMANGALA": "Nelamangala", "PEENYA": "Peenya"}


def _sec(name: str) -> str:
    return " → ".join(_NAMES.get(b, b) for b in name.split("-"))


def recommendations(det: dict, loc: dict) -> list[str]:
    """Rule-based inspection guidance from section + confidence."""
    cause = det["likely_cause"]
    if det["status"] == "normal":
        return ["No anomaly above the alarm threshold. Continue routine monitoring."]
    out = []
    main = max(cause, key=cause.get)
    lo, hi = loc["interval_km"]
    sec = loc["ranked_sections"][0] if loc["ranked_sections"] else None
    tap = loc["ranked_taps"][0] if loc["ranked_taps"] else None
    if main == "extraction":
        conf = "high" if cause["extraction"] > 0.8 and det["anomaly_score"] > 0.9 else "moderate"
        out.append(f"Suspected unauthorized extraction ({conf} confidence, "
                   f"{cause['extraction'] * 100:.0f}% of anomaly evidence).")
        if sec:
            out.append(f"Prioritise line patrol on section {_sec(sec['section'])} "
                       f"({sec['prob'] * 100:.0f}% probability).")
        out.append(f"Focus the patrol on km {lo:.0f}–{hi:.0f} (80% interval); "
                   f"estimated location km {loc['est_km']:.0f}"
                   + (f", most likely tap {tap['tap_id']}." if tap else "."))
        out.append("Before dispatch, cross-check the metering at both section ends "
                   "to rule out a sensor fault.")
    elif main == "sensor_error":
        feats = [c["feature"] for c in det["top_contributors"]
                 if c["feature"].startswith(("f_kcl", "f_vm_dev", "f_va_dev", "f_load_scale_spread",
                                             "f_n_missing"))]
        out.append(f"Anomaly most consistent with a metering fault "
                   f"({cause['sensor_error'] * 100:.0f}% of anomaly evidence).")
        out.append("Inspect / recalibrate the meters implicated by: "
                   + (", ".join(feats[:3]) if feats else "the top contributing features") + ".")
        out.append("Do not dispatch a line patrol until the metering has been verified.")
    else:
        out.append(f"Anomaly most consistent with a network event such as a line outage "
                   f"({cause['other'] * 100:.0f}% of anomaly evidence).")
        out.append("Check breaker status / SCADA event log for the affected sections.")
    return out


def analyze(req, explain: bool = True) -> dict:
    """explain=False (batch sweeps) skips SHAP and the rule-based recommendations."""
    from app.services.simulation_service import simulate
    t0 = time.perf_counter()
    sim = None
    if req.scenario_id:
        d = scenario_detail(req.scenario_id)
        meas = d["measurements"]
        sim = {"scenario_id": d["scenario_id"], "measurements": meas, "expected": {},
               "true_label": d["label"], "true_km": d["true_km"], "true_tap_id": d["true_tap_id"],
               "true_magnitude_mw": d["magnitude_mw"], "sensor_profile": "major_only",
               "warnings": ["Dataset scenario: expected values not available."],
               "disclaimer": DISCLAIMER}
    elif req.measurements:
        meas = req.measurements
    else:
        sim = simulate(req)
        meas = sim["measurements"]
    det = detect(meas, explain=explain)
    loc = localize(meas, req.localize_method)
    if explain:                                    # batch items are summarised once by batch_analyze
        _emit_analysis(req, sim, det, loc)
    return {
        "request_id": uuid.uuid4().hex[:12],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "simulate": sim, "detect": det, "localize": loc,
        "recommendations": recommendations(det, loc) if explain else [],
        "disclaimer": DISCLAIMER,
    }


def _emit_analysis(req, sim: dict | None, det: dict, loc: dict) -> None:
    """Console events: what was simulated (the user's own input) and what the models concluded."""
    if req.scenario_id:
        bus.publish("simulation", "info", f"Replaying dataset scenario {req.scenario_id[:8]}",
                    {"scenario_id": req.scenario_id})
    elif sim is not None:
        parts = [f"load ×{req.load_scale:.2f}", req.sensor_profile, f"noise ×{req.noise_mult:g}", f"seed {req.seed}"]
        if req.theft and sim.get("true_tap_id"):
            what = f"Injected {sim['true_magnitude_mw']:.1f} MW unmetered load at km {sim['true_km']:g} ({sim['true_tap_id']})"
        elif req.fault:
            what = f"Injected fault: {req.fault.type}" + (f" on {req.fault.channel}" if req.fault.channel else "")
        else:
            what = "Simulated normal operation"
        bus.publish("simulation", "info", f"{what} — {', '.join(parts)}",
                    {"theft_km": sim.get("true_km"), "magnitude_mw": sim.get("true_magnitude_mw"),
                     "fault": req.fault.type if req.fault else None, "label": sim.get("true_label")})
        for w in sim.get("warnings", []):
            bus.publish("simulation", "warn", w)
    else:
        bus.publish("simulation", "info", "Analysing posted measurements", {"n_channels": len(req.measurements or {})})

    lc = det["likely_cause"]
    cause = max(lc, key=lc.get)
    score = det["anomaly_score"]
    if det["status"] != "suspicious":
        bus.publish("detection", "info", f"Normal — anomaly score {score:.3f}", {"score": score})
        return
    lo, hi = loc["interval_km"]
    top = loc["ranked_sections"][0]["section"] if loc.get("ranked_sections") else None
    if cause == "extraction":
        bus.publish("detection", "alert",
                    f"SUSPECTED UNAUTHORIZED EXTRACTION near km {loc['est_km']:.0f} "
                    f"(80% interval {lo:.0f}–{hi:.0f} km) — score {score:.3f}, extraction {lc['extraction']:.0%}",
                    {"km": loc["est_km"], "interval_km": [lo, hi], "score": score, "section": top,
                     "likely_cause": lc, "truth_km": sim.get("true_km") if sim else None})
        if top:
            bus.publish("localization", "info", f"Most likely section {_sec(top)}; top tap "
                        f"{loc['ranked_taps'][0]['tap_id'] if loc.get('ranked_taps') else '—'} ({loc['method']})",
                        {"section": top, "method": loc["method"]})
    else:
        label = "metering / sensor fault" if cause == "sensor_error" else "network event (e.g. line outage)"
        bus.publish("detection", "warn", f"Anomaly (score {score:.3f}) most consistent with a {label} "
                    f"({lc[cause]:.0%})", {"score": score, "likely_cause": lc})


BATCH_MAX = 50


def batch_analyze(reqs: list, explain: bool = False) -> dict:
    """Run up to 50 analyses (sweeps / profile comparison). A failing item (e.g. PF not
    converged) returns an error entry instead of failing the whole batch."""
    if len(reqs) > BATCH_MAX:
        raise APIError(422, "batch_too_large", f"at most {BATCH_MAX} requests per batch (got {len(reqs)})")
    t0 = time.perf_counter()
    results = []
    for i, r in enumerate(reqs):
        try:
            results.append({"index": i, "result": analyze(r, explain=explain), "error": None})
        except APIError as e:
            if e.status == 503:                      # models missing: the whole batch is moot
                raise
            results.append({"index": i, "result": None, "error": {"code": e.code, "message": e.message}})
    ms = round((time.perf_counter() - t0) * 1000, 1)
    n_ok = sum(r["error"] is None for r in results)
    n_susp = sum(1 for r in results if r["result"] and r["result"]["detect"]["status"] == "suspicious")
    bus.publish("batch", "info", f"Batch of {len(results)} analyses: {n_ok} ok, {len(results) - n_ok} failed, "
                f"{n_susp} suspicious ({ms / 1000:.1f} s)", {"n": len(results), "n_ok": n_ok, "n_suspicious": n_susp})
    return {"request_id": uuid.uuid4().hex[:12], "latency_ms": ms, "n_ok": n_ok, "results": results}


SAMPLE_REQUESTS = {
    "theft": ("Big theft near Shimoga", {"load_scale": 1.0, "theft": {"km": 150, "magnitude_mw": 25}}),
    "small_theft": ("Small theft (near the noise floor)", {"load_scale": 0.9, "theft": {"km": 260, "magnitude_mw": 3}}),
    "normal": ("Normal operation", {"load_scale": 1.0}),
    "sensor_fault": ("Sensor bias only", {"load_scale": 1.0, "fault": {"type": "bias", "channel": "m_SHIMOGA_p", "param": 8}}),
    "line_outage": ("Line outage", {"load_scale": 0.8, "fault": {"type": "line_outage", "param": 20}}),
}


def sample_requests() -> list[dict]:
    base = {"theft": None, "fault": None, "sensor_profile": "major_only", "noise_mult": 1.0, "seed": 42}
    return [{"kind": k, "label": lbl, "request": {**base, **req}} for k, (lbl, req) in SAMPLE_REQUESTS.items()]


def sample_request(kind: str) -> dict:
    for s in sample_requests():
        if s["kind"] == kind:
            return s
    raise APIError(404, "unknown_kind", f"kind must be one of {sorted(SAMPLE_REQUESTS)}")


@lru_cache(maxsize=1)
def _case_index():
    from vectordb.cases import CaseIndex
    try:
        return CaseIndex.load_or_build()
    except Exception as e:
        raise APIError(503, "case_index_unavailable", f"similar-case index unavailable: {e}") from e


def similar_cases(measurements: dict, k: int = 5, label: str | None = None) -> dict:
    idx = _case_index()
    t0 = time.perf_counter()
    items = idx.similar(_meas(measurements), k=k, label=label)
    if items:
        labels = [i["label"] for i in items]
        bus.publish("detection", "info", f"Vector DB: {k} nearest past cases — "
                    + ", ".join(f"{labels.count(l)}× {l}" for l in dict.fromkeys(labels)),
                    {"top_similarity": items[0]["similarity"]})
    return {"items": items, "n_indexed": len(idx.store), "dim": idx.store.dim,
            "embedding": "standardized physics features (ml.features), train+val scenarios",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


# ---------------------------------------------------------------- dataset scenarios
@lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    if not settings.dataset_path.exists():
        raise APIError(503, "dataset_missing", "dataset_major_only.parquet not found — run `make data`.")
    df = pd.read_parquet(settings.dataset_path)
    df["split"] = None
    if settings.splits_path.exists():
        for k, idx in json.loads(settings.splits_path.read_text()).items():
            df.iloc[idx, df.columns.get_loc("split")] = k
    return df.set_index("scenario_id", drop=False)


@lru_cache(maxsize=1)
def _metered_cols() -> list[str]:
    from ml.features import get_grid_meta, metered_channels
    return [c for c in metered_channels(get_grid_meta("major_only")) if c in _dataset().columns]


def _summary(r) -> dict:
    return {"scenario_id": r["scenario_id"], "label": r["label"], "split": r["split"],
            "true_km": None if pd.isna(r["true_km"]) else float(r["true_km"]),
            "true_tap_id": r["true_tap_id"] if isinstance(r["true_tap_id"], str) else None,
            "magnitude_mw": float(r["magnitude_mw"]), "load_scale": float(r["load_scale"])}


def list_scenarios(label: str | None, tap: str | None, split: str | None,
                   limit: int, offset: int) -> dict:
    df = _dataset()
    if label:
        df = df[df["label"] == label]
    if tap:
        df = df[df["true_tap_id"] == tap]
    if split:
        df = df[df["split"] == split]
    page = df.iloc[offset: offset + limit]
    return {"total": int(len(df)), "items": [_summary(r) for _, r in page.iterrows()]}


def scenario_detail(scenario_id: str) -> dict:
    df = _dataset()
    if scenario_id not in df.index:
        raise APIError(404, "scenario_not_found", f"scenario {scenario_id!r} not in dataset")
    r = df.loc[scenario_id]
    meas = {c: (None if pd.isna(r[c]) else float(r[c])) for c in _metered_cols()}
    return {**_summary(r), "measurements": meas}


@lru_cache(maxsize=1)
def replay_sequence(n: int = 120, seed: int = 7) -> list[dict]:
    """Held-out test scenarios in a fixed mixed order (≈ half normal) for /ws/replay."""
    df = _dataset()
    test = df[df["split"] == "test"]
    rng = np.random.default_rng(seed)
    norm = test[test["label"] == "normal"].sample(n // 2, random_state=seed)
    rest = test[test["label"] != "normal"].sample(n - n // 2, random_state=seed)
    seq = pd.concat([norm, rest])
    return [_summary(r) | {"measurements": {c: (None if pd.isna(r[c]) else float(r[c]))
                                            for c in _metered_cols()}}
            for _, r in seq.iloc[rng.permutation(len(seq))].iterrows()]
