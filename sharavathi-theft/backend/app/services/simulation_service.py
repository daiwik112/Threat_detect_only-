"""
backend/app/services/simulation_service.py — network description + on-demand simulation.

simulate(): network (theft = real pp.create_load at the snapped tap bus, per 00 §6.2)
→ optional fault / line outage → AC PF → noisy metering (grid.sensors). `expected` comes
from a noiseless, theft-free, fault-free twin at the same load scale. Only channels the
chosen profile physically meters are returned.

Speed: grid.build.build_net costs ~0.5 s (element creation), so a template net is built
once and deep-copied per request; load scaling and the theft load are applied exactly as
build_net does. tests/test_api.py checks the result is identical to simdata's simulate_one.
"""

from __future__ import annotations

import hashlib
import uuid
import warnings
from functools import lru_cache

import numpy as np

from app.deps import DISCLAIMER, APIError

LOAD_RANGE = (0.5, 1.15)
KM_RANGE = (0.0, 361.0)
MW_RANGE = (0.0, 60.0)
DEFAULT_FAULT_CHANNEL = {"bias": "m_SHIMOGA_p", "drift": "m_NELAMANGALA_q", "missing": "m_PEENYA_vm"}


@lru_cache(maxsize=1)
def _configs():
    from grid.config import load_sensors, load_topology
    return load_topology(), load_sensors()


@lru_cache(maxsize=1)
def _template():
    from grid.build import build_net
    topo, _ = _configs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_net(topology=topo, load_scale=1.0)


def _net_for(spec):
    """Equivalent of build_net(topology, spec.load_scale, theft) + outage, from the template."""
    import copy
    import pandapower as pp
    from grid.build import TheftSpec, _snap_to_tap
    topo, _ = _configs()
    net = copy.deepcopy(_template())
    for lc in topo.loads:                                 # build_net §5: loads × load_scale
        i = net.load.index[net.load["name"] == f"LOAD_{lc.bus}"][0]
        net.load.at[i, "p_mw"] = lc.p_mw * spec.load_scale
        net.load.at[i, "q_mvar"] = lc.q_mvar * spec.load_scale
    net["theft_km"] = net["theft_bus_idx"] = None
    if spec.theft_km is not None:                         # build_net §6: real pandapower load
        th = TheftSpec(km=spec.theft_km, mw=spec.theft_mw, pf=spec.theft_pf)
        bus, km = _snap_to_tap(th.km, net["tap_map"], net["pos_to_idx"])
        pp.create_load(net, bus=bus, p_mw=th.mw, q_mvar=th.q_mvar, name="THEFT", in_service=True)
        net["theft_km"], net["theft_bus_idx"] = km, bus
    if spec.outage_line_idx is not None:
        net.line.at[spec.outage_line_idx, "in_service"] = False
    return net


def _simulate_spec(spec, sensors_cfg, seed: int) -> dict | None:
    """Same output as simdata.generate.simulate_one, using the cached template net."""
    from grid.flow import PFNonConvergence, run_pf
    from grid.sensors import FaultSpec as GFault, enrich_results_with_bus_power, measure
    rng = np.random.default_rng(seed)
    net = _net_for(spec)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = enrich_results_with_bus_power(run_pf(net), net)
    except PFNonConvergence:
        return None
    faults = []
    for ch in spec.fault_channels or []:
        if spec.fault_type == "bias":
            faults.append(GFault(ch, "bias", pct=spec.fault_pct))
        elif spec.fault_type == "drift":
            faults.append(GFault(ch, "drift", rate=spec.fault_rate, step=spec.fault_step))
        elif spec.fault_type == "missing":
            faults.append(GFault(ch, "missing", prob=1.0))
    names = {i: r["name"] for i, r in net.bus.iterrows()}
    m = measure(res, spec.sensor_profile, sensors_cfg, net["id_to_idx"], names, rng,
                faults=faults or None)
    tap = f"TAP_{int(net['theft_km'])}" if net.get("theft_km") is not None else None
    return {**{k: (np.nan if v is None else float(v)) for k, v in m.items()}, "true_tap_id": tap}


def _clean(d: dict) -> dict:
    return {k: (None if v is None or not np.isfinite(float(v)) else float(v)) for k, v in d.items()}


def build_network() -> dict:
    from grid.build import build_net
    from grid.flow import run_pf
    from ml.features import get_grid_meta

    topo, sensors = _configs()
    meta = get_grid_meta("major_only")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = run_pf(build_net(topology=topo, load_scale=1.0))
    names, km = meta.bus_names, meta.bus_km
    roles = {b.id: b.role for b in topo.buses}
    n = sensors.noise
    return {
        "buses": [{"id": b, "km": km[b], "role": roles[b], "metered": b in meta.metered}
                  for b in names if not b.startswith("TAP_")],
        "lines": [{"id": f"LINE{i}", "from_bus": names[i], "to_bus": names[i + 1],
                   "km_len": km[names[i + 1]] - km[names[i]],
                   "loading_pct": round(float(pf["line_loading_pct"][i]), 2)}
                  for i in range(len(names) - 1)],
        "taps": [{"id": b, "km": km[b], "line_id": f"LINE{names.index(b) - 1}"}
                 for b in names if b.startswith("TAP_")],
        "shunts": [{"bus": s.bus, "q_mvar_cap": s.q_mvar_cap} for s in topo.shunts],
        "sections": [{"id": s.name, "from_bus": s.from_bus, "to_bus": s.to_bus,
                      "km_from": s.km_from, "km_to": s.km_to} for s in meta.sections],
        "assumptions": [
            "Sharavathi = generation-side slack bus at km 0 (assumed).",
            "5-node chain is a proposed simplified model, NOT a verified KPTCL replica.",
            f"Line parameters r={topo.lines.r_ohm_per_km} Ω/km, x={topo.lines.x_ohm_per_km} Ω/km, "
            f"c={topo.lines.c_nf_per_km} nF/km, 2-circuit equivalent (source: assumed).",
            "Loads at scale 1.0: " + ", ".join(f"{l.bus} {l.p_mw:g} MW" for l in topo.loads) + " (assumed).",
            "Shunt capacitors always on: " + ", ".join(f"{s.bus} {s.q_mvar_cap:g} Mvar" for s in topo.shunts)
            + " (assumed).",
            "Ferranti effect: voltage exceeds 1.05 pu at light load (scale < 0.9); "
            "a real system would switch capacitors.",
            f"Virtual tap buses every {topo.tap_spacing_km:g} km are unmetered candidate extraction points.",
            f"Sensor noise (assumed): V {n.vm_pct}%, P {n.p_pct}%, Q {n.q_pct}%, angle {n.va_deg}°.",
            "ML models were trained on the major_only profile at noise ×1.",
        ],
        "disclaimer": DISCLAIMER,
    }


def _clamp(v: float, lo: float, hi: float, name: str, warns: list[str]) -> float:
    if v < lo or v > hi:
        c = float(min(max(v, lo), hi))
        warns.append(f"{name}={v:g} outside [{lo:g}, {hi:g}] — clamped to {c:g}")
        return c
    return float(v)


def simulate(req) -> dict:
    """req: SimulateRequest (pydantic). Returns SimulateResponse dict."""
    from ml.features import get_grid_meta, metered_channels
    from simdata.scenarios import ScenarioSpec

    topo, sensors = _configs()
    warns: list[str] = []
    ls = _clamp(req.load_scale, *LOAD_RANGE, "load_scale", warns)
    n = sensors.noise
    k = float(req.noise_mult)
    noisy = sensors.model_copy(update={"noise": n.model_copy(update={
        "vm_pct": n.vm_pct * k, "p_pct": n.p_pct * k, "q_pct": n.q_pct * k, "va_deg": n.va_deg * k})})
    exact = sensors.model_copy(update={"noise": n.model_copy(update={
        "vm_pct": 0.0, "p_pct": 0.0, "q_pct": 0.0, "va_deg": 0.0})})

    sid = str(uuid.UUID(hashlib.md5(req.model_dump_json().encode()).hexdigest()))
    base = dict(scenario_id=sid, group_id="api", load_scale=ls, bus_jitter={},
                sensor_profile=req.sensor_profile)
    kw: dict = {"label": "normal"}
    label = "normal"
    if req.fault is not None:
        f = req.fault
        if f.type == "line_outage":
            idx = int(f.param) if f.param is not None else (
                int(f.channel.replace("LINE", "")) if f.channel and f.channel.startswith("LINE") else 20)
            if not 0 <= idx < len(get_grid_meta("major_only").bus_names) - 1:
                raise APIError(422, "bad_fault", f"line index {idx} out of range 0–37")
            kw.update(outage_line_idx=idx)
            label = "line_outage"
        else:
            ch = f.channel or DEFAULT_FAULT_CHANNEL[f.type]
            valid = {c for c in metered_channels(get_grid_meta(req.sensor_profile)) if not c.startswith("m_LINE")}
            if ch not in valid:
                raise APIError(422, "bad_fault",
                               f"channel {ch!r} is not a metered bus channel for profile {req.sensor_profile}")
            kw.update(fault_type=f.type, fault_channels=[ch])
            if f.type == "bias":
                kw.update(fault_pct=float(f.param if f.param is not None else 5.0))
            elif f.type == "drift":
                kw.update(fault_rate=float(f.param if f.param is not None else 1.0), fault_step=6)
            label = {"bias": "sensor_bias", "drift": "sensor_drift", "missing": "missing"}[f.type]
    true_mw = 0.0
    if req.theft is not None:
        km = _clamp(req.theft.km, *KM_RANGE, "theft.km", warns)
        true_mw = _clamp(req.theft.magnitude_mw, *MW_RANGE, "theft.magnitude_mw", warns)
        if true_mw > 0:
            kw.update(theft_km=km, theft_mw=true_mw, theft_pf=req.theft.pf)
            label = "theft"
    kw["label"] = label

    row = _simulate_spec(ScenarioSpec(**base, **kw), noisy, req.seed)
    twin = _simulate_spec(ScenarioSpec(**{**base, "label": "normal"}), exact, req.seed)
    if row is None or twin is None:
        raise APIError(422, "pf_not_converged",
                       "AC power flow did not converge for this scenario (try a smaller theft "
                       "magnitude, a different load scale, or another outage line).")

    keep = metered_channels(get_grid_meta(req.sensor_profile))
    tap = row.get("true_tap_id") if label == "theft" else None
    true_km = float(tap.split("_")[1]) if tap else None        # snapped tap location
    if true_km is not None and abs(true_km - req.theft.km) > 1e-6:
        warns.append(f"theft.km snapped to nearest tap: {true_km:g} km")
    return {
        "scenario_id": sid,
        "measurements": _clean({c: row.get(c, np.nan) for c in keep}),
        "expected": _clean({c: twin.get(c, np.nan) for c in keep}),
        "true_label": label,
        "true_km": true_km,
        "true_tap_id": tap,
        "true_magnitude_mw": true_mw if label == "theft" else 0.0,
        "sensor_profile": req.sensor_profile,
        "warnings": warns,
        "disclaimer": DISCLAIMER,
    }
