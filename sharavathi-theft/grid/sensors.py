"""
grid/sensors.py — Noisy & faulty measurement simulation.

Public API
----------
measure(results, profile, noise_cfg, rng, faults=None) -> dict[str, float | None]
    Applies sensor profile (subset selection), Gaussian noise, and optional
    fault modes to the power-flow results dict.

Fault modes (any combination, per sensor key)
---------------------------------------------
  bias(pct)         : constant offset of +pct % on top of true value
  drift(rate, step) : incremental bias growing by rate*step each call
  missing(prob)     : NaN with given probability
  stuck(value)      : always returns given value regardless of true reading
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import numpy as np

from grid.config import SensorsConfig


# ---------------------------------------------------------------------------
# Fault specification
# ---------------------------------------------------------------------------

FaultType = Literal["bias", "drift", "missing", "stuck"]


class FaultSpec:
    def __init__(
        self,
        sensor_key: str,
        fault_type: FaultType,
        *,
        pct: float = 0.0,       # for bias
        rate: float = 0.0,      # for drift (% per step)
        step: int = 0,          # current step count (for drift)
        prob: float = 0.0,      # for missing
        value: float = 0.0,     # for stuck
    ) -> None:
        self.sensor_key = sensor_key
        self.fault_type = fault_type
        self.pct = pct
        self.rate = rate
        self.step = step
        self.prob = prob
        self.value = value


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------


def measure(
    results: dict[str, Any],
    profile: str,
    sensors_cfg: SensorsConfig,
    bus_id_to_idx: dict[str, int],
    net_bus_names: dict[int, str],   # idx -> name
    rng: np.random.Generator,
    faults: Optional[list[FaultSpec]] = None,
) -> dict[str, float | None]:
    """
    Generate a noisy measurement dict from power-flow results.

    Parameters
    ----------
    results      : output of grid.flow.run_pf()
    profile      : sensor profile name ("major_only", "major_plus_mid", "dense")
    sensors_cfg  : loaded SensorsConfig
    bus_id_to_idx: mapping bus name → pandapower bus index
    net_bus_names: mapping bus index → bus name
    rng          : numpy random Generator (for reproducibility)
    faults       : optional list of FaultSpec objects

    Returns
    -------
    dict keyed by  "m_<BUS>_vm", "m_<BUS>_va", "m_<BUS>_p", "m_<BUS>_q",
                   "m_LINE<idx>_p_from", "m_LINE<idx>_p_to",
                   "m_LINE<idx>_q_from", "m_LINE<idx>_q_to"
    Values are floats or NaN (missing measurement).
    """
    noise = sensors_cfg.noise
    all_bus_names = list(net_bus_names.values())
    metered_bus_names = sensors_cfg.resolve_profile(profile, all_bus_names)

    # Index lookup for quick fault application
    fault_map: dict[str, FaultSpec] = {}
    if faults:
        for f in faults:
            fault_map[f.sensor_key] = f

    measurements: dict[str, float | None] = {}

    # --- Bus measurements (vm, va, injected P and Q) ---
    for bus_name in metered_bus_names:
        bus_idx = bus_id_to_idx.get(bus_name)
        if bus_idx is None:
            continue

        true_vm = results["vm_pu"].get(bus_idx, float("nan"))
        true_va = results["va_deg"].get(bus_idx, float("nan"))

        # Injected power at the bus is not directly in run_pf results;
        # we use net.res_bus if available, but for simplicity we expose
        # signed generation − load convention:
        # (these will be filled by caller from net.res_bus if needed;
        #  sensors just pass them through with noise)
        true_p = results.get("bus_p_mw", {}).get(bus_idx, float("nan"))
        true_q = results.get("bus_q_mvar", {}).get(bus_idx, float("nan"))

        measurements[f"m_{bus_name}_vm"] = _noisy(
            true_vm, noise.vm_pct / 100.0 * true_vm, f"m_{bus_name}_vm", fault_map, rng
        )
        measurements[f"m_{bus_name}_va"] = _noisy(
            true_va, noise.va_deg, f"m_{bus_name}_va", fault_map, rng
        )
        measurements[f"m_{bus_name}_p"] = _noisy(
            true_p,
            abs(true_p) * noise.p_pct / 100.0 if not math.isnan(true_p) else 0,
            f"m_{bus_name}_p", fault_map, rng,
        )
        measurements[f"m_{bus_name}_q"] = _noisy(
            true_q,
            abs(true_q) * noise.q_pct / 100.0 if not math.isnan(true_q) else 0,
            f"m_{bus_name}_q", fault_map, rng,
        )

    # --- Line measurements (metered lines = those touching metered buses) ---
    metered_bus_idx_set = {
        bus_id_to_idx[bn] for bn in metered_bus_names if bn in bus_id_to_idx
    }
    for line_idx, p_from in results["line_p_from_mw"].items():
        p_to  = results["line_p_to_mw"][line_idx]
        q_from = results["line_q_from_mvar"][line_idx]
        q_to  = results["line_q_to_mvar"][line_idx]

        # Only emit measurement if either endpoint is a metered bus
        # (section_map gives us from_name, to_name)
        measurements[f"m_LINE{line_idx}_p_from"] = _noisy(
            p_from, abs(p_from) * noise.p_pct / 100.0,
            f"m_LINE{line_idx}_p_from", fault_map, rng,
        )
        measurements[f"m_LINE{line_idx}_p_to"] = _noisy(
            p_to, abs(p_to) * noise.p_pct / 100.0,
            f"m_LINE{line_idx}_p_to", fault_map, rng,
        )
        measurements[f"m_LINE{line_idx}_q_from"] = _noisy(
            q_from, abs(q_from) * noise.q_pct / 100.0,
            f"m_LINE{line_idx}_q_from", fault_map, rng,
        )
        measurements[f"m_LINE{line_idx}_q_to"] = _noisy(
            q_to, abs(q_to) * noise.q_pct / 100.0,
            f"m_LINE{line_idx}_q_to", fault_map, rng,
        )

    return measurements


# ---------------------------------------------------------------------------
# Noise application with fault modes
# ---------------------------------------------------------------------------


def _noisy(
    true_val: float,
    sigma: float,
    key: str,
    fault_map: dict[str, FaultSpec],
    rng: np.random.Generator,
) -> float | None:
    """Add Gaussian noise and apply any fault for `key`."""
    if math.isnan(true_val):
        return float("nan")

    noisy = true_val + rng.normal(0.0, max(sigma, 1e-12))

    fault = fault_map.get(key)
    if fault is None:
        return noisy

    if fault.fault_type == "bias":
        return noisy * (1.0 + fault.pct / 100.0)

    if fault.fault_type == "drift":
        accumulated = fault.rate * fault.step / 100.0
        return noisy * (1.0 + accumulated)

    if fault.fault_type == "missing":
        if rng.random() < fault.prob:
            return float("nan")
        return noisy

    if fault.fault_type == "stuck":
        return float(fault.value)

    return noisy


# ---------------------------------------------------------------------------
# Convenience: enrich results with bus-level P/Q from pandapower net
# ---------------------------------------------------------------------------


def enrich_results_with_bus_power(
    results: dict[str, Any],
    net,
) -> dict[str, Any]:
    """Add bus_p_mw and bus_q_mvar dicts to results from net.res_bus."""
    results["bus_p_mw"] = net.res_bus["p_mw"].to_dict()
    results["bus_q_mvar"] = net.res_bus["q_mvar"].to_dict()
    return results
