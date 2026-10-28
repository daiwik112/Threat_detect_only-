"""
grid/estimator.py — Weighted Least Squares (WLS) state estimation.

Public API
----------
estimate(net, measurements, sensors_cfg) -> EstimationResult | None
    Runs pandapower WLS state estimation using the provided measurements.
    Returns chi-square J-value, normalized residuals, and estimated bus states.
    Returns None (and flags it) if the network is unobservable.

EstimationResult
    .converged    : bool
    .chi2_j       : float (chi-square objective J = r^T W r)
    .norm_residuals : dict[str, float]  (normalized residual per measurement)
    .vm_pu        : dict[int, float]    (estimated voltage magnitudes)
    .va_deg       : dict[int, float]    (estimated voltage angles)
    .observable   : bool
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandapower as pp
import pandapower.estimation as ppe

from grid.config import SensorsConfig


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EstimationResult:
    """Holds WLS state estimation outputs."""
    converged: bool
    observable: bool
    chi2_j: float                           # chi-square objective value
    norm_residuals: dict[str, float]        # key -> |normalised residual|
    vm_pu: dict[int, float]                 # bus_idx -> estimated vm_pu
    va_deg: dict[int, float]                # bus_idx -> estimated va_deg
    # Raw residuals (est - measured for each measurement added)
    raw_residuals: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Measurement weight helpers
# ---------------------------------------------------------------------------


def _sigma_vm(true_vm: float, vm_pct: float) -> float:
    """Standard deviation for voltage measurement (proportional to reading)."""
    return max(true_vm * vm_pct / 100.0, 1e-6)


def _sigma_p(p_mw: float, p_pct: float) -> float:
    """Standard deviation for active power measurement."""
    return max(abs(p_mw) * p_pct / 100.0, 1e-4)


def _sigma_q(q_mvar: float, q_pct: float) -> float:
    """Standard deviation for reactive power measurement."""
    return max(abs(q_mvar) * q_pct / 100.0, 1e-4)


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------


def estimate(
    net: pp.pandapowerNet,
    measurements: dict[str, float | None],
    sensors_cfg: SensorsConfig,
    *,
    algorithm: str = "wls",
    tolerance: float = 1e-6,
    max_iter: int = 30,
) -> Optional[EstimationResult]:
    """
    Run WLS state estimation on `net` using noisy measurements.

    Parameters
    ----------
    net          : pandapowerNet (must have already run power flow so
                   section_map / id_to_idx attributes are populated)
    measurements : dict from grid.sensors.measure() -- keyed as
                     "m_<BUS>_vm", "m_<BUS>_va", "m_<BUS>_p", "m_<BUS>_q",
                     "m_LINE<idx>_p_from", "m_LINE<idx>_p_to", etc.
                   NaN values are treated as missing (not injected).
    sensors_cfg  : SensorsConfig (for noise sigmas)
    algorithm    : estimation algorithm ('wls', 'wls_with_zero_injection')
    tolerance    : convergence tolerance
    max_iter     : maximum iterations

    Returns
    -------
    EstimationResult or None if unobservable.
    """
    noise = sensors_cfg.noise

    # Clear any previously added measurements
    net.measurement = net.measurement.iloc[0:0]

    id_to_idx: dict[str, int] = net.get("id_to_idx", {})

    # --- inject bus measurements ---
    for key, val in measurements.items():
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue  # skip missing

        parts = key.split("_")
        # key format: m_<BUSNAME>_vm / va / p / q
        if key.startswith("m_") and not key.startswith("m_LINE"):
            # parse: m_{BUS}_{type}  -- BUS can contain underscores
            suffix = parts[-1]                   # "vm", "va", "p", "q"
            bus_name = "_".join(parts[1:-1])     # rejoin interior parts
            bus_idx = id_to_idx.get(bus_name)
            if bus_idx is None:
                continue

            if suffix == "vm":
                sigma = _sigma_vm(val, noise.vm_pct)
                pp.create_measurement(net, "v", "bus", val, sigma, element=bus_idx)
            elif suffix == "p":
                sigma = _sigma_p(val, noise.p_pct)
                pp.create_measurement(net, "p", "bus", val, sigma, element=bus_idx)
            elif suffix == "q":
                sigma = _sigma_q(val, noise.q_pct)
                pp.create_measurement(net, "q", "bus", val, sigma, element=bus_idx)
            # va (angle) -- skip; pandapower WLS uses it only if PMU

        # --- line measurements ---
        elif key.startswith("m_LINE"):
            # format: m_LINE<idx>_p_from / p_to / q_from / q_to
            try:
                after_line = key[len("m_LINE"):]   # e.g. "3_p_from"
                idx_str, *rest = after_line.split("_")
                line_idx = int(idx_str)
                meas_type = "_".join(rest)           # "p_from", "q_to", ...
            except (ValueError, IndexError):
                continue

            if line_idx not in net.line.index:
                continue

            if meas_type == "p_from":
                sigma = _sigma_p(val, noise.p_pct)
                pp.create_measurement(net, "p", "line", val, sigma,
                                      element=line_idx, side="from")
            elif meas_type == "p_to":
                sigma = _sigma_p(val, noise.p_pct)
                pp.create_measurement(net, "p", "line", val, sigma,
                                      element=line_idx, side="to")
            elif meas_type == "q_from":
                sigma = _sigma_q(val, noise.q_pct)
                pp.create_measurement(net, "q", "line", val, sigma,
                                      element=line_idx, side="from")
            elif meas_type == "q_to":
                sigma = _sigma_q(val, noise.q_pct)
                pp.create_measurement(net, "q", "line", val, sigma,
                                      element=line_idx, side="to")

    # --- check minimum observability (need at least 1 voltage meas) ---
    if len(net.measurement) == 0:
        return EstimationResult(
            converged=False,
            observable=False,
            chi2_j=float("nan"),
            norm_residuals={},
            vm_pu={},
            va_deg={},
        )

    # --- run WLS ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converged = ppe.estimate(
                net,
                algorithm=algorithm,
                tolerance=tolerance,
                maximum_iterations=max_iter,
                zero_injection=None,
            )
    except Exception as exc:
        # Unobservable or numerical failure
        return EstimationResult(
            converged=False,
            observable=False,
            chi2_j=float("nan"),
            norm_residuals={"_error": float("nan")},
            vm_pu={},
            va_deg={},
        )

    if not converged:
        return EstimationResult(
            converged=False,
            observable=True,
            chi2_j=float("nan"),
            norm_residuals={},
            vm_pu={},
            va_deg={},
        )

    # --- extract results ---
    vm_pu  = net.res_bus_est["vm_pu"].to_dict()
    va_deg = net.res_bus_est["va_degree"].to_dict()

    # Compute chi-square J and normalised residuals from measurement table
    chi2_j = 0.0
    norm_residuals: dict[str, float] = {}
    raw_residuals: dict[str, float] = {}

    meas_df = net.measurement
    if hasattr(net, "res_bus_est") and len(meas_df) > 0:
        for i, row in meas_df.iterrows():
            meas_val     = row["value"]
            sigma        = row["std_dev"]
            meas_element = row["element"]
            meas_type    = row["measurement_type"]
            meas_side    = row.get("side", None)

            try:
                if row["element_type"] == "bus":
                    if meas_type == "v":
                        est_val = net.res_bus_est.at[meas_element, "vm_pu"]
                    elif meas_type == "p":
                        est_val = net.res_bus_est.at[meas_element, "p_mw"]
                    elif meas_type == "q":
                        est_val = net.res_bus_est.at[meas_element, "q_mvar"]
                    else:
                        continue
                elif row["element_type"] == "line":
                    if meas_type == "p":
                        col = "p_from_mw" if meas_side == "from" else "p_to_mw"
                        est_val = net.res_line_est.at[meas_element, col]
                    elif meas_type == "q":
                        col = "q_from_mvar" if meas_side == "from" else "q_to_mvar"
                        est_val = net.res_line_est.at[meas_element, col]
                    else:
                        continue
                else:
                    continue
            except (KeyError, AttributeError):
                continue

            residual = est_val - meas_val
            norm_r   = residual / max(sigma, 1e-12)
            raw_residuals[f"m{i}"]  = float(residual)
            norm_residuals[f"m{i}"] = float(abs(norm_r))
            chi2_j += norm_r ** 2

    return EstimationResult(
        converged=True,
        observable=True,
        chi2_j=chi2_j,
        norm_residuals=norm_residuals,
        raw_residuals=raw_residuals,
        vm_pu=vm_pu,
        va_deg=va_deg,
    )
