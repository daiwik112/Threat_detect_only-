"""
grid/flow.py — AC power flow runner and result extractor.

Public API
----------
run_pf(net) -> dict
    Runs Newton-Raphson AC power flow.
    Returns a flat dict of key electrical quantities.

Raises
------
PFNonConvergence : if pandapower cannot achieve convergence.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandapower as pp


class PFNonConvergence(RuntimeError):
    """Raised when the AC power flow does not converge."""


def run_pf(net: pp.pandapowerNet) -> dict[str, Any]:
    """
    Run Newton-Raphson AC power flow on `net`.

    Returns
    -------
    dict with keys:
        vm_pu          : {bus_idx: vm_pu}
        va_deg         : {bus_idx: va_deg}
        line_p_from_mw : {line_idx: p_from_mw}
        line_p_to_mw   : {line_idx: p_to_mw}
        line_q_from_mvar: {line_idx: q_from_mvar}
        line_q_to_mvar : {line_idx: q_to_mvar}
        line_loading_pct: {line_idx: loading_%}
        line_pl_mw     : {line_idx: real power loss in segment}
        gen_p_mw       : total generation MW
        load_p_mw      : total accounted load MW
        total_loss_mw  : total network losses MW
        min_vm_pu      : float
        max_vm_pu      : float
        max_line_loading_pct : float
        converged      : bool
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.runpp(
                net,
                algorithm="nr",
                init="dc",
                max_iteration=50,
                tolerance_mva=1e-8,
                numba=False,   # avoid Numba JIT cold-start issues in testing
            )
    except pp.powerflow.LoadflowNotConverged as exc:
        raise PFNonConvergence(str(exc)) from exc

    if not net.converged:
        raise PFNonConvergence("Power flow did not converge (net.converged is False).")

    # --- bus results ---
    vm_pu  = net.res_bus["vm_pu"].to_dict()
    va_deg = net.res_bus["va_degree"].to_dict()

    # --- line results ---
    line_p_from   = net.res_line["p_from_mw"].to_dict()
    line_p_to     = net.res_line["p_to_mw"].to_dict()
    line_q_from   = net.res_line["q_from_mvar"].to_dict()
    line_q_to     = net.res_line["q_to_mvar"].to_dict()
    line_loading  = net.res_line["loading_percent"].to_dict()
    line_pl       = net.res_line["pl_mw"].to_dict()    # segment real power loss

    # --- totals ---
    gen_p_mw  = float(net.res_ext_grid["p_mw"].sum())
    load_p_mw = float(net.res_load["p_mw"].sum())
    total_loss_mw = float(net.res_line["pl_mw"].sum())

    return {
        "vm_pu": vm_pu,
        "va_deg": va_deg,
        "line_p_from_mw": line_p_from,
        "line_p_to_mw": line_p_to,
        "line_q_from_mvar": line_q_from,
        "line_q_to_mvar": line_q_to,
        "line_loading_pct": line_loading,
        "line_pl_mw": line_pl,
        "gen_p_mw": gen_p_mw,
        "load_p_mw": load_p_mw,
        "total_loss_mw": total_loss_mw,
        "min_vm_pu": float(net.res_bus["vm_pu"].min()),
        "max_vm_pu": float(net.res_bus["vm_pu"].max()),
        "max_line_loading_pct": float(net.res_line["loading_percent"].max()),
        "converged": True,
    }
