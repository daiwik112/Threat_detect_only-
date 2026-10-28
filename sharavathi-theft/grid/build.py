"""
grid/build.py — Build the Sharavathi 220 kV pandapower network.

Public API
----------
build_net(topology, load_scale=1.0, theft=None) -> pp.pandapowerNet
    Returns a fully-constructed network with:
      net.tap_map     : dict  tap_bus_idx -> km
      net.section_map : dict  line_idx    -> (from_km, to_km, from_bus_id, to_bus_id)

CLI
---
python -m grid.build [--check] [--theft-km 150 --theft-mw 20 --theft-pf 0.9]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandapower as pp

from grid.config import TopologyConfig, load_topology


# ---------------------------------------------------------------------------
# Theft specification dataclass
# ---------------------------------------------------------------------------


@dataclass
class TheftSpec:
    """Describe an unmetered extraction to inject into the network."""
    km: float          # km along corridor (snapped to nearest tap)
    mw: float          # extraction magnitude
    pf: float = 0.9   # power factor (lagging)

    @property
    def q_mvar(self) -> float:
        return self.mw * math.tan(math.acos(self.pf))


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_net(
    topology: TopologyConfig | None = None,
    load_scale: float = 1.0,
    theft: Optional[TheftSpec] = None,
) -> pp.pandapowerNet:
    """
    Construct and return a pandapower network for the Sharavathi corridor.

    Parameters
    ----------
    topology   : TopologyConfig (loaded from config/topology.yaml if None)
    load_scale : multiplier applied to all accounted loads (0.5 – 1.15)
    theft      : TheftSpec or None; if set, injects a load at the nearest tap bus

    Returns
    -------
    net        : pp.pandapowerNet with extra attributes:
                   net.tap_map     = {bus_index: km}
                   net.section_map = {line_index: (from_km, to_km, from_bus_id, to_bus_id)}
    """
    if topology is None:
        topology = load_topology()

    net = pp.create_empty_network(name="Sharavathi_220kV", f_hz=50.0, sn_mva=100.0)

    lp = topology.lines   # line parameters
    base_kv = topology.base_kv
    tap_spacing = topology.tap_spacing_km

    # -----------------------------------------------------------------------
    # 1. Collect all bus km positions (substations + taps)
    # -----------------------------------------------------------------------
    substation_km: dict[str, float] = {b.id: b.km for b in topology.buses}
    all_km_sorted = sorted(substation_km.values())

    # Build tap km set: every `tap_spacing` km along the corridor (excluding
    # positions already occupied by substations).
    max_km = max(all_km_sorted)
    tap_positions = set()
    km = tap_spacing
    while km < max_km:
        tap_positions.add(round(km, 6))
        km += tap_spacing
    # Remove positions that coincide with substations (within 0.1 km)
    for skm in all_km_sorted:
        tap_positions = {t for t in tap_positions if abs(t - skm) > 0.1}

    # Combined sorted list of all positions
    all_positions = sorted(
        [(km, "substation", bid) for bid, km in substation_km.items()]
        + [(t, "tap", f"TAP_{int(t)}") for t in sorted(tap_positions)]
    )

    # -----------------------------------------------------------------------
    # 2. Create buses
    # -----------------------------------------------------------------------
    pos_to_idx: dict[float, int] = {}  # km -> pandapower bus index
    id_to_idx: dict[str, int] = {}     # bus_id -> pandapower bus index

    tap_map: dict[int, float] = {}    # bus_idx -> km  (tap buses only)

    for km_pos, bus_type, bus_id in all_positions:
        b_config = topology.bus_by_id.get(bus_id)
        if b_config is not None:
            name = bus_id
            geodata = (0.0, km_pos)   # dummy lat; km as y for plotting
        else:
            name = bus_id  # e.g. "TAP_50"
            geodata = (0.0, km_pos)

        idx = pp.create_bus(
            net,
            vn_kv=base_kv,
            name=name,
            geodata=geodata,
            in_service=True,
        )
        pos_to_idx[km_pos] = idx
        id_to_idx[bus_id] = idx

        if bus_type == "tap":
            tap_map[idx] = km_pos

    # -----------------------------------------------------------------------
    # 3. External grid (slack) at SHARAVATHI
    # -----------------------------------------------------------------------
    slack_bus_idx = id_to_idx[topology.gen.bus]
    pp.create_ext_grid(
        net,
        bus=slack_bus_idx,
        vm_pu=topology.gen.vm_pu,
        name="SHARAVATHI_EXT",
    )

    # -----------------------------------------------------------------------
    # 4. Line segments between consecutive positions
    # -----------------------------------------------------------------------
    section_map: dict[int, tuple[float, float, str, str]] = {}

    # Build a sorted list of km positions for line creation
    all_km_list = sorted(pos_to_idx.keys())

    for i in range(len(all_km_list) - 1):
        from_km = all_km_list[i]
        to_km = all_km_list[i + 1]
        seg_len = to_km - from_km  # km

        from_idx = pos_to_idx[from_km]
        to_idx = pos_to_idx[to_km]

        # Reverse-lookup bus names
        from_name = net.bus.at[from_idx, "name"]
        to_name = net.bus.at[to_idx, "name"]

        # Effective impedance for double-circuit parallel lines:
        # parallel circuits halve R and X; capacitance doubles
        circuits = lp.circuits
        r_pu_total = (lp.r_ohm_per_km * seg_len) / circuits
        x_pu_total = (lp.x_ohm_per_km * seg_len) / circuits
        c_total = lp.c_nf_per_km * seg_len * circuits
        max_i_ka = lp.max_i_ka * circuits

        line_idx = pp.create_line_from_parameters(
            net,
            from_bus=from_idx,
            to_bus=to_idx,
            length_km=seg_len,
            r_ohm_per_km=lp.r_ohm_per_km / circuits,
            x_ohm_per_km=lp.x_ohm_per_km / circuits,
            c_nf_per_km=lp.c_nf_per_km * circuits,
            max_i_ka=max_i_ka,
            name=f"LINE_{from_name}_{to_name}",
            parallel=1,            # already folded parallel into parameters
            in_service=True,
        )
        section_map[line_idx] = (from_km, to_km, from_name, to_name)

    # -----------------------------------------------------------------------
    # 5. Accounted (metered) loads scaled by load_scale
    # -----------------------------------------------------------------------
    for lc in topology.loads:
        bus_idx = id_to_idx[lc.bus]
        pp.create_load(
            net,
            bus=bus_idx,
            p_mw=lc.p_mw * load_scale,
            q_mvar=lc.q_mvar * load_scale,
            name=f"LOAD_{lc.bus}",
            in_service=True,
        )

    # -----------------------------------------------------------------------
    # 5b. Shunt capacitor banks (reactive compensation)
    # pandapower shunt: positive q_mvar = inductive; negative = capacitive.
    # Our config stores capacitive injection as positive q_mvar_cap, so
    # we negate it when creating the shunt element.
    # -----------------------------------------------------------------------
    for sc in topology.shunts:
        bus_idx = id_to_idx[sc.bus]
        pp.create_shunt(
            net,
            bus=bus_idx,
            q_mvar=-sc.q_mvar_cap,   # negative = capacitive injection
            p_mw=0.0,
            name=f"SHUNT_CAP_{sc.bus}",
            in_service=True,
        )

    # -----------------------------------------------------------------------
    # 6. Optional theft injection
    # -----------------------------------------------------------------------
    if theft is not None:
        tap_bus_idx, snapped_km = _snap_to_tap(theft.km, tap_map, pos_to_idx)
        pp.create_load(
            net,
            bus=tap_bus_idx,
            p_mw=theft.mw,
            q_mvar=theft.q_mvar,
            name="THEFT",
            in_service=True,
        )
        # Store injection metadata on net for downstream modules
        net["theft_km"] = snapped_km
        net["theft_bus_idx"] = tap_bus_idx
    else:
        net["theft_km"] = None
        net["theft_bus_idx"] = None

    # -----------------------------------------------------------------------
    # 7. Attach extra maps
    # -----------------------------------------------------------------------
    net["tap_map"] = tap_map          # {bus_idx: km}
    net["section_map"] = section_map  # {line_idx: (from_km, to_km, from_name, to_name)}
    net["id_to_idx"] = id_to_idx
    net["pos_to_idx"] = pos_to_idx

    return net


# ---------------------------------------------------------------------------
# Snap helper
# ---------------------------------------------------------------------------


def _snap_to_tap(
    km: float,
    tap_map: dict[int, float],
    pos_to_idx: dict[float, int],
) -> tuple[int, float]:
    """Return (bus_index, snapped_km) of the tap bus nearest to `km`."""
    if not tap_map:
        raise ValueError("No tap buses in network — check tap_spacing_km.")
    best_idx = min(tap_map, key=lambda idx: abs(tap_map[idx] - km))
    snapped_km = tap_map[best_idx]
    return best_idx, snapped_km


def get_tap_id_for_km(km: float, net: pp.pandapowerNet) -> tuple[int, float, str]:
    """Public helper: returns (bus_idx, snapped_km, tap_name) for a given km."""
    tap_map: dict[int, float] = net["tap_map"]
    idx, snapped = _snap_to_tap(km, tap_map, net["pos_to_idx"])
    return idx, snapped, net.bus.at[idx, "name"]


# ---------------------------------------------------------------------------
# CLI entry point: python -m grid.build [--check]
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build Sharavathi corridor network and optionally run a sanity check."
    )
    parser.add_argument("--check", action="store_true", help="Print convergence & stats")
    parser.add_argument("--theft-km", type=float, default=None, help="Inject theft at km")
    parser.add_argument("--theft-mw", type=float, default=20.0, help="Theft magnitude MW")
    parser.add_argument("--theft-pf", type=float, default=0.90, help="Theft power factor")
    parser.add_argument("--load-scale", type=float, default=1.0, help="Load scaling factor")
    args = parser.parse_args()

    from grid.flow import run_pf, PFNonConvergence

    theft = None
    if args.theft_km is not None:
        theft = TheftSpec(km=args.theft_km, mw=args.theft_mw, pf=args.theft_pf)

    print("Building network …")
    net = build_net(load_scale=args.load_scale, theft=theft)

    n_buses = len(net.bus)
    n_lines = len(net.line)
    n_taps  = len(net["tap_map"])
    print(f"  Buses: {n_buses}  |  Lines: {n_lines}  |  Tap buses: {n_taps}")

    if args.check or True:  # always run PF when invoked as --check
        try:
            results = run_pf(net)
        except PFNonConvergence as exc:
            print(f"[FAIL] Power flow did not converge: {exc}")
            return

        total_loss_mw = results["total_loss_mw"]
        gen_mw        = results["gen_p_mw"]
        load_mw       = results["load_p_mw"]
        max_load_pct  = results["max_line_loading_pct"]
        min_vm        = results["min_vm_pu"]
        max_vm        = results["max_vm_pu"]
        loss_pct      = 100.0 * total_loss_mw / gen_mw if gen_mw else float("nan")

        print(f"\n[OK] Power flow converged.")
        print(f"  Generation : {gen_mw:.2f} MW")
        print(f"  Load       : {load_mw:.2f} MW")
        print(f"  Losses     : {total_loss_mw:.3f} MW  ({loss_pct:.2f} %)")
        print(f"  Max loading: {max_load_pct:.1f} %")
        print(f"  Voltage    : {min_vm:.4f} – {max_vm:.4f} pu")

        if theft:
            _, snapped_km, tap_name = get_tap_id_for_km(args.theft_km, net)
            print(f"\n  Theft injected: {args.theft_mw} MW @ km {snapped_km} ({tap_name})")

        # Sanity checks
        if not (0.95 <= min_vm and max_vm <= 1.05):
            print("  [WARN] Voltage outside 0.95–1.05 pu band — check parameters.")
        if not (1.0 <= loss_pct <= 4.0):
            print(f"  [WARN] Losses {loss_pct:.2f}% outside expected 1–4% range.")


if __name__ == "__main__":
    _cli()
