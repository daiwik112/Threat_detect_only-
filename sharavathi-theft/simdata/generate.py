"""
simdata/generate.py — Parallel scenario generator → data/processed/dataset.parquet

Public API
----------
simulate_one(spec, topology, sensors_cfg, rng_seed) -> dict | None
    Simulate a single ScenarioSpec: build net → run PF → measure → return row dict.
    Returns None if PF does not converge (non-converged cases are skipped + logged).

generate_dataset(scenarios_cfg, topology, sensors_cfg, profile, out_dir, seed, n_jobs)
    Full pipeline: operating points → specs → parallel simulate → parquet.

CLI
---
python -m simdata.generate --profile major_only --out data/processed/ --seed 42
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from grid.build import build_net, TheftSpec
from grid.config import ScenariosConfig, TopologyConfig, SensorsConfig
from grid.flow import run_pf, PFNonConvergence
from grid.sensors import measure, enrich_results_with_bus_power, FaultSpec

from simdata.operating_points import generate_operating_points, OperatingPoint
from simdata.scenarios import (
    ScenarioSpec,
    build_normal, build_theft, build_theft_small,
    build_sensor_bias, build_sensor_drift,
    build_missing, build_line_outage, build_mixed,
)

log = logging.getLogger(__name__)

_DEFAULT_OUT = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Single scenario simulation
# ---------------------------------------------------------------------------

def simulate_one(
    spec: ScenarioSpec,
    topology: TopologyConfig,
    sensors_cfg: SensorsConfig,
    rng_seed: int,
) -> Optional[dict[str, Any]]:
    """
    Simulate one ScenarioSpec end-to-end.

    Returns a flat dict matching the dataset.parquet schema, or None if the
    power flow fails to converge (caller should skip + count).
    """
    rng = np.random.default_rng(rng_seed)

    # --- Build network ---
    theft = None
    if spec.theft_km is not None:
        theft = TheftSpec(km=spec.theft_km, mw=spec.theft_mw, pf=spec.theft_pf)

    try:
        net = build_net(topology=topology, load_scale=spec.load_scale, theft=theft)
    except Exception as exc:
        log.warning("build_net failed for %s: %s", spec.scenario_id, exc)
        return None

    # Apply line outage
    if spec.outage_line_idx is not None:
        idx = spec.outage_line_idx
        if idx < len(net.line):
            net.line.at[idx, "in_service"] = False
        else:
            log.warning("outage_line_idx %d out of range (%d lines)", idx, len(net.line))

    # --- Power flow ---
    try:
        results = run_pf(net)
    except PFNonConvergence:
        log.debug("PF did not converge: %s (label=%s, ls=%.3f)",
                  spec.scenario_id, spec.label, spec.load_scale)
        return None

    results = enrich_results_with_bus_power(results, net)

    # --- Sensor measurement ---
    id_to_idx: dict[str, int] = net["id_to_idx"]
    bus_names: dict[int, str] = {idx: row["name"]
                                  for idx, row in net.bus.iterrows()}

    # Build fault specs from ScenarioSpec
    faults: list[FaultSpec] = []
    if spec.fault_type and spec.fault_channels:
        for ch in spec.fault_channels:
            if spec.fault_type == "bias":
                faults.append(FaultSpec(ch, "bias", pct=spec.fault_pct))
            elif spec.fault_type == "drift":
                faults.append(FaultSpec(ch, "drift",
                                        rate=spec.fault_rate, step=spec.fault_step))
            elif spec.fault_type == "missing":
                faults.append(FaultSpec(ch, "missing", prob=1.0))

    measurements = measure(
        results, spec.sensor_profile, sensors_cfg,
        id_to_idx, bus_names, rng,
        faults=faults if faults else None,
    )

    # --- Assemble row ---
    row: dict[str, Any] = {
        "scenario_id":   spec.scenario_id,
        "group_id":      spec.group_id,
        "label":         spec.label,
        "true_km":       spec.theft_km if spec.theft_km is not None else float("nan"),
        "true_tap_id":   (f"TAP_{int(net['theft_km'])}"
                          if net.get("theft_km") is not None else None),
        "magnitude_mw":  spec.theft_mw if spec.theft_mw is not None else 0.0,
        "load_scale":    spec.load_scale,
        "sensor_profile": spec.sensor_profile,
        # physics summary (useful for sanity checks)
        "_gen_p_mw":     results["gen_p_mw"],
        "_loss_mw":      results["total_loss_mw"],
        "_min_vm_pu":    results["min_vm_pu"],
        "_max_vm_pu":    results["max_vm_pu"],
    }

    # Add all measurements (NaN-safe)
    for key, val in measurements.items():
        if val is None:
            row[key] = float("nan")
        else:
            row[key] = float(val)

    return row


# ---------------------------------------------------------------------------
# Full dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    scenarios_cfg: ScenariosConfig,
    topology: TopologyConfig,
    sensors_cfg: SensorsConfig,
    profile: str = "major_only",
    out_dir: Path = _DEFAULT_OUT,
    seed: int = 42,
    n_jobs: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate the full synthetic dataset and save as parquet.

    Steps
    -----
    1. Generate operating points (one per group_id)
    2. Expand into ScenarioSpecs (all classes)
    3. Simulate in parallel (joblib)
    4. Save dataset.parquet

    Returns the DataFrame.
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # --- 1. Operating points ---
    # How many groups? Total scenarios / avg scenarios per group (~4-5)
    total = (scenarios_cfg.n_normal + scenarios_cfg.n_theft +
             scenarios_cfg.n_sensor_bias + scenarios_cfg.n_sensor_drift +
             scenarios_cfg.n_missing + scenarios_cfg.n_line_outage)
    n_groups = max(total // 5, 500)

    if verbose:
        print(f"Generating {n_groups} operating points for {total} scenarios …")

    ops = generate_operating_points(scenarios_cfg, n_groups, rng)

    # --- 2. Build ScenarioSpecs ---
    tap_kms = sorted(topology.tap_spacing_km * i
                     for i in range(1, 500)
                     if topology.tap_spacing_km * i < max(b.km for b in topology.buses))
    # Use actual tap positions from a quick net build
    try:
        from grid.build import build_net as _bn
        _net = _bn(topology=topology, load_scale=1.0)
        tap_kms = sorted(_net["tap_map"].values())
    except Exception:
        pass  # fall back to computed list

    n_lines = 38  # known from WF1

    specs: list[ScenarioSpec] = []
    seeds_per_spec: list[int] = []

    def _add(build_fn, n, *args, **kwargs):
        for _ in range(n):
            op = ops[rng.integers(0, len(ops))]
            sp = build_fn(op, *args, rng=rng, profile=profile, **kwargs)
            specs.append(sp)
            seeds_per_spec.append(int(rng.integers(0, 2**31)))

    _add(build_normal,       scenarios_cfg.n_normal)
    # 90% full range theft, 10% small theft (1-3 MW sensitivity)
    n_theft_full  = int(scenarios_cfg.n_theft * 0.90)
    n_theft_small = scenarios_cfg.n_theft - n_theft_full
    _add(build_theft,        n_theft_full,  tap_kms)
    _add(build_theft_small,  n_theft_small, tap_kms)
    _add(build_sensor_bias,  scenarios_cfg.n_sensor_bias)
    _add(build_sensor_drift, scenarios_cfg.n_sensor_drift)
    _add(build_missing,      scenarios_cfg.n_missing)
    _add(build_line_outage,  scenarios_cfg.n_line_outage, n_lines)
    # 5% mixed (theft + bias) — use 5% of n_theft count
    n_mixed = max(int(scenarios_cfg.n_theft * 0.05), 50)
    _add(build_mixed,        n_mixed, tap_kms)

    if verbose:
        print(f"Total specs: {len(specs)} | profile: {profile} | seed: {seed}")

    # --- 3. Simulate (parallel or serial) ---
    rows = _run_simulations(specs, seeds_per_spec, topology, sensors_cfg,
                            n_jobs=n_jobs, verbose=verbose)

    n_failed = len(specs) - len(rows)
    if verbose:
        elapsed = time.time() - t0
        print(f"\nSimulated {len(rows)}/{len(specs)} rows "
              f"({n_failed} non-converged skipped) in {elapsed:.1f}s")

    # --- 4. Save ---
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dataset_{profile}.parquet"
    df.to_parquet(out_path, index=False)

    if verbose:
        _print_summary(df, out_path)

    return df


def _run_simulations(
    specs: list[ScenarioSpec],
    seeds: list[int],
    topology: TopologyConfig,
    sensors_cfg: SensorsConfig,
    n_jobs: int = 1,
    verbose: bool = True,
) -> list[dict]:
    """Run simulate_one for each spec, serially or in parallel."""
    if n_jobs == 1:
        rows = []
        for i, (spec, seed) in enumerate(zip(specs, seeds)):
            row = simulate_one(spec, topology, sensors_cfg, seed)
            if row is not None:
                rows.append(row)
            if verbose and (i % 500 == 0 or i == len(specs) - 1):
                print(f"  [{i+1:5d}/{len(specs)}] done={len(rows)} "
                      f"label={spec.label:<14} ls={spec.load_scale:.3f}")
        return rows
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(simulate_one)(spec, topology, sensors_cfg, seed)
            for spec, seed in zip(specs, seeds)
        )
        return [r for r in results if r is not None]


def _print_summary(df: pd.DataFrame, path: Path) -> None:
    print(f"\nSaved: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print("\nClass counts:")
    for label, count in df["label"].value_counts().items():
        pct = 100.0 * count / len(df)
        print(f"  {label:<16} {count:6d}  ({pct:.1f}%)")
    theft_df = df[df["label"] == "theft"]
    if len(theft_df) > 0:
        print(f"\nTheft magnitude: {theft_df['magnitude_mw'].min():.1f} – "
              f"{theft_df['magnitude_mw'].max():.1f} MW")
        print(f"Theft locations: {theft_df['true_km'].min():.0f} – "
              f"{theft_df['true_km'].max():.0f} km")
    print(f"Load scale range: {df['load_scale'].min():.3f} – "
          f"{df['load_scale'].max():.3f}")
