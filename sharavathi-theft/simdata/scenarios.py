"""
simdata/scenarios.py — One builder function per scenario class.

All functions return a ScenarioSpec describing the parameters needed to
simulate one row in dataset.parquet. The actual PF + measurement call
happens in generate.py.

Scenario classes
----------------
normal       : base operating point, no anomaly
theft        : real pp.create_load at a tap bus (never subtract from meters)
sensor_bias  : +/- constant offset on 1-2 random measurement channels
sensor_drift : linearly growing offset on one channel
missing      : 1-3 channels become NaN
line_outage  : one line segment taken out of service (double-circuit => 1 remains)
mixed        : theft + sensor_bias combined (5% of dataset for robustness)

Public API
----------
ScenarioSpec      : dataclass
build_normal(op, rng)         -> ScenarioSpec
build_theft(op, tap_kms, rng) -> ScenarioSpec
build_sensor_bias(op, rng)    -> ScenarioSpec
build_sensor_drift(op, rng)   -> ScenarioSpec
build_missing(op, rng)        -> ScenarioSpec
build_line_outage(op, n_lines, rng) -> ScenarioSpec
build_mixed(op, tap_kms, rng) -> ScenarioSpec
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from simdata.operating_points import OperatingPoint


# ---------------------------------------------------------------------------
# Deterministic UUID helper — uses numpy RNG, reproducible with same seed
# ---------------------------------------------------------------------------

def _det_uuid(rng: np.random.Generator) -> str:
    """Generate a deterministic UUID4-format string from numpy RNG bytes."""
    b = rng.bytes(16)
    ba = bytearray(b)
    ba[6] = (ba[6] & 0x0F) | 0x40   # version 4
    ba[8] = (ba[8] & 0x3F) | 0x80   # variant bits
    return str(uuid.UUID(bytes=bytes(ba)))


def _new_id(rng: np.random.Generator) -> str:
    """Deterministic UUID from numpy RNG — reproducible with same seed."""
    return _det_uuid(rng)


# ---------------------------------------------------------------------------
# ScenarioSpec
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSpec:
    """All parameters needed to simulate and record one dataset row."""
    scenario_id: str
    group_id: str
    label: str                            # normal/theft/sensor_bias/sensor_drift/missing/line_outage
    load_scale: float
    bus_jitter: dict[str, float]

    # Theft fields (NaN/None if not theft)
    theft_km: Optional[float] = None      # requested km (before snap)
    theft_mw: Optional[float] = None
    theft_pf: Optional[float] = None

    # Sensor fault fields
    fault_channels: list[str] = field(default_factory=list)
    fault_type: Optional[str] = None      # "bias" | "drift" | "missing" | "stuck"
    fault_pct: float = 0.0                # bias %
    fault_rate: float = 0.0              # drift rate % per step
    fault_step: int = 0                  # drift step count (simulated time index)

    # Line outage
    outage_line_idx: Optional[int] = None  # pandapower line index to disable

    # Sensor profile used for this scenario
    sensor_profile: str = "major_only"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_channels(metered_buses: list[str], rng: np.random.Generator,
                     n: int = 2) -> list[str]:
    """Pick n random sensor channel keys from metered buses."""
    suffixes = ["vm", "p", "q"]
    pool = [f"m_{b}_{s}" for b in metered_buses for s in suffixes]
    n = min(n, len(pool))
    chosen = rng.choice(pool, size=n, replace=False).tolist()
    return chosen


_METERED_BUSES = ["SHARAVATHI", "SHIMOGA", "NELAMANGALA", "PEENYA"]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_normal(op: OperatingPoint, rng: np.random.Generator,
                 profile: str = "major_only") -> ScenarioSpec:
    """No anomaly — base operating point only."""
    return ScenarioSpec(
        scenario_id=_new_id(rng),
        group_id=op.group_id,
        label="normal",
        load_scale=op.load_scale,
        bus_jitter=op.bus_jitter,
        sensor_profile=profile,
    )


def build_theft(
    op: OperatingPoint,
    tap_kms: list[float],
    rng: np.random.Generator,
    profile: str = "major_only",
    magnitude_range: tuple[float, float] = (1.0, 40.0),
) -> ScenarioSpec:
    """
    Real unmetered extraction at a random tap bus.
    magnitude_range: (min_mw, max_mw) — includes small-theft regime 1–3 MW.
    """
    km = float(rng.choice(tap_kms))
    mw = float(rng.uniform(*magnitude_range))
    pf = float(rng.uniform(0.85, 0.98))
    return ScenarioSpec(
        scenario_id=_new_id(rng),
        group_id=op.group_id,
        label="theft",
        load_scale=op.load_scale,
        bus_jitter=op.bus_jitter,
        theft_km=km,
        theft_mw=mw,
        theft_pf=pf,
        sensor_profile=profile,
    )


def build_theft_small(
    op: OperatingPoint,
    tap_kms: list[float],
    rng: np.random.Generator,
    profile: str = "major_only",
) -> ScenarioSpec:
    """Small theft 1–3 MW — near noise floor, tests sensitivity limits."""
    return build_theft(op, tap_kms, rng, profile, magnitude_range=(1.0, 3.0))


def build_sensor_bias(
    op: OperatingPoint,
    rng: np.random.Generator,
    profile: str = "major_only",
) -> ScenarioSpec:
    """Constant +/- 2–10% bias on 1–2 random measurement channels."""
    n_channels = int(rng.choice([1, 2]))
    channels = _random_channels(_METERED_BUSES, rng, n=n_channels)
    pct = float(rng.uniform(2.0, 10.0)) * rng.choice([-1.0, 1.0])
    return ScenarioSpec(
        scenario_id=_new_id(rng),
        group_id=op.group_id,
        label="sensor_bias",
        load_scale=op.load_scale,
        bus_jitter=op.bus_jitter,
        fault_channels=channels,
        fault_type="bias",
        fault_pct=float(pct),
        sensor_profile=profile,
    )


def build_sensor_drift(
    op: OperatingPoint,
    rng: np.random.Generator,
    profile: str = "major_only",
) -> ScenarioSpec:
    """Linear drift on one channel — rate 0.2–2.0% per simulated step."""
    channels = _random_channels(_METERED_BUSES, rng, n=1)
    rate = float(rng.uniform(0.2, 2.0)) * rng.choice([-1.0, 1.0])
    step = int(rng.integers(5, 50))   # simulated elapsed time steps
    return ScenarioSpec(
        scenario_id=_new_id(rng),
        group_id=op.group_id,
        label="sensor_drift",
        load_scale=op.load_scale,
        bus_jitter=op.bus_jitter,
        fault_channels=channels,
        fault_type="drift",
        fault_rate=rate,
        fault_step=step,
        sensor_profile=profile,
    )


def build_missing(
    op: OperatingPoint,
    rng: np.random.Generator,
    profile: str = "major_only",
) -> ScenarioSpec:
    """Drop 1–3 measurement channels (prob=1.0 missing fault)."""
    n_channels = int(rng.choice([1, 2, 3]))
    channels = _random_channels(_METERED_BUSES, rng, n=n_channels)
    return ScenarioSpec(
        scenario_id=_new_id(rng),
        group_id=op.group_id,
        label="missing",
        load_scale=op.load_scale,
        bus_jitter=op.bus_jitter,
        fault_channels=channels,
        fault_type="missing",
        sensor_profile=profile,
    )


def build_line_outage(
    op: OperatingPoint,
    n_lines: int,
    rng: np.random.Generator,
    profile: str = "major_only",
) -> ScenarioSpec:
    """
    Take one random line segment out of service.
    Since this is already a 2-circuit equivalent model, an 'outage' doubles
    effective impedance — we model it by setting the segment in_service=False
    (pandapower re-routes flow through adjacent segments, PF may fail for
    isolated sections — those cases are skipped in generate.py).
    """
    line_idx = int(rng.integers(0, n_lines))
    return ScenarioSpec(
        scenario_id=_new_id(rng),
        group_id=op.group_id,
        label="line_outage",
        load_scale=op.load_scale,
        bus_jitter=op.bus_jitter,
        outage_line_idx=line_idx,
        sensor_profile=profile,
    )


def build_mixed(
    op: OperatingPoint,
    tap_kms: list[float],
    rng: np.random.Generator,
    profile: str = "major_only",
) -> ScenarioSpec:
    """
    Theft + sensor_bias combined — robustness case (5% of data).
    Label stays 'theft' since that's the primary anomaly.
    """
    spec = build_theft(op, tap_kms, rng, profile)
    # Add a bias on top
    n_channels = int(rng.choice([1, 2]))
    channels = _random_channels(_METERED_BUSES, rng, n=n_channels)
    pct = float(rng.uniform(2.0, 8.0)) * rng.choice([-1.0, 1.0])
    spec.fault_channels = channels
    spec.fault_type = "bias"
    spec.fault_pct = float(pct)
    # Keep label as "theft" so the mixed case trains robustness
    return spec
