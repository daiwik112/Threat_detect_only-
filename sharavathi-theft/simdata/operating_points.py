"""
simdata/operating_points.py — Generate base operating points.

Each operating point = one load_scale value drawn from a realistic daily/seasonal
load curve. Multiple scenario variants (normal, theft, fault) share the same
operating point via a group_id so group-based splitting prevents leakage.

Public API
----------
generate_operating_points(scenarios_cfg, n_groups, rng) -> list[OperatingPoint]
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from grid.config import ScenariosConfig


def _det_uuid(rng: np.random.Generator) -> str:
    """Deterministic UUID4-format string from numpy RNG bytes."""
    b = rng.bytes(16)
    ba = bytearray(b)
    ba[6] = (ba[6] & 0x0F) | 0x40
    ba[8] = (ba[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(ba)))


@dataclass
class OperatingPoint:
    """One base grid state that multiple scenarios share."""
    group_id: str                        # UUID shared by all scenarios from this point
    load_scale: float                    # overall scaling factor
    bus_jitter: dict[str, float]         # per-bus multiplicative jitter {bus_id: factor}
    hour: float                          # simulated hour-of-day (0-23.99)
    season: str                          # "summer" | "winter" | "monsoon" | "spring"


# Season multipliers (assumed; Karnataka load pattern)
_SEASON_FACTORS = {
    "summer":  1.10,   # high AC load
    "monsoon": 0.90,   # moderate
    "winter":  0.95,
    "spring":  0.85,
}

_SEASONS = list(_SEASON_FACTORS.keys())

# Substation buses that get per-bus jitter
_LOAD_BUSES = ["SHIMOGA", "SIRA_KBCROSS", "NELAMANGALA", "PEENYA"]


def _daily_curve(hour: float) -> float:
    """
    Normalized daily load factor (0-1) based on a double-hump sinusoid.
    Peak ~18:00 (evening), trough ~04:00 (night).
    """
    # morning hump at 09:00, evening peak at 18:00
    morning = 0.3 * math.sin(math.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.0
    evening = 0.5 * math.sin(math.pi * (hour - 14) / 10) if 14 <= hour <= 24 else 0.0
    base = 0.4  # night-time base
    return max(0.1, base + morning + evening)


def generate_operating_points(
    scenarios_cfg: ScenariosConfig,
    n_groups: int,
    rng: np.random.Generator,
) -> list[OperatingPoint]:
    """
    Generate `n_groups` base operating points spanning the load-scale range.

    Parameters
    ----------
    scenarios_cfg : ScenariosConfig from config/scenarios.yaml
    n_groups      : number of distinct operating points (= number of group_ids)
    rng           : numpy random Generator

    Returns
    -------
    list of OperatingPoint, one per group
    """
    ls_min, ls_max = scenarios_cfg.load_scale_range
    points: list[OperatingPoint] = []

    for _ in range(n_groups):
        # Random hour with realistic weighting (more daytime scenarios)
        hour = float(rng.choice(
            np.arange(0, 24, 0.5),
            p=_hour_weights(),
        ))
        season = _SEASONS[rng.integers(0, len(_SEASONS))]
        season_f = _SEASON_FACTORS[season]

        # Combine daily curve + seasonal factor + small noise
        raw_ls = _daily_curve(hour) * season_f
        # Scale into configured range
        span = ls_max - ls_min
        load_scale = float(np.clip(
            ls_min + span * (raw_ls / 1.1) + rng.normal(0, 0.02),
            ls_min, ls_max,
        ))

        # Per-bus demand jitter ±5%
        bus_jitter = {
            bus: float(np.clip(rng.normal(1.0, 0.05), 0.85, 1.15))
            for bus in _LOAD_BUSES
        }

        points.append(OperatingPoint(
            group_id=_det_uuid(rng),
            load_scale=round(load_scale, 4),
            bus_jitter=bus_jitter,
            hour=hour,
            season=season,
        ))

    return points


def _hour_weights() -> np.ndarray:
    """Probability weights for each half-hour slot 0..23.5 — daytime heavier."""
    hours = np.arange(0, 24, 0.5)
    w = np.array([_daily_curve(h) for h in hours])
    w += 0.05  # ensure no zero probability at night
    return w / w.sum()
