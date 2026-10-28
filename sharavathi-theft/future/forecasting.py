"""
future/forecasting.py — Demand-forecast interface (07_FUTURE_MITIGATION). STUB ONLY.

Planned: short-term load forecast per bus (LSTM / XGBoost on synthetic daily profiles from
simdata.operating_points). get_forecaster() returns a forecaster that reports
"not_implemented" until a real model is added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DemandForecast:
    status: str                                    # "not_implemented" | "ok"
    horizon_steps: int = 0
    step_minutes: int = 15
    per_bus_mw: dict[str, list[float]] = field(default_factory=dict)   # bus id → forecast series
    interval_mw: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


class DemandForecaster(ABC):
    @abstractmethod
    def forecast(self, history: dict[str, list[float]], horizon_steps: int) -> DemandForecast:
        """history: bus id → past load series (MW)."""


class NotImplementedForecaster(DemandForecaster):
    def forecast(self, history, horizon_steps):
        return DemandForecast("not_implemented", horizon_steps=horizon_steps)


def get_forecaster() -> DemandForecaster:
    # later: read feature_flags.yaml and return a trained forecaster
    return NotImplementedForecaster()
