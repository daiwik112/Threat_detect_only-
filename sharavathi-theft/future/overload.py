"""
future/overload.py — Overload-prediction interface (07_FUTURE_MITIGATION). STUB ONLY.

Planned: predict line-segment loading above its limit N steps ahead (classifier on power-flow
time series, reusing grid.flow for labels). get_overload_predictor() returns a predictor that
reports "not_implemented" until a real model is added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SegmentRisk:
    line_id: str              # e.g. "LINE12"
    p_overload: float         # probability loading > limit within the horizon
    expected_peak_pct: float


@dataclass
class OverloadForecast:
    status: str                                     # "not_implemented" | "ok"
    horizon_steps: int = 0
    risks: list[SegmentRisk] = field(default_factory=list)


class OverloadPredictor(ABC):
    @abstractmethod
    def predict(self, net, horizon_steps: int) -> OverloadForecast: ...


class NotImplementedOverloadPredictor(OverloadPredictor):
    def predict(self, net, horizon_steps):
        return OverloadForecast("not_implemented", horizon_steps=horizon_steps)


def get_overload_predictor() -> OverloadPredictor:
    # later: read feature_flags.yaml and return a trained predictor
    return NotImplementedOverloadPredictor()
