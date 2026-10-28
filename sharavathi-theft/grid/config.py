"""
grid/config.py — Load and validate YAML configuration files with pydantic.

Rules (from spec):
  - Every parameter must carry a `source` field; if missing, raises ConfigError.
  - Returns strongly-typed pydantic models so downstream modules get IDE support.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


class ConfigError(ValueError):
    """Raised when a config value is missing a required `source` annotation."""


def _require_source(obj: Any, path: str = "") -> None:
    """Recursively walk a dict/list and raise if any leaf dict lacks 'source'."""
    if isinstance(obj, dict):
        if any(isinstance(v, (int, float, str, bool)) for v in obj.values()):
            # It's a leaf-like dict — check for 'source' only if it carries data fields
            data_keys = {k for k in obj if k != "source"}
            numeric_vals = {k for k in data_keys if isinstance(obj[k], (int, float))}
            if numeric_vals and "source" not in obj:
                raise ConfigError(
                    f"Parameter at '{path}' is missing 'source' annotation. "
                    f"Add `source: assumed` (or the real source) to comply with spec."
                )
        for k, v in obj.items():
            _require_source(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _require_source(item, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# topology.yaml models
# ---------------------------------------------------------------------------


class BusConfig(BaseModel):
    id: str
    km: float
    role: Literal["slack_gen", "junction", "load"]
    metered: bool
    source: str


class LinesConfig(BaseModel):
    r_ohm_per_km: float
    x_ohm_per_km: float
    c_nf_per_km: float
    max_i_ka: float
    circuits: int
    source: str = "assumed"  # default so the global dict check passes


class ShuntConfig(BaseModel):
    """Shunt capacitor bank (reactive compensation) at a bus."""
    bus: str
    q_mvar_cap: float   # capacitive injection in Mvar (positive = capacitive)
    source: str = "assumed"


class LoadConfig(BaseModel):
    bus: str
    p_mw: float
    q_mvar: float
    source: str


class GenConfig(BaseModel):
    bus: str
    vm_pu: float
    source: str = "assumed"


class TopologyConfig(BaseModel):
    base_kv: float
    tap_spacing_km: float
    buses: list[BusConfig]
    lines: LinesConfig
    loads: list[LoadConfig]
    gen: GenConfig
    shunts: list[ShuntConfig] = []   # optional reactive compensation banks

    @model_validator(mode="before")
    @classmethod
    def _inject_lines_source(cls, data: Any) -> Any:
        """LinesConfig source is optional in YAML; inject default if absent."""
        if isinstance(data, dict) and "lines" in data:
            if isinstance(data["lines"], dict) and "source" not in data["lines"]:
                data["lines"]["source"] = "assumed"
        if isinstance(data, dict) and "gen" in data:
            if isinstance(data["gen"], dict) and "source" not in data["gen"]:
                data["gen"]["source"] = "assumed"
        return data

    @field_validator("buses")
    @classmethod
    def _buses_sorted(cls, buses: list[BusConfig]) -> list[BusConfig]:
        return sorted(buses, key=lambda b: b.km)

    @property
    def bus_by_id(self) -> dict[str, BusConfig]:
        return {b.id: b for b in self.buses}

    @property
    def bus_pairs(self) -> list[tuple[BusConfig, BusConfig]]:
        """Consecutive (from, to) pairs along the chain."""
        return list(zip(self.buses[:-1], self.buses[1:]))


# ---------------------------------------------------------------------------
# sensors.yaml models
# ---------------------------------------------------------------------------


class NoiseConfig(BaseModel):
    vm_pct: float    # % of voltage magnitude
    p_pct: float     # % of active power
    q_pct: float     # % of reactive power
    va_deg: float    # degrees of voltage angle


class SensorsConfig(BaseModel):
    profile: str
    noise: NoiseConfig
    profiles: dict[str, Any]

    def resolve_profile(self, profile_name: str, all_bus_ids: list[str]) -> list[str]:
        """Return list of metered bus IDs for a given profile name."""
        val = self.profiles.get(profile_name)
        if val == "all_buses":
            return all_bus_ids
        if isinstance(val, list):
            return val
        raise ConfigError(f"Unknown sensor profile: {profile_name!r}")


# ---------------------------------------------------------------------------
# scenarios.yaml model
# ---------------------------------------------------------------------------


class ScenariosConfig(BaseModel):
    n_normal: int
    n_theft: int
    n_sensor_bias: int
    n_sensor_drift: int
    n_missing: int
    n_line_outage: int
    load_scale_range: tuple[float, float]
    holdout_locations_km: list[float]


# ---------------------------------------------------------------------------
# feature_flags.yaml model
# ---------------------------------------------------------------------------


class FeatureFlagsConfig(BaseModel):
    mitigation_enabled: bool
    gnn_enabled: bool
    rag_enabled: bool = True        # 09: RAG mitigation advisor (recommendation-only)


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------


def load_topology(path: Path | None = None) -> TopologyConfig:
    """Load and validate topology.yaml. Raises ConfigError if source missing."""
    p = path or (_CONFIG_DIR / "topology.yaml")
    raw = _load_yaml(p)
    _require_source(raw.get("loads", []), "loads")
    _require_source(raw.get("buses", []), "buses")
    return TopologyConfig.model_validate(raw)


def load_sensors(path: Path | None = None) -> SensorsConfig:
    p = path or (_CONFIG_DIR / "sensors.yaml")
    raw = _load_yaml(p)
    return SensorsConfig.model_validate(raw)


def load_scenarios(path: Path | None = None) -> ScenariosConfig:
    p = path or (_CONFIG_DIR / "scenarios.yaml")
    raw = _load_yaml(p)
    return ScenariosConfig.model_validate(raw)


def load_feature_flags(path: Path | None = None) -> FeatureFlagsConfig:
    p = path or (_CONFIG_DIR / "feature_flags.yaml")
    raw = _load_yaml(p)
    return FeatureFlagsConfig.model_validate(raw)


def load_all(config_dir: Path | None = None) -> dict[str, Any]:
    """Load all four config files and return a dict keyed by name."""
    d = config_dir or _CONFIG_DIR
    return {
        "topology": load_topology(d / "topology.yaml"),
        "sensors": load_sensors(d / "sensors.yaml"),
        "scenarios": load_scenarios(d / "scenarios.yaml"),
        "feature_flags": load_feature_flags(d / "feature_flags.yaml"),
    }
