"""
ml/features.py — Physics-based feature engineering (Step 1).

Only channels that a sensor profile actually meters are used:
  * bus channels   m_<BUS>_{vm,va,p,q}   for metered buses
  * line-end flows m_LINE<i>_{p,q}_{from,to} ONLY at the end touching a metered bus
    (the dataset also stores flows at unmetered tap segments — those are ignored,
    otherwise localization would read the answer off unmetered meters).

Expected (no-theft) values come from data/raw/baseline_states.parquet, linearly
interpolated at an ESTIMATED load scale. The true load_scale is never used.

Features (all prefixed f_)
--------------------------
Global
  f_load_scale_est      median of (metered load P / nominal P)        — robust to 1 bad channel
  f_load_scale_spread   max − min of those ratios                     — sensor-fault indicator
  f_n_missing           NaN count over the profile's metered channels
  f_r_p_total           sum of section active-power residuals (≈ unmetered MW)
Per section a→b (consecutive metered buses)
  f_r_p_<sec>, f_r_q_<sec>    absorbed power (P_in + P_to) minus expected — losses + unmetered load
  f_dp_in_<sec>               entry flow minus expected (rises for every section upstream of theft)
  f_dva_<sec>, f_dvm_<sec>    angle / voltage drop across section minus expected
  f_dva_mw_<sec>              f_dva converted to MW·(section length) units via X/V²
  f_xfrac_<sec>               angle-drop / (k·L·r_p): physics estimate of the theft position
                              inside the section as a fraction of its length (NaN if |r_p| < 1 MW)
Per metered bus
  f_vm_dev_<bus>, f_va_dev_<bus>   measured − expected
  f_kcl_p_<bus>, f_kcl_q_<bus>     bus injection + incident line-end flows (≈0 unless a bus sensor is faulty)

NaN is kept for unavailable values (XGBoost handles missing natively).

Public API
----------
build_feature_matrix(df, baseline_df, profile="major_only") -> DataFrame (copy with f_* appended)
get_feature_cols(df) -> list[str]
get_grid_meta(profile="major_only") -> GridMeta
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd


@dataclass
class Section:
    name: str
    from_bus: str
    to_bus: str
    km_from: float
    km_to: float
    first_line: int          # line whose from-end is at from_bus
    last_line: int           # line whose to-end is at to_bus
    from_idx: int
    to_idx: int
    taps: list[tuple[str, float]] = field(default_factory=list)   # candidate locations inside

    @property
    def length_km(self) -> float:
        return self.km_to - self.km_from


@dataclass
class GridMeta:
    profile: str
    bus_names: list[str]               # by pandapower index (chain order)
    bus_km: dict[str, float]
    metered: list[str]                 # sorted by km
    sections: list[Section]
    nominal_load_mw: dict[str, float]
    x_ohm_per_km: float
    base_kv: float
    candidates: list[tuple[str, float]]  # all theft-candidate buses (taps) with km

    @property
    def bus_idx(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.bus_names)}

    def section_of_km(self, km: float) -> str:
        for s in self.sections:
            if s.km_from < km <= s.km_to:
                return s.name
        return self.sections[0].name if km <= self.sections[0].km_from else self.sections[-1].name


@lru_cache(maxsize=8)
def get_grid_meta(profile: str = "major_only") -> GridMeta:
    """Network geometry + which channels the profile meters (built from config + grid.build)."""
    import warnings
    from grid.build import build_net
    from grid.config import load_sensors, load_topology

    topo = load_topology()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = build_net(topology=topo, load_scale=1.0)

    bus_names = [str(n) for n in net.bus.sort_index()["name"]]
    topo_km = {b.id: float(b.km) for b in topo.buses}
    bus_km = {n: topo_km[n] if n in topo_km else float(n.split("_")[1]) for n in bus_names}

    # Chain check: line i connects bus i → bus i+1
    for i, (fb, tb) in enumerate(zip(net.line.from_bus, net.line.to_bus)):
        if (fb, tb) != (i, i + 1):
            raise ValueError("ml.features expects a radial chain with line i = bus i → i+1")

    metered_raw = load_sensors().resolve_profile(profile, bus_names)
    metered = sorted(metered_raw, key=lambda n: bus_km[n])
    idx = {n: i for i, n in enumerate(bus_names)}

    sections = []
    for a, b in zip(metered[:-1], metered[1:]):
        ia, ib = idx[a], idx[b]
        taps = [(bus_names[j], bus_km[bus_names[j]]) for j in range(ia + 1, ib + 1)
                if bus_names[j].startswith("TAP_")]
        sections.append(Section(f"{a}-{b}", a, b, bus_km[a], bus_km[b],
                                first_line=ia, last_line=ib - 1,
                                from_idx=ia, to_idx=ib, taps=taps))

    candidates = [(n, bus_km[n]) for n in bus_names if n.startswith("TAP_")]
    return GridMeta(
        profile=profile, bus_names=bus_names, bus_km=bus_km, metered=metered,
        sections=sections,
        nominal_load_mw={l.bus: float(l.p_mw) for l in topo.loads},
        x_ohm_per_km=float(topo.lines.x_ohm_per_km), base_kv=float(topo.base_kv),
        candidates=candidates,
    )


def metered_channels(meta: GridMeta) -> list[str]:
    """All m_* columns that are physically metered under this profile."""
    cols = [f"m_{b}_{k}" for b in meta.metered for k in ("vm", "va", "p", "q")]
    for i, name in enumerate(meta.bus_names):
        if name not in meta.metered:
            continue
        if i < len(meta.bus_names) - 1:              # line i starts here
            cols += [f"m_LINE{i}_p_from", f"m_LINE{i}_q_from"]
        if i > 0:                                    # line i-1 ends here
            cols += [f"m_LINE{i-1}_p_to", f"m_LINE{i-1}_q_to"]
    return cols


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def _expected(baseline_df: pd.DataFrame, col: str, ls: np.ndarray) -> np.ndarray:
    bl = baseline_df.sort_values("load_scale")
    return np.interp(ls, bl["load_scale"].to_numpy(), bl[col].to_numpy(dtype=float))


def estimate_load_scale(df: pd.DataFrame, meta: GridMeta) -> tuple[np.ndarray, np.ndarray]:
    """Median and spread of metered-load / nominal-load ratios (theft is unmetered → not included)."""
    ratios = [
        _col(df, f"m_{b}_p") / p_nom
        for b, p_nom in meta.nominal_load_mw.items()
        if b in meta.metered and p_nom > 0
    ]
    if not ratios:
        return np.ones(len(df)), np.zeros(len(df))
    R = np.column_stack(ratios)
    with np.errstate(all="ignore"), _quiet():
        med = np.nanmedian(R, axis=1)
        spread = np.nanmax(R, axis=1) - np.nanmin(R, axis=1)
    med = np.where(np.isfinite(med), med, 1.0)
    return np.clip(med, 0.4, 1.3), np.where(np.isfinite(spread), spread, 0.0)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def build_feature_matrix(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    profile: str = "major_only",
) -> pd.DataFrame:
    """Return a copy of df with f_* feature columns appended. Uses no label columns."""
    meta = get_grid_meta(profile)
    idx = meta.bus_idx
    ls_est, ls_spread = estimate_load_scale(df, meta)
    F: dict[str, np.ndarray] = {
        "f_load_scale_est": ls_est,
        "f_load_scale_spread": ls_spread,
        "f_n_missing": df.reindex(columns=metered_channels(meta)).isna().sum(axis=1).to_numpy(float),
    }

    def exp(col):
        return _expected(baseline_df, col, ls_est)

    # --- Per metered bus ---
    for b in meta.metered:
        i = idx[b]
        F[f"f_vm_dev_{b}"] = _col(df, f"m_{b}_vm") - exp(f"vm_{i}")
        F[f"f_va_dev_{b}"] = _col(df, f"m_{b}_va") - exp(f"va_{i}")
        for k in ("p", "q"):
            s = _col(df, f"m_{b}_{k}").copy()
            if i < len(meta.bus_names) - 1:
                s = s + _col(df, f"m_LINE{i}_{k}_from")
            if i > 0:
                s = s + _col(df, f"m_LINE{i-1}_{k}_to")
            F[f"f_kcl_{k}_{b}"] = s

    # --- Per section ---
    r_total = np.zeros(len(df))
    for s in meta.sections:
        a, b, la, lb = s.from_bus, s.to_bus, s.first_line, s.last_line
        p_in = _col(df, f"m_LINE{la}_p_from")
        p_in_exp = exp(f"p_from_line{la}")
        r_p = (p_in + _col(df, f"m_LINE{lb}_p_to")) - (p_in_exp + exp(f"p_to_line{lb}"))
        r_q = (_col(df, f"m_LINE{la}_q_from") + _col(df, f"m_LINE{lb}_q_to")) \
            - (exp(f"q_from_line{la}") + exp(f"q_to_line{lb}"))
        dva = (_col(df, f"m_{a}_va") - _col(df, f"m_{b}_va")) \
            - (exp(f"va_{s.from_idx}") - exp(f"va_{s.to_idx}"))
        dvm = (_col(df, f"m_{a}_vm") - _col(df, f"m_{b}_vm")) \
            - (exp(f"vm_{s.from_idx}") - exp(f"vm_{s.to_idx}"))

        # Δθ[deg] ≈ (X'/V²)·(180/π)·MW·km  →  convert angle residual to MW·L units
        with _quiet():
            vm_avg = np.nanmean(np.column_stack([_col(df, f"m_{a}_vm"), _col(df, f"m_{b}_vm")]), axis=1)
        vm_avg = np.where(np.isfinite(vm_avg) & (vm_avg > 0.5) & (vm_avg < 1.5), vm_avg, 1.0)
        k_deg = meta.x_ohm_per_km / (meta.base_kv * vm_avg) ** 2 * (180.0 / math.pi)
        dva_mw = dva / (k_deg * s.length_km)
        with np.errstate(all="ignore"):
            xfrac = np.where(np.abs(r_p) >= 1.0, dva_mw / r_p, np.nan)

        n = s.name
        F[f"f_r_p_{n}"] = r_p
        F[f"f_r_q_{n}"] = r_q
        F[f"f_dp_in_{n}"] = p_in - p_in_exp
        F[f"f_dva_{n}"] = dva
        F[f"f_dvm_{n}"] = dvm
        F[f"f_dva_mw_{n}"] = dva_mw
        F[f"f_xfrac_{n}"] = np.clip(xfrac, -2.0, 3.0)
        r_total = r_total + np.nan_to_num(r_p)
    F["f_r_p_total"] = r_total

    out = df.copy()
    feat_df = pd.DataFrame(F, index=df.index)
    out = pd.concat([out.drop(columns=[c for c in feat_df.columns if c in out.columns]), feat_df], axis=1)
    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted list of f_* feature column names."""
    return sorted(c for c in df.columns if c.startswith("f_"))
