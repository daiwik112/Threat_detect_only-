"""
grid/viz_check.py — Quick diagnostic plots for the Sharavathi corridor.

Plots
-----
1. Voltage profile vs km (all buses + tap buses)
2. Line loading % vs km of segment midpoint
3. Active power loss vs load_scale (from baseline parquet, if available)
4. Comparison: no-theft vs theft voltage profile

Public API
----------
plot_voltage_profile(net, results, save_path=None)
plot_line_loading(net, results, save_path=None)
plot_baseline_losses(baseline_df, save_path=None)
plot_theft_comparison(net_clean, res_clean, net_theft, res_theft, theft_km, save_path=None)
plot_all(net, results, baseline_df=None, save_path_prefix=None)

CLI
---
python -m grid.viz_check [--theft-km 150 --theft-mw 20] [--save figures/]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless-safe backend; swap to "TkAgg" locally if needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import pandapower as pp


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
_PALETTE = {
    "primary": "#2563EB",
    "secondary": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#6B7280",
    "bg": "#0F172A",
    "surface": "#1E293B",
    "text": "#F1F5F9",
    "grid": "#334155",
}

plt.rcParams.update({
    "figure.facecolor": _PALETTE["bg"],
    "axes.facecolor": _PALETTE["surface"],
    "axes.edgecolor": _PALETTE["grid"],
    "axes.labelcolor": _PALETTE["text"],
    "axes.titlecolor": _PALETTE["text"],
    "xtick.color": _PALETTE["text"],
    "ytick.color": _PALETTE["text"],
    "grid.color": _PALETTE["grid"],
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "legend.facecolor": _PALETTE["surface"],
    "legend.edgecolor": _PALETTE["grid"],
    "legend.labelcolor": _PALETTE["text"],
    "text.color": _PALETTE["text"],
    "font.family": "DejaVu Sans",
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bus_km_map(net: pp.pandapowerNet) -> dict[int, float]:
    """Return {bus_idx: km} using geodata column (y = km in our encoding)."""
    km_map: dict[int, float] = {}
    for idx, row in net.bus.iterrows():
        geo = row.get("geodata", None)
        if geo is not None and len(geo) >= 2:
            km_map[idx] = geo[1]
    return km_map


def _sorted_by_km(net, results, key) -> tuple[list[float], list[float]]:
    """Return (km_list, val_list) sorted by km for bus-level data."""
    km_map = _bus_km_map(net)
    data = results.get(key, {})
    pairs = [(km_map[idx], val) for idx, val in data.items() if idx in km_map]
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ---------------------------------------------------------------------------
# Plot 1: Voltage profile
# ---------------------------------------------------------------------------


def plot_voltage_profile(
    net: pp.pandapowerNet,
    results: dict,
    *,
    label: str = "Base case",
    color: str | None = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Voltage magnitude (pu) vs km along corridor."""
    own_fig = ax is None
    fig = None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("Voltage Profile — Sharavathi 220 kV Corridor",
                     fontsize=14, fontweight="bold", color=_PALETTE["text"])

    km_list, vm_list = _sorted_by_km(net, results, "vm_pu")
    c = color or _PALETTE["primary"]
    ax.plot(km_list, vm_list, "-o", color=c, linewidth=2, markersize=4, label=label)

    # 0.95/1.05 bands
    ax.axhline(0.95, color=_PALETTE["danger"], linestyle="--", linewidth=1,
               alpha=0.7, label="0.95 limit")
    ax.axhline(1.05, color=_PALETTE["warning"], linestyle="--", linewidth=1,
               alpha=0.7, label="1.05 limit")

    # Label substation positions
    km_map = _bus_km_map(net)
    for idx, row in net.bus.iterrows():
        name = row["name"]
        if not name.startswith("TAP_") and idx in km_map:
            km_val = km_map[idx]
            vm_val = results["vm_pu"].get(idx, None)
            if vm_val is not None:
                ax.annotate(name, (km_val, vm_val),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=7, color=_PALETTE["text"], ha="center",
                            arrowprops=dict(arrowstyle="-", color=_PALETTE["grid"]))

    ax.set_xlabel("Distance from Sharavathi (km)")
    ax.set_ylabel("Voltage Magnitude (pu)")
    ax.set_ylim(0.90, 1.08)
    ax.grid(True)
    ax.legend(loc="lower left")

    if own_fig:
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=_PALETTE["bg"])
            print(f"Saved: {save_path}")
        if show:
            plt.show()
    return fig or ax.figure


# ---------------------------------------------------------------------------
# Plot 2: Line loading %
# ---------------------------------------------------------------------------


def plot_line_loading(
    net: pp.pandapowerNet,
    results: dict,
    *,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Line loading % vs midpoint km of each segment."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Line Loading — Sharavathi 220 kV Corridor",
                 fontsize=14, fontweight="bold")

    km_map = _bus_km_map(net)
    section_map: dict = net.get("section_map", {})

    xs, ys, labels = [], [], []
    for line_idx, loading in results["line_loading_pct"].items():
        if line_idx in section_map:
            from_km, to_km, from_name, to_name = section_map[line_idx]
        else:
            # Fall back to bus geodata
            row = net.line.iloc[line_idx] if line_idx < len(net.line) else None
            if row is None:
                continue
            from_km = km_map.get(row["from_bus"], 0)
            to_km   = km_map.get(row["to_bus"], 0)
            from_name, to_name = str(row["from_bus"]), str(row["to_bus"])
        mid_km = (from_km + to_km) / 2
        xs.append(mid_km)
        ys.append(loading)
        labels.append(f"{from_name}→{to_name}")

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]

    colors = [
        _PALETTE["danger"] if y > 80 else
        _PALETTE["warning"] if y > 60 else
        _PALETTE["secondary"]
        for y in ys
    ]
    ax.bar(xs, ys, width=7, color=colors, edgecolor=_PALETTE["grid"], alpha=0.8)
    ax.axhline(80, color=_PALETTE["danger"], linestyle="--", linewidth=1.5,
               label="80% warning")
    ax.set_xlabel("Segment midpoint (km)")
    ax.set_ylabel("Loading (%)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=_PALETTE["bg"])
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Baseline losses vs load scale
# ---------------------------------------------------------------------------


def plot_baseline_losses(
    baseline_df: pd.DataFrame,
    *,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Active-power loss (MW and %) vs load scale from baseline parquet."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Baseline Losses vs Load Scale (No Theft)",
                 fontsize=14, fontweight="bold")

    df = baseline_df.sort_values("load_scale")

    ax1.plot(df["load_scale"], df["total_loss_mw"],
             color=_PALETTE["primary"], linewidth=2)
    ax1.fill_between(df["load_scale"], df["total_loss_mw"],
                     alpha=0.2, color=_PALETTE["primary"])
    ax1.set_xlabel("Load Scale")
    ax1.set_ylabel("Total Loss (MW)")
    ax1.set_title("Absolute Losses")
    ax1.grid(True)

    ax2.plot(df["load_scale"], df["loss_pct"],
             color=_PALETTE["secondary"], linewidth=2)
    ax2.axhspan(1.0, 4.0, alpha=0.1, color=_PALETTE["secondary"],
                label="Expected 1–4% band")
    ax2.fill_between(df["load_scale"], df["loss_pct"],
                     alpha=0.2, color=_PALETTE["secondary"])
    ax2.set_xlabel("Load Scale")
    ax2.set_ylabel("Loss (%)")
    ax2.set_title("Loss as % of Generation")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=_PALETTE["bg"])
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Theft comparison
# ---------------------------------------------------------------------------


def plot_theft_comparison(
    net_clean: pp.pandapowerNet,
    res_clean: dict,
    net_theft: pp.pandapowerNet,
    res_theft: dict,
    theft_km: float,
    *,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Overlay no-theft and theft voltage profiles; mark theft location."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Theft Impact — 20 MW Extraction at km {theft_km:.0f}",
                 fontsize=14, fontweight="bold")

    plot_voltage_profile(net_clean, res_clean,
                         label="No theft", color=_PALETTE["secondary"], ax=ax)
    plot_voltage_profile(net_theft, res_theft,
                         label=f"Theft {theft_km:.0f} km", color=_PALETTE["danger"], ax=ax)

    # Mark theft location
    ax.axvline(theft_km, color=_PALETTE["warning"], linestyle="--", linewidth=2,
               label=f"Theft injection @ km {theft_km:.0f}")
    ax.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=_PALETTE["bg"])
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Convenience: plot everything
# ---------------------------------------------------------------------------


def plot_all(
    net: pp.pandapowerNet,
    results: dict,
    *,
    baseline_df: Optional[pd.DataFrame] = None,
    save_path_prefix: Optional[str] = None,
    show: bool = False,
) -> list[plt.Figure]:
    """Run all available plots and optionally save them."""
    prefix = save_path_prefix or ""
    figs = []
    figs.append(plot_voltage_profile(
        net, results,
        save_path=Path(f"{prefix}voltage_profile.png") if prefix else None,
        show=show,
    ))
    figs.append(plot_line_loading(
        net, results,
        save_path=Path(f"{prefix}line_loading.png") if prefix else None,
        show=show,
    ))
    if baseline_df is not None:
        figs.append(plot_baseline_losses(
            baseline_df,
            save_path=Path(f"{prefix}baseline_losses.png") if prefix else None,
            show=show,
        ))
    return figs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for Sharavathi corridor."
    )
    parser.add_argument("--theft-km", type=float, default=None)
    parser.add_argument("--theft-mw", type=float, default=20.0)
    parser.add_argument("--theft-pf", type=float, default=0.9)
    parser.add_argument("--load-scale", type=float, default=1.0)
    parser.add_argument("--save", type=str, default="figures/",
                        help="Directory prefix for saved figures")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive matplotlib window")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline_states.parquet")
    args = parser.parse_args()

    from grid.build import build_net, TheftSpec
    from grid.flow import run_pf, PFNonConvergence

    save = args.save.rstrip("/") + "/" if args.save else ""

    print("Building base network …")
    net_clean = build_net(load_scale=args.load_scale)
    res_clean = run_pf(net_clean)
    print(f"  loss={res_clean['total_loss_mw']:.2f} MW  "
          f"vm=[{res_clean['min_vm_pu']:.4f}, {res_clean['max_vm_pu']:.4f}]")

    baseline_df = None
    if args.baseline:
        baseline_df = pd.read_parquet(args.baseline)
    elif Path("data/raw/baseline_states.parquet").exists():
        baseline_df = pd.read_parquet("data/raw/baseline_states.parquet")

    if args.theft_km is not None:
        theft = TheftSpec(km=args.theft_km, mw=args.theft_mw, pf=args.theft_pf)
        net_theft = build_net(load_scale=args.load_scale, theft=theft)
        res_theft = run_pf(net_theft)
        print(f"  theft loss={res_theft['total_loss_mw']:.2f} MW  "
              f"vm=[{res_theft['min_vm_pu']:.4f}, {res_theft['max_vm_pu']:.4f}]")
        plot_theft_comparison(
            net_clean, res_clean,
            net_theft, res_theft,
            theft_km=net_theft["theft_km"],
            save_path=Path(f"{save}theft_comparison.png"),
            show=args.show,
        )

    plot_all(
        net_clean, res_clean,
        baseline_df=baseline_df,
        save_path_prefix=save,
        show=args.show,
    )
    print("Done.")


if __name__ == "__main__":
    _cli()
