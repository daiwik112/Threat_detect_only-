"""
grid/baseline.py — Sweep load scales and save no-theft expected states.

Public API
----------
generate_baseline(topology, n_points=200, load_range=(0.5, 1.15),
                  out_path=None, rng_seed=42) -> pd.DataFrame
    Sweeps load_scale linearly, runs AC PF for each, collects vm/va/line
    results, and saves to `data/raw/baseline_states.parquet`.

    Returns the DataFrame for immediate use.

CLI
---
python -m grid.baseline --n 200 --out data/raw/baseline_states.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from grid.build import build_net
from grid.config import TopologyConfig, load_topology
from grid.flow import run_pf, PFNonConvergence


_DEFAULT_OUT = (
    Path(__file__).parent.parent / "data" / "raw" / "baseline_states.parquet"
)


# ---------------------------------------------------------------------------
# Core sweep function
# ---------------------------------------------------------------------------


def generate_baseline(
    topology: Optional[TopologyConfig] = None,
    n_points: int = 200,
    load_range: tuple[float, float] = (0.5, 1.15),
    out_path: Optional[Path] = None,
    rng_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep load_scale in [load_range[0], load_range[1]] (n_points steps),
    run AC PF for each (no theft), and return a flat DataFrame.

    Columns
    -------
    load_scale        : float
    gen_p_mw          : float
    load_p_mw         : float
    total_loss_mw     : float
    loss_pct          : float
    min_vm_pu         : float
    max_vm_pu         : float
    max_line_loading  : float
    vm_<bus_idx>      : float  (one per bus)
    va_<bus_idx>      : float  (one per bus)
    p_from_line<idx>  : float  (one per line)
    p_to_line<idx>    : float
    q_from_line<idx>  : float
    q_to_line<idx>    : float
    loading_line<idx> : float
    pl_line<idx>      : float  (segment loss MW)
    converged         : bool
    """
    if topology is None:
        topology = load_topology()

    load_scales = np.linspace(load_range[0], load_range[1], n_points)
    rows: list[dict] = []

    n_failed = 0
    t0 = time.time()

    for i, ls in enumerate(load_scales):
        try:
            net = build_net(topology=topology, load_scale=float(ls), theft=None)
            res = run_pf(net)
        except PFNonConvergence:
            n_failed += 1
            if verbose:
                print(f"  [WARN] load_scale={ls:.3f}: PF did not converge — skipping")
            continue
        except Exception as exc:
            n_failed += 1
            if verbose:
                print(f"  [WARN] load_scale={ls:.3f}: {exc} — skipping")
            continue

        row: dict = {
            "load_scale": float(ls),
            "gen_p_mw": res["gen_p_mw"],
            "load_p_mw": res["load_p_mw"],
            "total_loss_mw": res["total_loss_mw"],
            "loss_pct": (
                100.0 * res["total_loss_mw"] / res["gen_p_mw"]
                if res["gen_p_mw"] else float("nan")
            ),
            "min_vm_pu": res["min_vm_pu"],
            "max_vm_pu": res["max_vm_pu"],
            "max_line_loading": res["max_line_loading_pct"],
            "converged": True,
        }

        # Per-bus voltage
        for bus_idx, vm in res["vm_pu"].items():
            row[f"vm_{bus_idx}"] = vm
        for bus_idx, va in res["va_deg"].items():
            row[f"va_{bus_idx}"] = va

        # Per-line flows
        for line_idx, val in res["line_p_from_mw"].items():
            row[f"p_from_line{line_idx}"] = val
        for line_idx, val in res["line_p_to_mw"].items():
            row[f"p_to_line{line_idx}"] = val
        for line_idx, val in res["line_q_from_mvar"].items():
            row[f"q_from_line{line_idx}"] = val
        for line_idx, val in res["line_q_to_mvar"].items():
            row[f"q_to_line{line_idx}"] = val
        for line_idx, val in res["line_loading_pct"].items():
            row[f"loading_line{line_idx}"] = val
        for line_idx, val in res["line_pl_mw"].items():
            row[f"pl_line{line_idx}"] = val

        rows.append(row)

        if verbose and (i % 20 == 0 or i == n_points - 1):
            elapsed = time.time() - t0
            print(
                f"  [{i+1:3d}/{n_points}] ls={ls:.3f}  "
                f"gen={res['gen_p_mw']:.1f} MW  "
                f"loss={res['total_loss_mw']:.2f} MW  "
                f"vm=[{res['min_vm_pu']:.4f},{res['max_vm_pu']:.4f}]  "
                f"elapsed={elapsed:.1f}s"
            )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("All power flows failed — check topology/parameters.")

    # Save to parquet
    save_path = out_path or _DEFAULT_OUT
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)

    if verbose:
        elapsed = time.time() - t0
        print(
            f"\nBaseline sweep complete: {len(df)}/{n_points} converged "
            f"({n_failed} failed) in {elapsed:.1f}s"
        )
        print(f"Saved to: {save_path}")
        print(f"  load_scale range: {df['load_scale'].min():.3f} – {df['load_scale'].max():.3f}")
        print(f"  gen MW range:     {df['gen_p_mw'].min():.1f} – {df['gen_p_mw'].max():.1f}")
        print(f"  loss % range:     {df['loss_pct'].min():.2f} – {df['loss_pct'].max():.2f} %")
        print(f"  vm_pu range:      {df['min_vm_pu'].min():.4f} – {df['max_vm_pu'].max():.4f}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate no-theft baseline parquet by sweeping load scale."
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Number of load-scale points to sweep (default: 200)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help=f"Output parquet path (default: {_DEFAULT_OUT})"
    )
    parser.add_argument(
        "--load-min", type=float, default=0.5,
        help="Minimum load scale (default: 0.5)"
    )
    parser.add_argument(
        "--load-max", type=float, default=1.15,
        help="Maximum load scale (default: 1.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (currently unused; reserved for future noise injection)"
    )
    args = parser.parse_args()

    print(f"Generating baseline: {args.n} points, "
          f"load_scale in [{args.load_min}, {args.load_max}]")

    df = generate_baseline(
        n_points=args.n,
        load_range=(args.load_min, args.load_max),
        out_path=Path(args.out) if args.out else None,
        rng_seed=args.seed,
        verbose=True,
    )
    print(f"\nDataFrame shape: {df.shape}")


if __name__ == "__main__":
    _cli()
