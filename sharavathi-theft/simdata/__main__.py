"""
simdata/__main__.py — CLI entry point for WF2.

Usage
-----
python -m simdata.generate --profile major_only --out data/processed/ --seed 42
python -m simdata.generate --all-profiles --seed 42
python -m simdata.generate --splits-only --profile major_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="WF2: Generate synthetic Sharavathi corridor dataset."
    )
    parser.add_argument(
        "--profile", default="major_only",
        choices=["major_only", "major_plus_mid", "dense"],
        help="Sensor profile to use (default: major_only)",
    )
    parser.add_argument(
        "--all-profiles", action="store_true",
        help="Generate dataset for ALL sensor profiles (ignores --profile)",
    )
    parser.add_argument(
        "--out", default="data/processed/",
        help="Output directory (default: data/processed/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed (default: 42)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel jobs via joblib (default: 1 = serial)",
    )
    parser.add_argument(
        "--splits-only", action="store_true",
        help="Only compute splits from existing parquet (no new simulation)",
    )
    parser.add_argument(
        "--config", default="config/scenarios.yaml",
        help="Path to scenarios.yaml",
    )
    args = parser.parse_args()

    from grid.config import load_topology, load_sensors, load_scenarios

    topology     = load_topology()
    sensors_cfg  = load_sensors()
    scenarios_cfg = load_scenarios(Path(args.config))
    out_dir      = Path(args.out)

    if args.splits_only:
        # Load existing parquet and recompute splits
        import pandas as pd
        from simdata.splits import make_splits, save_splits, validate_splits
        from simdata.dataset_card import write_dataset_card

        parquet_path = out_dir / f"dataset_{args.profile}.parquet"
        if not parquet_path.exists():
            print(f"[ERROR] {parquet_path} not found — run without --splits-only first.")
            sys.exit(1)

        df = pd.read_parquet(parquet_path)
        splits = make_splits(df, scenarios_cfg.holdout_locations_km, seed=args.seed)
        validate_splits(df, splits)
        save_splits(splits, out_dir / "splits.json")
        write_dataset_card(df, splits, out_dir / "dataset_card.md",
                           args.profile, sensors_cfg, scenarios_cfg)
        return

    if args.all_profiles:
        from simdata.profiles import generate_all_profiles
        dfs = generate_all_profiles(
            scenarios_cfg=scenarios_cfg,
            topology=topology,
            sensors_cfg=sensors_cfg,
            out_dir=out_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            verbose=True,
        )
        # Splits + card for primary profile
        primary = "major_only"
        df = dfs[primary]
    else:
        from simdata.generate import generate_dataset
        df = generate_dataset(
            scenarios_cfg=scenarios_cfg,
            topology=topology,
            sensors_cfg=sensors_cfg,
            profile=args.profile,
            out_dir=out_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            verbose=True,
        )
        primary = args.profile

    # --- Splits ---
    from simdata.splits import make_splits, save_splits, validate_splits
    from simdata.dataset_card import write_dataset_card

    splits = make_splits(df, scenarios_cfg.holdout_locations_km, seed=args.seed)
    validate_splits(df, splits)
    save_splits(splits, out_dir / "splits.json")

    # --- Dataset card ---
    write_dataset_card(
        df, splits,
        out_path=out_dir / "dataset_card.md",
        profile=primary,
        sensors_cfg=sensors_cfg,
        scenarios_cfg=scenarios_cfg,
    )

    print("\n[WF2 DONE] dataset.parquet + splits.json + dataset_card.md written.")


if __name__ == "__main__":
    _cli()
