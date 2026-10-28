"""
simdata/splits.py — Train/val/test split with group integrity + leave-location-out.

Rules (from 02_SYNTHETIC_DATA.md)
----------------------------------
1. Split is by group_id — no group appears in >1 split.
2. Ratio: 70 / 15 / 15 (train / val / test).
3. Leave-location-out: theft rows with true_km within ±5 km of any
   holdout_locations_km go exclusively to test_loc_holdout split.
4. Save split indices to splits.json.

Public API
----------
make_splits(df, holdout_locations_km, seed) -> dict[str, list[int]]
save_splits(splits, out_path)
load_splits(path) -> dict[str, list[int]]
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


_HOLDOUT_RADIUS_KM = 5.0   # ± km around each holdout location


def make_splits(
    df: pd.DataFrame,
    holdout_locations_km: list[float],
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    # test_frac = 1 - train - val = 0.15
) -> dict[str, list[int]]:
    """
    Produce leak-free train / val / test / test_loc_holdout splits.

    Parameters
    ----------
    df                   : full dataset DataFrame (index = row position)
    holdout_locations_km : from scenarios.yaml — theft km values held out
    seed                 : random seed for group shuffle
    train_frac / val_frac: target fractions (test = remainder)

    Returns
    -------
    dict with keys: "train", "val", "test", "test_loc_holdout"
    Each value is a list of integer row indices into df.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)

    # --- Step 1: Mark leave-location-out rows ---
    def _is_holdout_location(km) -> bool:
        if km is None or (isinstance(km, float) and math.isnan(km)):
            return False
        return any(abs(km - h) <= _HOLDOUT_RADIUS_KM for h in holdout_locations_km)

    loc_holdout_mask = df["true_km"].apply(_is_holdout_location)
    loc_holdout_idx  = df.index[loc_holdout_mask].tolist()

    # These rows are removed from the main split pool
    remaining_df = df[~loc_holdout_mask].copy()

    # --- Step 2: Group-level split on remaining rows ---
    group_ids = remaining_df["group_id"].unique().tolist()
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train  = int(n_groups * train_frac)
    n_val    = int(n_groups * val_frac)

    train_groups = set(group_ids[:n_train])
    val_groups   = set(group_ids[n_train:n_train + n_val])
    test_groups  = set(group_ids[n_train + n_val:])

    train_idx = remaining_df.index[remaining_df["group_id"].isin(train_groups)].tolist()
    val_idx   = remaining_df.index[remaining_df["group_id"].isin(val_groups)].tolist()
    test_idx  = remaining_df.index[remaining_df["group_id"].isin(test_groups)].tolist()

    splits = {
        "train":            sorted(train_idx),
        "val":              sorted(val_idx),
        "test":             sorted(test_idx),
        "test_loc_holdout": sorted(loc_holdout_idx),
    }

    return splits


def save_splits(splits: dict[str, list[int]], out_path: Path) -> None:
    """Persist splits as JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved: {out_path}")
    for name, idx in splits.items():
        print(f"  {name:<20} {len(idx):6d} rows")


def load_splits(path: Path) -> dict[str, list[int]]:
    """Load splits from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def validate_splits(df: pd.DataFrame, splits: dict[str, list[int]]) -> None:
    """
    Assert split integrity:
    - No index appears in >1 split
    - No group_id appears in >1 of train/val/test
    Raises AssertionError on violation.
    """
    all_keys   = ["train", "val", "test", "test_loc_holdout"]
    all_indices = []
    for k in all_keys:
        all_indices.extend(splits.get(k, []))

    assert len(all_indices) == len(set(all_indices)), \
        "Duplicate row indices across splits!"

    # Group-level check (excluding holdout which may share groups)
    for split_a in ["train", "val", "test"]:
        for split_b in ["train", "val", "test"]:
            if split_a >= split_b:
                continue
            groups_a = set(df.loc[splits[split_a], "group_id"])
            groups_b = set(df.loc[splits[split_b], "group_id"])
            overlap  = groups_a & groups_b
            assert not overlap, \
                f"Group leakage: {len(overlap)} groups shared between {split_a} and {split_b}!"
