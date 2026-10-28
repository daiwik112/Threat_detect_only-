"""
simdata/profiles.py — Generate the same scenarios under all sensor profiles.

Used for the sensor-placement study (WF3 step 6 / WF4 endpoint).
Runs generate_dataset for each profile using the SAME seed so results
are directly comparable across profiles.

Public API
----------
generate_all_profiles(scenarios_cfg, topology, sensors_cfg, out_dir, seed, n_jobs)
    Writes dataset_major_only.parquet, dataset_major_plus_mid.parquet,
    dataset_dense.parquet to out_dir.
    Returns dict[profile_name -> DataFrame].
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from grid.config import ScenariosConfig, TopologyConfig, SensorsConfig
from simdata.generate import generate_dataset


_ALL_PROFILES = ["major_only", "major_plus_mid", "dense"]


def generate_all_profiles(
    scenarios_cfg: ScenariosConfig,
    topology: TopologyConfig,
    sensors_cfg: SensorsConfig,
    out_dir: Path,
    seed: int = 42,
    n_jobs: int = 1,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Generate dataset.parquet for each sensor profile.
    Same operating points and scenario seeds are used across profiles
    so measurement differences are due to sensor coverage only.

    Returns dict: profile_name -> DataFrame
    """
    results: dict[str, pd.DataFrame] = {}
    t0 = time.time()

    for profile in _ALL_PROFILES:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Profile: {profile}")
            print(f"{'='*60}")
        df = generate_dataset(
            scenarios_cfg=scenarios_cfg,
            topology=topology,
            sensors_cfg=sensors_cfg,
            profile=profile,
            out_dir=Path(out_dir),
            seed=seed,          # same seed → same operating points, different measurements
            n_jobs=n_jobs,
            verbose=verbose,
        )
        results[profile] = df

    if verbose:
        print(f"\nAll profiles done in {time.time() - t0:.1f}s")
        print("Files written:")
        for profile in _ALL_PROFILES:
            p = Path(out_dir) / f"dataset_{profile}.parquet"
            if p.exists():
                df = results[profile]
                print(f"  {p.name}  {df.shape[0]} rows × {df.shape[1]} cols")

    return results
