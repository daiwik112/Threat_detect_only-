"""
simdata/study_datasets.py — Generate datasets for the WF3 sensor-placement study.

Writes data/processed/study/<profile>_noise<k>x/dataset_<profile>.parquet for
every (profile, noise factor) pair. All configs share the same seed, so they contain identical scenarios
(same scenario_ids, same order); ml.sensor_study builds one study split from them.
Does NOT touch data/processed/splits.json or the main dataset.

Usage: python -m simdata.study_datasets [--n-jobs 16] [--skip-existing]
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROFILES      = ["major_only", "major_plus_mid", "dense"]
NOISE_FACTORS = [0.5, 1.0, 2.0, 4.0]
STUDY_DIR     = Path("data/processed/study")


def study_path(profile: str, noise: float) -> Path:
    return STUDY_DIR / f"{profile}_noise{noise:g}x" / f"dataset_{profile}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    from grid.config import load_topology, load_sensors, load_scenarios
    from simdata.generate import generate_dataset

    topology, sensors, scen = load_topology(), load_sensors(), load_scenarios()

    for profile in PROFILES:
        for k in NOISE_FACTORS:
            out = study_path(profile, k)
            if args.skip_existing and out.exists():
                print(f"[skip] {out}")
                continue
            # Every config is generated with the current generator (no copy of the main dataset),
            # so all 12 share identical scenarios and differ only in metering / noise.
            n = sensors.noise
            cfg = sensors.model_copy(update={"noise": n.model_copy(update={
                "vm_pct": n.vm_pct * k, "p_pct": n.p_pct * k,
                "q_pct": n.q_pct * k, "va_deg": n.va_deg * k})})
            print(f"\n=== {profile} noise x{k:g} ===", flush=True)
            generate_dataset(scen, topology, cfg, profile=profile,
                             out_dir=out.parent, seed=args.seed,
                             n_jobs=args.n_jobs, verbose=False)
            print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
