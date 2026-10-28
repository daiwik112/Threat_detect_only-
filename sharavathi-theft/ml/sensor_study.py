"""
ml/sensor_study.py — Sensor placement & observability study (Step 6).

For every (profile, noise factor) dataset produced by `python -m simdata.study_datasets`
(data/processed/study/<profile>_noise<k>x/), features are rebuilt for THAT profile and
Steps 3–4 are re-trained from scratch.

Splits: one study split (ml.sensor_study.study_split_ids, WF2 make_splits) shared by all 12 configs,
which contain identical scenarios. Per config → detection recall / FPR (test, theft+normal at the model's 0.5 threshold),
recall at 1 % FPR, localization MAE / median / ≤10 km (holdout, xgb_reg) and
top-3-within-10 km (holdout, section_rank).

Indistinguishability (major_only, noise ×1): for each pair of candidate taps, the distance
between their mean feature vectors, in units of pooled within-tap std, using 15–40 MW thefts
from train. Pairs with distance < 1 are flagged "not separable"; adjacent-tap runs of such
pairs are reported as unresolvable km ranges.

Writes data/models/sensor_study.json  (configs[] matches SensorStudyResponse).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_DIR = Path("data/processed/study")
PROFILES = ["major_only", "major_plus_mid", "dense"]
NOISE_FACTORS = [0.5, 1.0, 2.0, 4.0]


def _study_path(profile: str, k: float) -> Path:
    return STUDY_DIR / f"{profile}_noise{k:g}x" / f"dataset_{profile}.parquet"


def _splits_by_id(main_parquet: Path, splits_path: Path) -> dict[str, set]:
    ids = pd.read_parquet(main_parquet, columns=["scenario_id"])["scenario_id"].to_numpy()
    splits = json.loads(Path(splits_path).read_text())
    return {k: set(ids[v]) for k, v in splits.items()}


STUDY_SPLITS = STUDY_DIR / "splits_study.json"


def study_split_ids(seed: int = 42) -> dict[str, set]:
    """One split shared by all 12 study datasets (they contain identical scenarios).
    Built with WF2's make_splits (group-held-out + leave-location-out) and cached."""
    from grid.config import load_scenarios
    from simdata.splits import make_splits
    ref = _study_path("major_only", 0.5)
    if STUDY_SPLITS.exists():
        cached = json.loads(STUDY_SPLITS.read_text())
        if cached.get("ref_rows") == len(pd.read_parquet(ref, columns=["scenario_id"])):
            return {k: set(v) for k, v in cached["splits"].items()}
    df = pd.read_parquet(ref, columns=["scenario_id", "group_id", "label", "true_km"])
    idx = make_splits(df, load_scenarios().holdout_locations_km, seed=seed)
    ids = df["scenario_id"].to_numpy()
    out = {k: ids[v].tolist() for k, v in idx.items()}
    STUDY_SPLITS.write_text(json.dumps({"ref": str(ref), "ref_rows": len(df), "seed": seed, "splits": out}))
    return {k: set(v) for k, v in out.items()}


def evaluate_config(df: pd.DataFrame, baseline_df: pd.DataFrame, profile: str,
                    split_ids: dict[str, set], seed: int = 42, fast: bool = True) -> dict:
    from ml.evaluate import detection_metrics
    from ml.features import build_feature_matrix, get_feature_cols
    from ml.localize import LocalizerSectionRank, LocalizerXGBReg
    from ml.xgb_classifier import XGBClassifier

    df = build_feature_matrix(df, baseline_df, profile)
    fc = get_feature_cols(df)
    S = {k: df[df["scenario_id"].isin(v)] for k, v in split_ids.items()}
    n_est = 300 if fast else 600

    clf = XGBClassifier(seed=seed, n_estimators=n_est, learning_rate=0.1).fit(S["train"], S["val"], fc)
    t = S["test"][S["test"]["label"].isin(["normal", "theft"])]
    y = (t["label"] != "normal").astype(int).to_numpy()
    det = detection_metrics(y, clf.predict(t), clf.predict_proba_binary(t))

    reg = LocalizerXGBReg(seed=seed, n_estimators=n_est * 2, learning_rate=0.1, profile=profile)
    reg.fit(S["train"], S["val"], fc)
    rank = LocalizerSectionRank(seed=seed, n_estimators=n_est, learning_rate=0.1, profile=profile)
    rank.fit(S["train"], S["val"], fc)
    hold = S["test_loc_holdout"]
    lm, rm = reg.metrics(hold), rank.metrics(hold)
    return {
        "detection_recall": round(det["recall"], 4),
        "fpr": round(det["fpr"], 4),
        "recall_at_fpr_1pct": det.get("recall_at_fpr_1pct"),
        "mae_km": round(lm["mae_km"], 2),
        "median_km": round(lm["median_km"], 2),
        "within_10km": round(lm["within_10km"], 4),
        "mae_km_15_40mw": lm["by_magnitude"]["15_40"]["mae_km"],
        "top3_tap_acc": round(rm["top3_within_10km"], 4),
        "section_rank_mae_km": round(rm["mae_km"], 2),
        "n_features": len(fc),
    }


def indistinguishability(df: pd.DataFrame, baseline_df: pd.DataFrame, train_ids: set,
                         profile: str = "major_only", threshold: float = 1.0) -> dict:
    from ml.features import build_feature_matrix, get_feature_cols
    df = build_feature_matrix(df[df["scenario_id"].isin(train_ids)], baseline_df, profile)
    fc = [c for c in get_feature_cols(df) if c not in ("f_n_missing",)]
    th = df[(df["label"] == "theft") & (df["magnitude_mw"] >= 15)]
    g = th.groupby("true_km")[fc]
    mu, var, n = g.mean(), g.var(), g.size()
    pooled = np.sqrt(((var.mul(n - 1, axis=0)).sum() / (n - 1).sum()).replace(0, np.nan))
    Z = (mu / pooled).fillna(0.0)
    kms = Z.index.to_numpy(dtype=float)
    D = np.sqrt(((Z.to_numpy()[:, None, :] - Z.to_numpy()[None, :, :]) ** 2).mean(axis=2))

    adjacent = [{"km_a": float(kms[i]), "km_b": float(kms[i + 1]),
                 "distance": round(float(D[i, i + 1]), 3),
                 "separable": bool(D[i, i + 1] >= threshold)} for i in range(len(kms) - 1)]
    ranges, cur = [], None
    for p in adjacent:
        if not p["separable"]:
            cur = [cur[0] if cur else p["km_a"], p["km_b"]]
        elif cur:
            ranges.append(cur); cur = None
    if cur:
        ranges.append(cur)
    return {
        "profile": profile, "magnitude_filter_mw": [15, 40], "threshold": threshold,
        "metric": "RMS standardized mean-feature distance (pooled within-tap std)",
        "adjacent_pairs": adjacent,
        "unresolvable_ranges_km": ranges,
        "pair_matrix": {"km": kms.tolist(), "distance": np.round(D, 3).tolist()},
        "n_pairs_not_separable": int(((D < threshold) & ~np.eye(len(kms), dtype=bool)).sum() // 2),
        "n_pairs_total": int(len(kms) * (len(kms) - 1) // 2),
    }


def run_sensor_study(out_dir: Path = Path("data/models"), seed: int = 42,
                     stamp: dict | None = None, verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    baseline_df = pd.read_parquet("data/raw/baseline_states.parquet")
    split_ids = study_split_ids(seed)          # shared by all 12 configs (identical scenarios)
    configs, missing = [], []
    for profile in PROFILES:
        for k in NOISE_FACTORS:
            p = _study_path(profile, k)
            if not p.exists():
                missing.append(str(p))
                continue
            t0 = time.time()
            r = evaluate_config(pd.read_parquet(p), baseline_df, profile, split_ids, seed)
            configs.append({"name": f"{profile}@noise×{k:g}", "profile": profile, "noise_factor": k, **r})
            if verbose:
                print(f"  {profile:<15} noise×{k:<4g} recall={r['detection_recall']:.3f} "
                      f"fpr={r['fpr']:.3f} MAE={r['mae_km']:.1f} km top3={r['top3_tap_acc']:.2f} "
                      f"({time.time() - t0:.0f}s)")
    if missing and verbose:
        print(f"  [WARN] {len(missing)} study datasets missing — run: python -m simdata.study_datasets")

    indist = {}
    src = _study_path("major_only", 1.0)
    if src.exists():
        indist = indistinguishability(pd.read_parquet(src), baseline_df, split_ids["train"])

    out = {"configs": configs, "indistinguishability": indist,
           "missing_datasets": missing, "artifact_meta": stamp or {}}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "sensor_study.json").write_text(json.dumps(out, indent=2, default=lambda x: None))
    if verbose:
        print(f"Saved {Path(out_dir) / 'sensor_study.json'}")
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_sensor_study()
