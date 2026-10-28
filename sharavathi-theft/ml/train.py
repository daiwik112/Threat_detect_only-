"""
ml/train.py — WF3 training pipeline CLI.

Usage
-----
python -m ml.train                      # features → baseline → XGB → localizers → (GNN) → metrics.json
python -m ml.train --tune               # Optuna tuning of the binary head (≤50 trials, needs optuna)
python -m ml.train --eval-only          # re-evaluate saved models
python -m ml.train --sensor-study       # also run Step 6 (needs data/processed/study/*)
python -m ml.train --skip-gnn           # skip Step 5 (GNN, 3 seeds — the slow part)

Artifacts (data/models/)
------------------------
baseline_detector.joblib, xgb_classifier.joblib, localizer_xgb_reg.joblib,
localizer_section_rank.joblib, gnn.pt + gnn_results.json, feat_cols.json, model_meta.json, metrics.json,
eval_*.png, eval_arrays.json, (sensor_study.json)
Every joblib model carries `.artifact_meta` = {version, dataset_hash, profile, seed, trained_at}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VERSION = "wf3-v2"
_MODEL_DIR = Path("data/models")
_DATA_DIR = Path("data/processed")
_RAW_DIR = Path("data/raw")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def split_frames(df, splits: dict):
    """splits.json holds positional row indices into the parquet (WF2)."""
    return {k: df.iloc[v] for k, v in splits.items()}


def _cli() -> None:
    ap = argparse.ArgumentParser(description="WF3: train ML models for theft detection.")
    ap.add_argument("--profile", default="major_only",
                    choices=["major_only", "major_plus_mid", "dense"])
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--sensor-study", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-gnn", action="store_true")
    ap.add_argument("--gnn-threads", type=int, default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    if hasattr(sys.stdout, "reconfigure"):          # Windows cp1252 consoles choke on ≤ ± …
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import joblib
    import pandas as pd
    from ml.evaluate import evaluate_all
    from ml.features import build_feature_matrix, get_feature_cols

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    parquet = _DATA_DIR / f"dataset_{args.profile}.parquet"
    if not parquet.exists():
        sys.exit(f"[ERROR] {parquet} not found. Run WF2 first: python -m simdata.generate")
    bl_path = _RAW_DIR / "baseline_states.parquet"
    if not bl_path.exists():
        sys.exit("[ERROR] data/raw/baseline_states.parquet not found. Run: make grid-baseline")

    t0 = time.time()
    df = pd.read_parquet(parquet)
    baseline_df = pd.read_parquet(bl_path)
    splits = json.loads((_DATA_DIR / "splits.json").read_text())
    ds_hash = file_hash(parquet)
    print(f"Dataset {parquet} {df.shape}  hash={ds_hash}")

    print("\n[Step 1] Features …")
    df = build_feature_matrix(df, baseline_df, args.profile)
    feat_cols = get_feature_cols(df)
    (_MODEL_DIR / "feat_cols.json").write_text(json.dumps(feat_cols, indent=1))
    df.index = range(len(df))              # positional == label index for splits
    S = split_frames(df, splits)
    print(f"  {len(feat_cols)} features | " + " | ".join(f"{k}: {len(v)}" for k, v in S.items()))

    stamp = {"version": VERSION, "dataset_hash": ds_hash, "profile": args.profile,
             "seed": args.seed, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def save(obj, name):
        obj.artifact_meta = stamp
        joblib.dump(obj, _MODEL_DIR / f"{name}.joblib")
        print(f"  saved {name}.joblib")

    if not args.eval_only:
        from ml.detector_baseline import BaselineDetector
        from ml.localize import LocalizerSectionRank, LocalizerXGBReg
        from ml.xgb_classifier import XGBClassifier

        print("\n[Step 2] Baseline residual detector …")
        save(BaselineDetector(percentile=99.5).fit(S["train"]), "baseline_detector")

        print("\n[Step 3] XGBoost classifier …")
        clf = XGBClassifier(seed=args.seed).fit(S["train"], S["val"], feat_cols, tune=args.tune)
        save(clf, "xgb_classifier")

        print("\n[Step 4] Localizers (theft rows only) …")
        reg = LocalizerXGBReg(seed=args.seed, profile=args.profile).fit(S["train"], S["val"], feat_cols)
        print(f"  xgb_reg conformal 80% radius: ±{reg.conformal_radius:.1f} km")
        save(reg, "localizer_xgb_reg")
        rank = LocalizerSectionRank(seed=args.seed, profile=args.profile).fit(S["train"], S["val"], feat_cols)
        save(rank, "localizer_section_rank")

        status = _gnn_status()
        if status.startswith("available") and not args.skip_gnn:
            from ml.gnn.train import train_gnn
            train_gnn(df, splits, baseline_df, profile=args.profile,
                      n_threads=args.gnn_threads, stamp=stamp)

    print("\n[Step 7] Evaluation …")
    M = evaluate_all(df, splits, feat_cols, baseline_df, _MODEL_DIR, _MODEL_DIR, args.profile)
    M["artifact_meta"] = stamp
    (_MODEL_DIR / "metrics.json").write_text(json.dumps(M, indent=2, default=lambda x: None))

    meta = {**stamp, "n_features": len(feat_cols),
            "selected_localizer": M.get("selected_localizer"),
            "gnn": _gnn_status(quiet=True)}
    (_MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    _print_summary(M)

    if args.sensor_study:
        print("\n[Step 6] Sensor placement study …")
        from ml.sensor_study import run_sensor_study
        run_sensor_study(out_dir=_MODEL_DIR, seed=args.seed, stamp=stamp)

    print(f"\n[WF3 DONE] {time.time() - t0:.0f}s")


def _gnn_status(quiet: bool = False) -> str:
    """Step 5 is optional and only runs when torch + torch_geometric are installed."""
    try:
        from grid.config import load_feature_flags
        if not load_feature_flags().gnn_enabled:
            msg = "disabled (feature_flags.gnn_enabled=false)"
            if not quiet:
                print(f"\n[Step 5] GNN {msg}")
            return msg
    except Exception:
        pass
    from ml.gnn.model import is_gnn_available
    if not is_gnn_available():
        msg = "skipped: torch / torch_geometric not installed"
    else:
        msg = "available"
    if not quiet:
        print(f"\n[Step 5] GNN {msg}")
    return msg


def _print_summary(M: dict) -> None:
    f = lambda v, p=3: "n/a" if v is None else f"{v:.{p}f}"
    print("\n" + "=" * 64 + "\nEVALUATION SUMMARY (held-out)\n" + "=" * 64)
    d = M.get("detection_detail", {}).get("test", {})
    for name in ("baseline", "xgb", "baseline_theft_vs_normal", "xgb_theft_vs_normal"):
        m = d.get(name)
        if m:
            print(f"{name:<26} P={f(m['precision'])} R={f(m['recall'])} F1={f(m['f1'])} "
                  f"FPR={f(m['fpr'])} AUC={f(m.get('roc_auc'))} R@1%FPR={f(m.get('recall_at_fpr_1pct'))}")
    c = M.get("classification", {}).get("per_class_recall", {})
    if c:
        print("per-class recall: " + ", ".join(f"{k}={f(v, 2)}" for k, v in c.items()))
    for split in ("test", "holdout"):
        for name, m in M.get("localization_detail", {}).get(split, {}).items():
            if m:
                print(f"loc[{split:<7}] {name:<17} MAE={f(m['mae_km'], 1)} km  "
                      f"median={f(m['median_km'], 1)}  ≤10km={f(m['within_10km'], 2)}  "
                      f"≤25km={f(m['within_25km'], 2)}  sec_acc={f(m.get('section_accuracy'), 2)}")
    print(f"selected localizer: {M.get('selected_localizer')}")
    print(f"beats baseline: {M.get('beats_baseline')}")
    print(f"\n{M.get('honest_claim', '')}")


if __name__ == "__main__":
    _cli()
