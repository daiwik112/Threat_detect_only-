"""
ml/gnn/train.py — Train/evaluate the GNN over 3 seeds (Step 5).

    python -m ml.gnn.train [--seeds 0 1 2] [--epochs 120] [--threads 8]

Writes
  data/models/gnn.pt            model from the seed with the best validation loss
  data/models/gnn_results.json  per-seed + mean±std metrics on test and test_loc_holdout
The keep/drop decision vs XGBoost is made in ml/evaluate.py (metrics.json → gnn_decision).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_MODEL_DIR = Path("data/models")
_KEYS = ("mae_km", "median_km", "within_10km", "within_25km", "within_50km",
         "top3_within_10km", "section_accuracy", "interval_coverage")


def train_gnn(df: pd.DataFrame, splits: dict, baseline_df: pd.DataFrame,
              seeds=(0, 1, 2), profile: str = "major_only", epochs: int = 120,
              n_threads: int | None = None, out_dir: Path = _MODEL_DIR,
              stamp: dict | None = None, verbose: bool = True) -> dict:
    from ml.evaluate import detection_metrics
    from ml.gnn.model import GNNLocalizer

    S = {k: df.iloc[v] for k, v in splits.items()}
    runs, best = [], None
    for seed in seeds:
        t0 = time.time()
        if verbose:
            print(f"  [GNN] seed {seed}")
        m = GNNLocalizer(seed=seed, profile=profile, epochs=epochs, n_threads=n_threads)
        m.fit(S["train"], S["val"], baseline_df, verbose=verbose)
        r = {"seed": seed, "best_val_loss": m.best_val_loss, "epochs_run": len(m.history),
             "train_seconds": round(time.time() - t0, 1)}
        for name, key in (("test", "test"), ("holdout", "test_loc_holdout")):
            full = m.metrics(S[key])
            r[name] = {k: full.get(k) for k in _KEYS}
            r[name]["by_magnitude"] = full.get("by_magnitude")
        t = S["test"]
        y = (t["label"] != "normal").astype(int).to_numpy()
        p = m.predict_proba_binary(t)
        r["detection_test"] = detection_metrics(y, (p >= 0.5).astype(int), p)
        runs.append(r)
        if verbose:
            print(f"    test MAE {r['test']['mae_km']:.1f} km | holdout MAE {r['holdout']['mae_km']:.1f} km "
                  f"| holdout top3 {r['holdout']['top3_within_10km']:.2f} | {r['train_seconds']:.0f}s")
        if best is None or m.best_val_loss < best[0]:
            best = (m.best_val_loss, seed, m)

    def agg(split, k):
        v = [r[split][k] for r in runs if r[split].get(k) is not None]
        return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else None

    summary = {s: {k: agg(s, k) for k in _KEYS} for s in ("test", "holdout")}
    det = [r["detection_test"] for r in runs]
    summary["detection_test"] = {k: {"mean": float(np.mean([d[k] for d in det])),
                                     "std": float(np.std([d[k] for d in det]))}
                                 for k in ("recall", "fpr", "roc_auc", "recall_at_fpr_1pct")}
    out = {"seeds": list(seeds), "selected_seed": best[1], "runs": runs, "summary": summary,
           "config": best[2].cfg, "artifact_meta": stamp or {}}

    out_dir = Path(out_dir)
    best[2].artifact_meta = stamp or {}
    best[2].save(out_dir / "gnn.pt")
    (out_dir / "gnn_results.json").write_text(json.dumps(out, indent=2, default=lambda x: None))
    if verbose:
        h = summary["holdout"]
        print(f"  [GNN] holdout MAE {h['mae_km']['mean']:.1f} ± {h['mae_km']['std']:.1f} km, "
              f"top3 {h['top3_within_10km']['mean']:.2f} ± {h['top3_within_10km']['std']:.2f} "
              f"→ saved gnn.pt (seed {best[1]})")
    return out


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="major_only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    from ml.train import file_hash
    parquet = Path(f"data/processed/dataset_{a.profile}.parquet")
    df = pd.read_parquet(parquet)
    meta_p = _MODEL_DIR / "model_meta.json"
    stamp = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    stamp = {k: stamp.get(k) for k in ("version", "profile", "trained_at")} | \
            {"dataset_hash": file_hash(parquet), "seeds": a.seeds}
    train_gnn(df, json.loads(Path("data/processed/splits.json").read_text()),
              pd.read_parquet("data/raw/baseline_states.parquet"), a.seeds, a.profile,
              a.epochs, a.threads, stamp=stamp)


if __name__ == "__main__":
    _cli()
