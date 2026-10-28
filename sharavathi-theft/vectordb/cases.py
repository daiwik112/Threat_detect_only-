"""
vectordb/cases.py — "Similar past cases" over the labelled dataset.

Each scenario is embedded as its physics-feature vector (ml.features, the same 41 features the
models use), standardized with mean/std from the TRAIN split so no single feature dominates.
Only train + val scenarios are indexed, so held-out test cases are never returned as their own
"neighbour" during evaluation.

    idx = CaseIndex.load_or_build()
    idx.similar(measurements, k=5)  -> [{scenario_id, label, true_km, magnitude_mw, similarity}]

Build: python -m vectordb.cases   (writes data/models/case_index.joblib)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vectordb.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "dataset_major_only.parquet"
SPLITS = ROOT / "data" / "processed" / "splits.json"
BASELINE = ROOT / "data" / "raw" / "baseline_states.parquet"
INDEX_PATH = ROOT / "data" / "models" / "case_index.joblib"


class CaseIndex:
    def __init__(self, store: VectorStore, feat_cols: list[str], mu: np.ndarray, sd: np.ndarray,
                 baseline_df: pd.DataFrame, profile: str = "major_only"):
        self.store, self.feat_cols, self.mu, self.sd = store, feat_cols, mu, sd
        self.baseline_df, self.profile = baseline_df, profile

    def embed_frame(self, df_feat: pd.DataFrame) -> np.ndarray:
        X = df_feat.reindex(columns=self.feat_cols).to_numpy(dtype=np.float32)
        X = (X - self.mu) / self.sd
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def embed_measurements(self, measurements: dict) -> np.ndarray:
        from ml.features import build_feature_matrix
        raw = {k: (np.nan if v is None else float(v)) for k, v in measurements.items() if k.startswith("m_")}
        return self.embed_frame(build_feature_matrix(pd.DataFrame([raw]), self.baseline_df, self.profile))[0]

    def similar(self, measurements: dict, k: int = 5, label: str | None = None) -> list[dict]:
        hits = self.store.search(self.embed_measurements(measurements), k=k,
                                 where={"label": label} if label else None)
        return [{"scenario_id": h.id, "similarity": round(h.score, 4), **h.metadata} for h in hits]

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, save: bool = True) -> "CaseIndex":
        from ml.features import build_feature_matrix, get_feature_cols
        df = pd.read_parquet(DATASET)
        bl = pd.read_parquet(BASELINE)
        splits = json.loads(SPLITS.read_text())
        df = build_feature_matrix(df, bl).reset_index(drop=True)
        feat_cols = get_feature_cols(df)
        train = df.iloc[splits["train"]]
        Xtr = train[feat_cols].to_numpy(dtype=np.float32)
        mu = np.nanmean(Xtr, axis=0).astype(np.float32)
        sd = (np.nanstd(Xtr, axis=0) + 1e-6).astype(np.float32)
        pool = df.iloc[splits["train"] + splits["val"]]
        idx = cls(VectorStore(len(feat_cols), "cases"), feat_cols, mu, sd, bl)
        meta = [{"label": r.label,
                 "true_km": None if pd.isna(r.true_km) else float(r.true_km),
                 "magnitude_mw": round(float(r.magnitude_mw), 2),
                 "load_scale": round(float(r.load_scale), 3)} for r in pool.itertuples()]
        idx.store.add(pool["scenario_id"].tolist(), idx.embed_frame(pool), meta)
        if save:
            INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"store": {"dim": idx.store.dim, "ids": idx.store.ids, "metadata": idx.store.metadata,
                                   "vectors": idx.store.vectors},
                         "feat_cols": feat_cols, "mu": mu, "sd": sd,
                         "dataset_mtime": DATASET.stat().st_mtime}, INDEX_PATH)
        return idx

    @classmethod
    def load_or_build(cls) -> "CaseIndex":
        if INDEX_PATH.exists():
            d = joblib.load(INDEX_PATH)
            if d.get("dataset_mtime") == DATASET.stat().st_mtime:
                vs = VectorStore(d["store"]["dim"], "cases")
                vs.ids, vs.metadata, vs.vectors = d["store"]["ids"], d["store"]["metadata"], d["store"]["vectors"]
                vs._pos = {id_: i for i, id_ in enumerate(vs.ids)}
                return cls(vs, d["feat_cols"], d["mu"], d["sd"], pd.read_parquet(BASELINE))
        return cls.build()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    i = CaseIndex.build()
    print(f"case index: {len(i.store)} scenarios × {i.store.dim} features → {INDEX_PATH}")
