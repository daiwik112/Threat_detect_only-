"""
ml/infer.py — Inference API (Step 8). Used by the WF4 backend.

    bundle = load_models()
    bundle.detect({"m_SHARAVATHI_p": ..., ...})    -> DetectResponse dict
    bundle.localize({"m_...": ...}, method=None)   -> LocalizeResponse dict

Input is RAW measurements (m_* keys, NaN/None for missing). Features are recomputed
here with the same ml.features code used in training, so the API cannot drift from it.
Output schemas match 08_DATA_CONTRACTS.md §4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_DEFAULT_MODEL_DIR = Path("data/models")
_DEFAULT_BASELINE = Path("data/raw/baseline_states.parquet")


@dataclass
class ModelBundle:
    baseline_detector: Any = None
    xgb_classifier: Any = None
    localizer_reg: Any = None
    localizer_rank: Any = None
    gnn: Any = None
    baseline_df: Optional[pd.DataFrame] = None
    feat_cols: list = field(default_factory=list)
    profile: str = "major_only"
    selected_localizer: str = "xgb_reg"
    dataset_hash: str = ""
    version: str = "wf3-v2"

    # ------------------------------------------------------------------
    def features(self, measurements: dict[str, Any]) -> pd.DataFrame:
        """One-row DataFrame with m_* inputs + recomputed f_* features."""
        from ml.features import build_feature_matrix
        raw = {k: (np.nan if v is None else float(v))
               for k, v in measurements.items() if k.startswith("m_")}
        df = pd.DataFrame([raw])
        if self.baseline_df is not None:
            df = build_feature_matrix(df, self.baseline_df, self.profile)
        for c in self.feat_cols:           # guarantee every expected column exists
            if c not in df.columns:
                df[c] = np.nan
        return df

    # ------------------------------------------------------------------
    def detect(self, measurements: dict[str, Any], explain: bool = True) -> dict:
        """explain=False skips SHAP (used for batch sweeps; top_contributors is then empty)."""
        df = self.features(measurements)
        if self.xgb_classifier is not None:
            score = float(self.xgb_classifier.predict_proba_binary(df)[0])
            likely = self.xgb_classifier.likely_cause(df)[0]
            top = []
            if explain:
                try:
                    top = self.xgb_classifier.shap_top(df, n=5)[0]
                except Exception:
                    top = []
        elif self.baseline_detector is not None:
            score = float(self.baseline_detector.score(df)[0])
            likely = {"extraction": 1.0, "sensor_error": 0.0, "other": 0.0}
            top = []
        else:
            raise RuntimeError("No detector loaded — run `make train` first")

        return {
            "status": "suspicious" if score >= 0.5 else "normal",
            "anomaly_score": round(score, 4),
            "residuals": self._residuals(df),
            "top_contributors": [{"feature": t["feature"], "shap": round(t["shap"], 4)} for t in top],
            "likely_cause": {k: round(v, 4) for k, v in likely.items()},
        }

    def _residuals(self, df: pd.DataFrame) -> list[dict]:
        from ml.features import get_grid_meta
        bd = self.baseline_detector
        out = []
        for s in get_grid_meta(self.profile).sections:
            r_p = df.get(f"f_r_p_{s.name}", pd.Series([np.nan])).iloc[0]
            r_q = df.get(f"f_r_q_{s.name}", pd.Series([np.nan])).iloc[0]
            sigma = bd.sigma_p.get(s.name) if bd is not None else None
            z = abs(r_p) / sigma if (sigma and np.isfinite(r_p)) else None
            out.append({
                "section": s.name,
                "r_p_mw": None if not np.isfinite(r_p) else round(float(r_p), 3),
                "r_q_mvar": None if not np.isfinite(r_q) else round(float(r_q), 3),
                "z": None if z is None else round(float(z), 3),
            })
        return out

    # ------------------------------------------------------------------
    def localize(self, measurements: dict[str, Any], method: Optional[str] = None) -> dict:
        """method: "xgb_reg" | "section_rank" | "gnn" | None (= best on leave-location-out holdout)."""
        method = method or self.selected_localizer
        df = self.features(measurements)
        ranked_taps: list[dict] = []
        ranked_sections: list[dict] = []

        if self.localizer_rank is not None:
            ranked_taps = self.localizer_rank.predict(df)[3][0]
            ranked_sections = self.localizer_rank.ranked_sections(df)[0]

        if method == "gnn" and self.gnn is not None:
            est, lo, hi, ranked_taps = self.gnn.predict(df)
            ranked_taps = ranked_taps[0]
            ranked_sections = self.gnn.ranked_sections(df)[0]
        elif method == "section_rank" and self.localizer_rank is not None:
            est, lo, hi, _ = self.localizer_rank.predict(df)
        elif self.localizer_reg is not None:
            method = "xgb_reg"
            est, lo, hi = self.localizer_reg.predict(df)
        else:
            raise RuntimeError("No localizer loaded — run `make train` first")

        return {
            "method": method,
            "est_km": round(float(est[0]), 1),
            "interval_km": [round(float(lo[0]), 1), round(float(hi[0]), 1)],
            "ranked_sections": [{"section": r["section"], "prob": round(r["prob"], 4)}
                                for r in ranked_sections],
            "ranked_taps": [{"tap_id": t["tap_id"], "km": t["km"], "prob": round(t["prob"], 4)}
                            for t in ranked_taps[:10]],
        }


def load_models(model_dir: Path = _DEFAULT_MODEL_DIR,
                baseline_path: Path = _DEFAULT_BASELINE, verbose: bool = False) -> ModelBundle:
    """Load all artifacts from model_dir; missing ones are left as None."""
    import joblib
    model_dir = Path(model_dir)
    b = ModelBundle()

    def _load(name):
        p = model_dir / f"{name}.joblib"
        if verbose:
            print(f"  {'Loaded' if p.exists() else '[MISSING]'}: {p.name}")
        return joblib.load(p) if p.exists() else None

    b.baseline_detector = _load("baseline_detector")
    b.xgb_classifier = _load("xgb_classifier")
    b.localizer_reg = _load("localizer_xgb_reg")
    b.localizer_rank = _load("localizer_section_rank")
    if Path(baseline_path).exists():
        b.baseline_df = pd.read_parquet(baseline_path)
    if (model_dir / "gnn.pt").exists() and b.baseline_df is not None:
        from ml.gnn.model import GNNLocalizer, is_gnn_available
        if is_gnn_available():
            b.gnn = GNNLocalizer.load(model_dir / "gnn.pt", b.baseline_df)
    if (model_dir / "feat_cols.json").exists():
        b.feat_cols = json.loads((model_dir / "feat_cols.json").read_text())
    if (model_dir / "model_meta.json").exists():
        meta = json.loads((model_dir / "model_meta.json").read_text())
        b.dataset_hash = meta.get("dataset_hash", "")
        b.version = meta.get("version", b.version)
        b.profile = meta.get("profile", b.profile)
        b.selected_localizer = meta.get("selected_localizer") or b.selected_localizer
    return b
