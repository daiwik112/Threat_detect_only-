"""
ml/detector_baseline.py — Physics residual baseline detector (Step 2).

Fitted on NORMAL training rows only. This is the benchmark every ML model must beat.

  1. Residual threshold: z_s = |f_r_p_<s>| / σ_s (σ from normal train).
     suspicious if max_s z_s > 99.5th percentile of max-z on normal train.
  2. Chi-square-style test: J = Σ_s (r_p/σ)² + (r_q/σ)², threshold = 99.5th pct on normal train.
     (A full WLS J is not used: the 39-bus state is unobservable under major_only,
      so the residual-based J is the comparable statistic.)

Public API
----------
BaselineDetector
    fit(train_df)
    statistic(df) -> np.ndarray     raw stat (max over both tests, in threshold units; >1 = alarm)
    score(df)     -> np.ndarray     0–1 anomaly score, 0.5 at the threshold
    predict(df)   -> np.ndarray     0=normal, 1=suspicious
    section_z(df) -> DataFrame      per-section z-scores (for DetectResponse.residuals)
    metrics(df, y_true) -> dict
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BaselineDetector:
    def __init__(self, percentile: float = 99.5):
        self.percentile = percentile
        self.sections: list[str] = []
        self.sigma_p: dict[str, float] = {}
        self.sigma_q: dict[str, float] = {}
        self.z_threshold: float = np.inf
        self.j_threshold: float = np.inf

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "BaselineDetector":
        normal = train_df[train_df["label"] == "normal"]
        if len(normal) == 0:
            raise ValueError("No normal rows in train_df — cannot fit baseline detector")
        self.sections = [c[len("f_r_p_"):] for c in train_df.columns
                         if c.startswith("f_r_p_") and c != "f_r_p_total"]
        for s in self.sections:
            self.sigma_p[s] = float(np.nanstd(normal[f"f_r_p_{s}"])) or 1.0
            self.sigma_q[s] = float(np.nanstd(normal[f"f_r_q_{s}"])) or 1.0
        z, J = self._raw(normal)
        self.z_threshold = float(np.percentile(z, self.percentile))
        self.j_threshold = float(np.percentile(J, self.percentile))
        return self

    def section_z(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            s: df[f"f_r_p_{s}"].fillna(0.0).abs().to_numpy() / self.sigma_p[s]
            for s in self.sections
        }, index=df.index)

    def _raw(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        zp = self.section_z(df).to_numpy()
        zq = np.column_stack([df[f"f_r_q_{s}"].fillna(0.0).to_numpy() / self.sigma_q[s]
                              for s in self.sections])
        return zp.max(axis=1), (zp ** 2).sum(axis=1) + (zq ** 2).sum(axis=1)

    # ------------------------------------------------------------------
    def statistic(self, df: pd.DataFrame) -> np.ndarray:
        z, J = self._raw(df)
        return np.maximum(z / self.z_threshold, J / self.j_threshold)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Monotone map of statistic to [0,1]; equals 0.5 exactly at the alarm threshold."""
        return 1.0 - np.power(0.5, self.statistic(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.statistic(df) > 1.0).astype(int)

    # ------------------------------------------------------------------
    def metrics(self, df: pd.DataFrame, y_true: np.ndarray) -> dict:
        from ml.evaluate import detection_metrics
        return detection_metrics(y_true, self.predict(df), self.score(df))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "BaselineDetector":
        return joblib.load(path)
