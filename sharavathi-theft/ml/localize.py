"""
ml/localize.py — Theft localization (Step 4). Trained on theft rows only.

A. LocalizerXGBReg      XGBRegressor → km; split-conformal interval calibrated on val.
B. LocalizerSectionRank XGBClassifier over candidate taps → probability vector →
                        ranked_taps / ranked_sections; est_km = prob-weighted mean;
                        interval = smallest contiguous tap window covering ≥80 % prob.

Leave-location-out note: taps in the holdout set never appear in training, so the
ranker cannot put mass on them directly — only est_km (weighted mean) can land there.
Top-k metrics therefore count a hit if a ranked tap lies within 10 km of the truth.

Public API
----------
localization_metrics(true_km, est_km, lo, hi, magnitude_mw, ...) -> dict
LocalizerXGBReg / LocalizerSectionRank : fit(train, val, feat_cols), predict(df), metrics(df), save/load
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

CORRIDOR_KM = 361.0
MAG_BUCKETS = {"1_5": (1.0, 5.0), "5_15": (5.0, 15.0), "15_40": (15.0, 40.01)}


def localization_metrics(true_km, est_km, lo=None, hi=None, magnitude_mw=None,
                         section_true=None, section_pred=None, top_k_hit=None,
                         method: str = "") -> dict:
    true_km, est_km = np.asarray(true_km, float), np.asarray(est_km, float)
    if len(true_km) == 0:
        return {}
    err = np.abs(est_km - true_km)

    def block(mask):
        if mask.sum() == 0:
            return {"n": 0, "mae_km": None, "median_km": None, "within_10km": None,
                    "within_25km": None}
        e = err[mask]
        return {"n": int(mask.sum()), "mae_km": float(e.mean()), "median_km": float(np.median(e)),
                "within_10km": float((e <= 10).mean()), "within_25km": float((e <= 25).mean())}

    out = {
        "method": method,
        "n": int(len(err)),
        "mae_km": float(err.mean()),
        "median_km": float(np.median(err)),
        "rmse_km": float(np.sqrt((err ** 2).mean())),
        "p90_km": float(np.percentile(err, 90)),
        "within_10km": float((err <= 10).mean()),
        "within_25km": float((err <= 25).mean()),
        "within_50km": float((err <= 50).mean()),
    }
    if lo is not None and hi is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        out["interval_coverage"] = float(((true_km >= lo - 1e-6) & (true_km <= hi + 1e-6)).mean())
        out["interval_width_km"] = float(np.mean(hi - lo))
    if section_true is not None and section_pred is not None:
        out["section_accuracy"] = float(np.mean(np.asarray(section_true) == np.asarray(section_pred)))
    if top_k_hit is not None:
        out["top3_within_10km"] = float(np.mean(top_k_hit))
    if magnitude_mw is not None:
        mag = np.asarray(magnitude_mw, float)
        out["by_magnitude"] = {k: block((mag >= a) & (mag < b)) for k, (a, b) in MAG_BUCKETS.items()}
    return out


def _theft_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["true_km"].notna()]


# ---------------------------------------------------------------------------
# A. Regression
# ---------------------------------------------------------------------------

class LocalizerXGBReg:
    method = "xgb_reg"

    def __init__(self, seed: int = 42, n_estimators: int = 800, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, coverage: float = 0.80, profile: str = "major_only"):
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        self.coverage = coverage
        self.profile = profile
        self.conformal_radius: float = CORRIDOR_KM
        self._feat_cols: list[str] = []
        self._model = None
        self._params = dict(n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate, subsample=subsample,
                            colsample_bytree=colsample_bytree, random_state=seed,
                            tree_method="hist", objective="reg:absoluteerror",
                            eval_metric="mae", early_stopping_rounds=40)

    def _X(self, df):
        return df.reindex(columns=self._feat_cols).to_numpy(dtype=float)

    def fit(self, train_df, val_df, feat_cols) -> "LocalizerXGBReg":
        self._feat_cols = list(feat_cols)
        tr, vl = _theft_rows(train_df), _theft_rows(val_df)
        self._model = xgb.XGBRegressor(**self._params)
        self._model.fit(self._X(tr), tr["true_km"].to_numpy(),
                        eval_set=[(self._X(vl), vl["true_km"].to_numpy())], verbose=False)
        # split-conformal: finite-sample-corrected quantile of |val residual|
        res = np.abs(self._model.predict(self._X(vl)) - vl["true_km"].to_numpy())
        q = min(1.0, np.ceil((len(res) + 1) * self.coverage) / len(res))
        self.conformal_radius = float(np.quantile(res, q))
        return self

    def predict(self, df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        est = np.clip(self._model.predict(self._X(df)), 0.0, CORRIDOR_KM)
        return est, np.clip(est - self.conformal_radius, 0, CORRIDOR_KM), \
            np.clip(est + self.conformal_radius, 0, CORRIDOR_KM)

    def metrics(self, df) -> dict:
        from ml.features import get_grid_meta
        sub = _theft_rows(df)
        if len(sub) == 0:
            return {}
        meta = get_grid_meta(self.profile)
        est, lo, hi = self.predict(sub)
        m = localization_metrics(sub["true_km"], est, lo, hi, sub["magnitude_mw"],
                                 [meta.section_of_km(k) for k in sub["true_km"]],
                                 [meta.section_of_km(k) for k in est], method=self.method)
        m["conformal_radius_km"] = self.conformal_radius
        return m

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "LocalizerXGBReg":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# B. Tap / section ranking
# ---------------------------------------------------------------------------

class LocalizerSectionRank:
    method = "section_rank"

    def __init__(self, seed: int = 42, n_estimators: int = 600, max_depth: int = 5,
                 learning_rate: float = 0.05, coverage: float = 0.80, profile: str = "major_only"):
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        self.coverage = coverage
        self.profile = profile
        self.tap_ids: list[str] = []     # classes seen in training, in label order
        self.tap_kms: np.ndarray = np.array([])
        self._feat_cols: list[str] = []
        self._model = None
        self._params = dict(n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                            random_state=seed, tree_method="hist",
                            objective="multi:softprob", eval_metric="mlogloss",
                            early_stopping_rounds=40)

    def _X(self, df):
        return df.reindex(columns=self._feat_cols).to_numpy(dtype=float)

    def fit(self, train_df, val_df, feat_cols) -> "LocalizerSectionRank":
        self._feat_cols = list(feat_cols)
        tr, vl = _theft_rows(train_df), _theft_rows(val_df)
        km_by_tap = tr.groupby("true_tap_id")["true_km"].first()
        km_by_tap = km_by_tap.sort_values()
        self.tap_ids = list(km_by_tap.index)
        self.tap_kms = km_by_tap.to_numpy(dtype=float)
        enc = {t: i for i, t in enumerate(self.tap_ids)}
        vl = vl[vl["true_tap_id"].isin(enc)]
        self._model = xgb.XGBClassifier(**self._params, num_class=len(self.tap_ids))
        self._model.fit(self._X(tr), tr["true_tap_id"].map(enc).to_numpy(),
                        eval_set=[(self._X(vl), vl["true_tap_id"].map(enc).to_numpy())],
                        verbose=False)
        return self

    def predict_proba(self, df) -> np.ndarray:
        return self._model.predict_proba(self._X(df))

    def predict(self, df) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict]]]:
        """Returns (est_km, lo, hi, ranked_taps per row)."""
        P = self.predict_proba(df)
        est = P @ self.tap_kms
        lo, hi, ranked = [], [], []
        for p in P:
            a, b = _smallest_window(p, self.coverage)
            lo.append(self.tap_kms[a])
            hi.append(self.tap_kms[b])
            order = np.argsort(-p)[:10]
            ranked.append([{"tap_id": self.tap_ids[i], "km": float(self.tap_kms[i]),
                            "prob": float(p[i])} for i in order])
        return est, np.array(lo), np.array(hi), ranked

    def ranked_sections(self, df) -> list[list[dict]]:
        from ml.features import get_grid_meta
        meta = get_grid_meta(self.profile)
        sec_of = np.array([meta.section_of_km(k) for k in self.tap_kms])
        out = []
        for p in self.predict_proba(df):
            agg = {s.name: float(p[sec_of == s.name].sum()) for s in meta.sections}
            out.append(sorted(({"section": k, "prob": v} for k, v in agg.items()),
                              key=lambda d: -d["prob"]))
        return out

    def metrics(self, df) -> dict:
        from ml.features import get_grid_meta
        sub = _theft_rows(df)
        if len(sub) == 0:
            return {}
        meta = get_grid_meta(self.profile)
        est, lo, hi, ranked = self.predict(sub)
        true = sub["true_km"].to_numpy()
        hits = [any(abs(t["km"] - k) <= 10 for t in r[:3]) for r, k in zip(ranked, true)]
        sec_pred = [r[0]["section"] for r in self.ranked_sections(sub)]
        return localization_metrics(true, est, lo, hi, sub["magnitude_mw"],
                                    [meta.section_of_km(k) for k in true], sec_pred,
                                    top_k_hit=hits, method=self.method)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "LocalizerSectionRank":
        return joblib.load(path)


def _smallest_window(p: np.ndarray, coverage: float) -> tuple[int, int]:
    """Indices (a, b) of the narrowest contiguous window with Σp ≥ coverage (two-pointer)."""
    coverage -= 1e-9                       # float tolerance (0.1 + 0.7 < 0.8)
    best = (0, len(p) - 1)
    a, s = 0, 0.0
    for b in range(len(p)):
        s += p[b]
        while a < b and s - p[a] >= coverage:
            s -= p[a]
            a += 1
        if s >= coverage and (b - a) < (best[1] - best[0]):
            best = (a, b)
    return best
