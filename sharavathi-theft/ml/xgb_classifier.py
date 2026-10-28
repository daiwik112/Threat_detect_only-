"""
ml/xgb_classifier.py — XGBoost anomaly classifier (Step 3).

Heads
-----
1. Binary  : normal vs anomaly (scale_pos_weight balanced)
2. 6-class : normal/theft/sensor_bias/sensor_drift/missing/line_outage (balanced sample weights)
Plus SHAP top contributors (binary head) and likely_cause mapping.

Missing features stay NaN — XGBoost learns a default split direction for them.

Public API
----------
XGBClassifier
    fit(train_df, val_df, feat_cols, tune=False)
    predict(df) / predict_proba_binary(df) / predict_proba_multi(df)
    likely_cause(df) -> list[{extraction, sensor_error, other}]
    shap_top(df, n=5) -> list[list[{feature, shap}]]
    metrics(df) -> {detection, classification}
    save(path) / load(path)
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

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

LABELS = ["normal", "theft", "sensor_bias", "sensor_drift", "missing", "line_outage"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

# likely_cause is computed over the ANOMALY classes only (renormalised), so it answers
# "if this is an anomaly, what caused it?" independently of the anomaly score.
CAUSE_GROUPS = {
    "extraction":   ["theft"],
    "sensor_error": ["sensor_bias", "sensor_drift", "missing"],
    "other":        ["line_outage"],
}


class XGBClassifier:
    def __init__(self, seed: int = 42, n_estimators: int = 600,
                 max_depth: int = 6, learning_rate: float = 0.05,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 early_stopping_rounds: int = 40):
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost not installed — run: pip install xgboost")
        self.seed = seed
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, subsample=subsample,
                           colsample_bytree=colsample_bytree, random_state=seed,
                           tree_method="hist", early_stopping_rounds=early_stopping_rounds)
        self._binary_model: Optional["xgb.XGBClassifier"] = None
        self._multi_model: Optional["xgb.XGBClassifier"] = None
        self._feat_cols: list[str] = []
        self.tuned_params: dict = {}

    @property
    def feat_cols(self) -> list[str]:
        return self._feat_cols

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df.reindex(columns=self._feat_cols).to_numpy(dtype=float)

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            feat_cols: list[str], tune: bool = False) -> "XGBClassifier":
        self._feat_cols = list(feat_cols)
        X_tr, X_vl = self._X(train_df), self._X(val_df)
        y_bin_tr = (train_df["label"] != "normal").astype(int).to_numpy()
        y_bin_vl = (val_df["label"] != "normal").astype(int).to_numpy()
        y_mc_tr = train_df["label"].map(LABEL_TO_IDX).to_numpy(dtype=int)
        y_mc_vl = val_df["label"].map(LABEL_TO_IDX).to_numpy(dtype=int)

        if tune:
            self.tuned_params = self._tune_binary(X_tr, y_bin_tr, X_vl, y_bin_vl)
        p = {**self.params, **self.tuned_params}

        n_pos = int(y_bin_tr.sum())
        self._binary_model = xgb.XGBClassifier(
            **p, objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=(len(y_bin_tr) - n_pos) / max(n_pos, 1))
        self._binary_model.fit(X_tr, y_bin_tr, eval_set=[(X_vl, y_bin_vl)], verbose=False)

        counts = np.bincount(y_mc_tr, minlength=len(LABELS)).astype(float)
        w_cls = len(y_mc_tr) / (len(LABELS) * np.maximum(counts, 1))
        self._multi_model = xgb.XGBClassifier(
            **p, objective="multi:softprob", num_class=len(LABELS), eval_metric="mlogloss")
        self._multi_model.fit(X_tr, y_mc_tr, sample_weight=w_cls[y_mc_tr],
                              eval_set=[(X_vl, y_mc_vl)],
                              sample_weight_eval_set=[w_cls[y_mc_vl]], verbose=False)
        return self

    def _tune_binary(self, X_tr, y_tr, X_vl, y_vl, n_trials: int = 50) -> dict:
        """Light Optuna tuning (≤50 trials) on val logloss. Returns XGBoost param names."""
        try:
            import optuna
        except ImportError:
            print("  [tune] optuna not installed — using default params")
            return {}
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            prm = dict(
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_float("min_child_weight", 1, 10, log=True),
            )
            m = xgb.XGBClassifier(**{**self.params, **prm}, objective="binary:logistic",
                                  eval_metric="logloss")
            m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            return m.best_score

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._binary_model.predict(self._X(df))

    def predict_proba_binary(self, df: pd.DataFrame) -> np.ndarray:
        return self._binary_model.predict_proba(self._X(df))[:, 1]

    def predict_proba_multi(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 6) probability matrix in LABELS order."""
        return self._multi_model.predict_proba(self._X(df))

    def predict_label(self, df: pd.DataFrame) -> list[str]:
        return [LABELS[i] for i in self.predict_proba_multi(df).argmax(axis=1)]

    def likely_cause(self, df: pd.DataFrame) -> list[dict]:
        proba = self.predict_proba_multi(df)
        out = []
        for row in proba:
            c = {k: float(sum(row[LABEL_TO_IDX[l]] for l in ls)) for k, ls in CAUSE_GROUPS.items()}
            tot = sum(c.values()) or 1.0
            out.append({k: v / tot for k, v in c.items()})
        return out

    def shap_top(self, df: pd.DataFrame, n: int = 5) -> list[list[dict]]:
        """SHAP top-n contributors per row (binary head, log-odds units)."""
        if not _SHAP_AVAILABLE:
            return [[] for _ in range(len(df))]
        if getattr(self, "_explainer", None) is None:      # building it costs ~100s of ms
            self._explainer = _shap.TreeExplainer(self._binary_model)
        sv = self._explainer.shap_values(self._X(df))
        if isinstance(sv, list):
            sv = sv[1]
        return [[{"feature": self._feat_cols[j], "shap": float(r[j])}
                 for j in np.argsort(-np.abs(r))[:n]] for r in sv]

    # ------------------------------------------------------------------
    def metrics(self, df: pd.DataFrame) -> dict:
        from sklearn.metrics import confusion_matrix
        from ml.evaluate import detection_metrics

        y_true_bin = (df["label"] != "normal").astype(int).to_numpy()
        y_true_mc = df["label"].map(LABEL_TO_IDX).to_numpy(dtype=int)
        y_pred_mc = self.predict_proba_multi(df).argmax(axis=1)
        cm = confusion_matrix(y_true_mc, y_pred_mc, labels=list(range(len(LABELS))))
        return {
            "detection": detection_metrics(y_true_bin, self.predict(df), self.predict_proba_binary(df)),
            "classification": {
                "labels": LABELS,
                "confusion": cm.tolist(),
                "per_class_recall": {
                    LABELS[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None)
                    for i in range(len(LABELS))
                },
                "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
            },
        }

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "XGBClassifier":
        return joblib.load(path)
