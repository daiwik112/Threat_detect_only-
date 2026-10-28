"""
ml/evaluate.py — Held-out evaluation → data/models/metrics.json (Step 7).

All numbers come from `test` (group-held-out) and `test_loc_holdout` (leave-location-out).

  detection       precision/recall/F1/FPR at the model's operating point, ROC-AUC,
                  recall at fixed FPR (1 %, 5 %) — XGBoost vs residual baseline
  classification  6-class confusion matrix + per-class recall
  localization    MAE / median / within 10-25-50 km / by magnitude bucket / interval coverage
                  for xgb_reg, section_rank and a physics baseline (max-residual section +
                  angle-drop position estimate). Localization is scored on true theft rows.
  robustness      1–3 extra missing metered channels, noise ×2 (features recomputed)
  gnn             if data/models/gnn.pt exists: same detection/localization metrics, seed
                  mean±std from gnn_results.json, and gnn_decision (kept only if it beats
                  XGBoost on holdout MAE or top-3 accuracy)

Top-level keys detection / classification / localization / baseline follow
MetricsResponse in 08_DATA_CONTRACTS.md; everything else is extra detail.

Public API
----------
detection_metrics(y_true, y_pred, score) -> dict
physics_localize(df, profile) -> (est_km, section_names)
evaluate_all(df, splits, feat_cols, baseline_df, model_dir, out_dir, profile) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_MODEL_DIR = Path("data/models")
FPR_TARGETS = (0.01, 0.05)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def recall_at_fpr(y_true: np.ndarray, score: np.ndarray, fpr: float) -> float | None:
    neg, pos = score[y_true == 0], score[y_true == 1]
    if len(neg) == 0 or len(pos) == 0:
        return None
    thr = np.quantile(neg, 1.0 - fpr, method="higher")
    return float((pos > thr).mean())


def detection_metrics(y_true, y_pred, score=None) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    n_neg = int((y_true == 0).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / n_neg) if n_neg else None,
        "n_pos": int(y_true.sum()), "n_neg": n_neg,
    }
    if score is not None and n_neg and y_true.sum():
        score = np.asarray(score, float)
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        for f in FPR_TARGETS:
            out[f"recall_at_fpr_{int(f * 100)}pct"] = recall_at_fpr(y_true, score, f)
    return out


def _binary(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] != "normal").astype(int).to_numpy()


def physics_localize(df: pd.DataFrame, profile: str = "major_only") -> tuple[np.ndarray, list[str]]:
    """Non-ML benchmark: section with largest |r_p|, position from angle-drop fraction."""
    from ml.features import get_grid_meta
    meta = get_grid_meta(profile)
    R = np.column_stack([df[f"f_r_p_{s.name}"].fillna(0).abs() for s in meta.sections])
    X = np.column_stack([df[f"f_xfrac_{s.name}"].to_numpy(dtype=float) for s in meta.sections])
    pick = R.argmax(axis=1)
    frac = X[np.arange(len(df)), pick]
    frac = np.where(np.isfinite(frac), np.clip(frac, 0, 1), 0.5)
    km_from = np.array([s.km_from for s in meta.sections])[pick]
    length = np.array([s.length_km for s in meta.sections])[pick]
    return km_from + frac * length, [meta.sections[k].name for k in pick]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _load_models(model_dir: Path, baseline_df: pd.DataFrame | None = None) -> dict:
    import joblib
    models = {n: joblib.load(model_dir / f"{n}.joblib")
              for n in ["baseline_detector", "xgb_classifier", "localizer_xgb_reg",
                        "localizer_section_rank"]
              if (model_dir / f"{n}.joblib").exists()}
    if (model_dir / "gnn.pt").exists() and baseline_df is not None:
        from ml.gnn.model import GNNLocalizer, is_gnn_available
        if is_gnn_available():
            models["gnn"] = GNNLocalizer.load(model_dir / "gnn.pt", baseline_df)
    return models


def evaluate_all(df: pd.DataFrame, splits: dict, feat_cols: list[str],
                 baseline_df: pd.DataFrame, model_dir: Path = _MODEL_DIR,
                 out_dir: Path = _MODEL_DIR, profile: str = "major_only",
                 make_plots: bool = True) -> dict:
    from ml.features import get_grid_meta
    from ml.localize import localization_metrics

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = _load_models(model_dir, baseline_df)
    meta = get_grid_meta(profile)
    gnn = models.get("gnn")

    sets = {"test": df.loc[splits["test"]], "holdout": df.loc[splits["test_loc_holdout"]]}
    # Detection on holdout needs negatives: holdout thefts + all test normals
    sets["holdout_vs_normal"] = pd.concat([sets["holdout"],
                                           sets["test"][sets["test"]["label"] == "normal"]])
    M: dict = {"splits": {k: int(len(v)) for k, v in sets.items()}}

    bd, clf = models.get("baseline_detector"), models.get("xgb_classifier")
    det = {}
    for name, d in sets.items():
        y = _binary(d)
        det[name] = {}
        if bd is not None:
            det[name]["baseline"] = detection_metrics(y, bd.predict(d), bd.score(d))
        if clf is not None:
            det[name]["xgb"] = detection_metrics(y, clf.predict(d), clf.predict_proba_binary(d))
        if gnn is not None:
            g = gnn.predict_proba_binary(d)
            det[name]["gnn"] = detection_metrics(y, (g >= 0.5).astype(int), g)
        # theft-only recall (the class that matters most)
        t = d[d["label"].isin(["theft", "normal"])]
        yt = _binary(t)
        if bd is not None:
            det[name]["baseline_theft_vs_normal"] = detection_metrics(yt, bd.predict(t), bd.score(t))
        if clf is not None:
            det[name]["xgb_theft_vs_normal"] = detection_metrics(yt, clf.predict(t),
                                                                 clf.predict_proba_binary(t))
    M["detection_detail"] = det

    if clf is not None:
        M["classification_detail"] = {k: clf.metrics(sets[k])["classification"]
                                      for k in ("test", "holdout")}

    # --- Localization (true theft rows) ---
    loc = {}
    for name in ("test", "holdout"):
        th = sets[name][sets[name]["label"] == "theft"]
        loc[name] = {}
        est, secs = physics_localize(th, profile)
        loc[name]["physics_baseline"] = localization_metrics(
            th["true_km"], est, magnitude_mw=th["magnitude_mw"],
            section_true=[meta.section_of_km(k) for k in th["true_km"]], section_pred=secs,
            method="physics_baseline")
        for key in ("localizer_xgb_reg", "localizer_section_rank", "gnn"):
            if key in models:
                loc[name][models[key].method] = models[key].metrics(th)
    M["localization_detail"] = loc

    cands = {m: v.get("mae_km") for m, v in loc["holdout"].items()
             if m != "physics_baseline" and v.get("mae_km") is not None}
    if gnn is not None:
        M["gnn_decision"] = _gnn_decision(loc, model_dir)
        if not M["gnn_decision"]["keep"]:
            cands.pop("gnn", None)
    selected = min(cands, key=cands.get) if cands else None
    M["selected_localizer"] = selected

    M["robustness"] = _robustness(sets["test"], models, baseline_df, profile)

    # --- MetricsResponse-shaped summary (test split, selected localizer on holdout) ---
    pick = lambda d, *ks: {k: d.get(k) for k in ks}
    dkeys = ("precision", "recall", "f1", "fpr", "roc_auc", "recall_at_fpr_1pct", "recall_at_fpr_5pct")
    lkeys = ("mae_km", "median_km", "within_10km", "within_25km", "within_50km")
    if clf is not None:
        M["detection"] = pick(det["test"]["xgb"], *dkeys)
        M["classification"] = pick(M["classification_detail"]["test"], "confusion",
                                   "per_class_recall", "labels")
    if selected:
        M["localization"] = {"method": selected, "split": "test_loc_holdout",
                             **pick(loc["holdout"][selected], *lkeys)}
    if bd is not None:
        M["baseline"] = {**pick(det["test"]["baseline"], *dkeys),
                         **pick(loc["holdout"]["physics_baseline"], *lkeys)}
    M["beats_baseline"] = _beats_baseline(M)
    M["honest_claim"] = _honest_claim(M)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(M, f, indent=2, default=_json_default)
    print(f"Saved: {out_dir / 'metrics.json'}")

    if make_plots:
        try:
            _plots_and_arrays(sets, models, profile, out_dir)
        except Exception as e:  # plots are non-essential
            print(f"[WARN] Plot generation failed: {e}")
    return M


def _gnn_decision(loc: dict, model_dir: Path) -> dict:
    """03 §5: keep the GNN only if it improves holdout MAE or top-3 accuracy vs XGBoost.
    Uses the 3-seed mean when gnn_results.json is available."""
    h = loc["holdout"]
    g_mae, g_top3 = h["gnn"]["mae_km"], h["gnn"].get("top3_within_10km")
    seeds = None
    p = Path(model_dir) / "gnn_results.json"
    if p.exists():
        seeds = json.loads(p.read_text())["summary"]
        g_mae = seeds["holdout"]["mae_km"]["mean"]
        g_top3 = seeds["holdout"]["top3_within_10km"]["mean"]
    x_mae = h.get("xgb_reg", {}).get("mae_km")
    x_top3 = h.get("section_rank", {}).get("top3_within_10km")
    better_mae = x_mae is not None and g_mae < x_mae
    better_top3 = x_top3 is not None and g_top3 is not None and g_top3 > x_top3
    return {
        "keep": bool(better_mae or better_top3),
        "holdout_mae_km": {"gnn_seed_mean": g_mae, "xgb_reg": x_mae, "gnn_better": bool(better_mae)},
        "holdout_top3_within_10km": {"gnn_seed_mean": g_top3, "section_rank": x_top3,
                                     "gnn_better": bool(better_top3)},
        "seed_summary": seeds,
    }


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return None


def _beats_baseline(M: dict) -> dict:
    out = {}
    d, b = M.get("detection", {}), M.get("baseline", {})
    for k in ("recall_at_fpr_1pct", "recall_at_fpr_5pct", "roc_auc", "f1"):
        if d.get(k) is not None and b.get(k) is not None:
            out[k] = bool(d[k] > b[k])
    l = M.get("localization", {})
    if l.get("mae_km") is not None and b.get("mae_km") is not None:
        out["localization_mae_km"] = bool(l["mae_km"] < b["mae_km"])
    return out


def _honest_claim(M: dict) -> str:
    sel = M.get("selected_localizer")
    if not sel:
        return "Localization model not yet evaluated — run the full training pipeline."
    hold = M["localization_detail"]["holdout"][sel]
    parts = []
    for k, (lo, hi) in (("15_40", (15, 40)), ("5_15", (5, 15)), ("1_5", (1, 5))):
        b = hold.get("by_magnitude", {}).get(k, {})
        if b.get("n"):
            parts.append(f"±{b['median_km']:.0f} km median (MAE {b['mae_km']:.0f} km) "
                         f"for {lo}–{hi} MW thefts")
    det = M.get("detection_detail", {}).get("test", {}).get("xgb_theft_vs_normal", {})
    rec = det.get("recall_at_fpr_1pct")
    rec_s = f" Theft detection recall at 1 % false-alarm rate: {rec * 100:.0f} %." if rec is not None else ""
    return ("Under the simulated measurement setup (major_only metering, assumed noise from "
            "sensors.yaml, leave-location-out holdout), location is resolvable to "
            + "; ".join(parts) + "." + rec_s +
            " These are simulation results on an assumed corridor model, not field-validated.")


def _robustness(test_df: pd.DataFrame, models: dict, baseline_df: pd.DataFrame,
                profile: str) -> dict:
    """Perturb RAW metered channels, recompute features, re-score."""
    from ml.features import build_feature_matrix, get_grid_meta, metered_channels
    clf = models.get("xgb_classifier")
    if clf is None or baseline_df is None or len(baseline_df) == 0:
        return {}
    bd = models.get("baseline_detector")
    reg = models.get("localizer_xgb_reg")
    base = test_df[test_df["label"].isin(["normal", "theft"])].copy()
    chans = [c for c in metered_channels(get_grid_meta(profile)) if c in base.columns]
    rng = np.random.default_rng(0)

    def score(d: pd.DataFrame) -> dict:
        # NB: with dropped channels the binary head correctly says "anomaly" (missing is an
        # anomaly class), so its FPR→1 there. xgb_p_theft ranks theft vs normal by the
        # 6-class P(theft) instead — the operationally relevant question.
        y = _binary(d)
        p_theft = clf.predict_proba_multi(d)[:, 1]
        r = {"xgb": detection_metrics(y, clf.predict(d), clf.predict_proba_binary(d)),
             "xgb_p_theft": detection_metrics(y, (p_theft >= 0.5).astype(int), p_theft)}
        if bd is not None:
            r["baseline"] = detection_metrics(y, bd.predict(d), bd.score(d))
        if reg is not None:
            r["xgb_reg_mae_km"] = reg.metrics(d[d["label"] == "theft"]).get("mae_km")
        keep = ("recall", "fpr", "recall_at_fpr_1pct", "roc_auc")
        return {k: ({kk: v.get(kk) for kk in keep} if isinstance(v, dict) else v) for k, v in r.items()}

    raw_cols = [c for c in base.columns if not c.startswith("f_")]
    out = {"clean": score(base)}
    for n_drop in (1, 2, 3):
        d = base[raw_cols].copy()
        arr = d[chans].to_numpy(dtype=float)
        for i in range(len(arr)):
            arr[i, rng.choice(len(chans), n_drop, replace=False)] = np.nan
        d[chans] = arr
        out[f"missing_{n_drop}_channels"] = score(build_feature_matrix(d, baseline_df, profile))

    # noise ×2: add independent noise with σ_extra = √(2²−1)·σ_base (sensors.yaml model)
    from grid.config import load_sensors
    nz = load_sensors().noise
    d = base[raw_cols].copy()
    for c in chans:
        v = d[c].to_numpy(dtype=float)
        if c.endswith("_vm"):
            sig = np.abs(v) * nz.vm_pct / 100
        elif c.endswith("_va"):
            sig = np.full_like(v, nz.va_deg)
        elif "_q" in c:
            sig = np.abs(v) * nz.q_pct / 100
        else:
            sig = np.abs(v) * nz.p_pct / 100
        d[c] = v + rng.normal(0, 1, len(v)) * np.sqrt(3) * np.nan_to_num(sig)
    out["noise_x2"] = score(build_feature_matrix(d, baseline_df, profile))
    return out


def _plots_and_arrays(sets: dict, models: dict, profile: str, out_dir: Path) -> None:
    """PNG plots + eval_arrays.json (raw arrays for frontend charts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays: dict = {}
    hold = sets["holdout"][sets["holdout"]["label"] == "theft"]
    test = sets["test"][sets["test"]["label"] == "theft"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key, color in (("localizer_xgb_reg", "tab:blue"), ("localizer_section_rank", "tab:orange"),
                       ("gnn", "tab:green")):
        if key not in models:
            continue
        m = models[key]
        for split_name, d in (("test", test), ("holdout", hold)):
            est = m.predict(d)[0]
            err = np.abs(est - d["true_km"].to_numpy())
            arrays[f"{m.method}_{split_name}"] = {
                "true_km": d["true_km"].round(1).tolist(), "est_km": np.round(est, 1).tolist(),
                "magnitude_mw": d["magnitude_mw"].round(2).tolist()}
            if split_name == "holdout":
                axes[0].scatter(d["magnitude_mw"], err, s=6, alpha=0.35, c=color, label=m.method)
                axes[1].scatter(d["true_km"], est, s=6, alpha=0.35, c=color, label=m.method)
    axes[0].set(xlabel="Theft magnitude (MW)", ylabel="|error| (km)",
                title="Holdout: localization error vs magnitude")
    axes[0].axhline(10, ls="--", c="g", lw=1); axes[0].axhline(25, ls="--", c="orange", lw=1)
    axes[1].plot([0, 361], [0, 361], "k--", lw=1)
    axes[1].set(xlabel="True km", ylabel="Estimated km", title="Holdout: estimated vs true location")
    for a in axes:
        a.legend()
    plt.tight_layout(); plt.savefig(out_dir / "eval_localization.png", dpi=120); plt.close()

    clf, bd = models.get("xgb_classifier"), models.get("baseline_detector")
    if clf is not None:
        from sklearn.metrics import roc_curve
        d = sets["test"]
        y = _binary(d)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sc = clf.predict_proba_binary(d)
        for lab in sorted(d["label"].unique()):
            axes[0].hist(sc[(d["label"] == lab).to_numpy()], bins=40, alpha=0.5, label=lab, density=True)
        axes[0].set(xlabel="XGB anomaly score", title="Test: score distribution per class"); axes[0].legend()
        arrays["roc"] = {}
        gnn = models.get("gnn")
        for name, s in (("xgb", sc), ("baseline", bd.score(d) if bd else None),
                        ("gnn", gnn.predict_proba_binary(d) if gnn else None)):
            if s is None:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            sel = np.unique(np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int))
            arrays["roc"][name] = {"fpr": fpr[sel].round(4).tolist(), "tpr": tpr[sel].round(4).tolist()}
            axes[1].plot(fpr, tpr, label=name)
        axes[1].set(xlabel="FPR", ylabel="TPR", xscale="log", xlim=(1e-3, 1), title="Test ROC")
        axes[1].legend()
        plt.tight_layout(); plt.savefig(out_dir / "eval_detection.png", dpi=120); plt.close()

    with open(out_dir / "eval_arrays.json", "w") as f:
        json.dump(arrays, f)
    print(f"Eval plots + arrays saved to {out_dir}/")
