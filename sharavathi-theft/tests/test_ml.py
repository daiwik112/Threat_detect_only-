"""
tests/test_ml.py — WF3 acceptance tests.

Fast tests train small models on a stratified slice; the artifact tests at the end
check the outputs of `make train` (skipped if it has not been run).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATASET = Path("data/processed/dataset_major_only.parquet")
BASELINE = Path("data/raw/baseline_states.parquet")
MODELS = Path("data/models")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_full():
    if not DATASET.exists():
        pytest.skip("dataset_major_only.parquet not found — run WF2 first")
    return pd.read_parquet(DATASET)


@pytest.fixture(scope="module")
def baseline_df():
    if not BASELINE.exists():
        pytest.skip("baseline_states.parquet not found — run WF1 first")
    return pd.read_parquet(BASELINE)


@pytest.fixture(scope="module")
def dataset(raw_full):
    """Stratified 2000-row slice (the parquet is sorted by class, so head() is all-normal)."""
    return (raw_full.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), 400), random_state=0))
            .reset_index(drop=True))


@pytest.fixture(scope="module")
def df_feat(dataset, baseline_df):
    from ml.features import build_feature_matrix
    return build_feature_matrix(dataset, baseline_df)


@pytest.fixture(scope="module")
def feat_cols(df_feat):
    from ml.features import get_feature_cols
    return get_feature_cols(df_feat)


@pytest.fixture(scope="module")
def tvt(df_feat):
    """Group-wise 60/20/20 split (no group_id in two parts)."""
    groups = df_feat["group_id"].unique()
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    n = len(groups)
    g_tr, g_vl = set(groups[: int(0.6 * n)]), set(groups[int(0.6 * n): int(0.8 * n)])
    tr = df_feat[df_feat["group_id"].isin(g_tr)]
    vl = df_feat[df_feat["group_id"].isin(g_vl)]
    te = df_feat[~df_feat["group_id"].isin(g_tr | g_vl)]
    return tr, vl, te


@pytest.fixture(scope="module")
def clf(tvt, feat_cols):
    from ml.xgb_classifier import XGBClassifier
    tr, vl, _ = tvt
    return XGBClassifier(n_estimators=60, learning_rate=0.2).fit(tr, vl, feat_cols)


@pytest.fixture(scope="module")
def reg(tvt, feat_cols):
    from ml.localize import LocalizerXGBReg
    tr, vl, _ = tvt
    return LocalizerXGBReg(n_estimators=100, learning_rate=0.2).fit(tr, vl, feat_cols)


@pytest.fixture(scope="module")
def rank(tvt, feat_cols):
    from ml.localize import LocalizerSectionRank
    tr, vl, _ = tvt
    return LocalizerSectionRank(n_estimators=40, learning_rate=0.3).fit(tr, vl, feat_cols)


@pytest.fixture(scope="module")
def bd(tvt):
    from ml.detector_baseline import BaselineDetector
    return BaselineDetector().fit(tvt[0])


# ---------------------------------------------------------------------------
# 1. Features
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_shape(self, dataset, df_feat, feat_cols):
        assert len(df_feat) == len(dataset)
        assert len(feat_cols) >= 30

    def test_no_label_leakage(self, feat_cols):
        for c in feat_cols:
            for bad in ("label", "true_", "magnitude", "load_scale_true", "group"):
                assert bad not in c, c

    def test_true_load_scale_not_used(self, dataset, baseline_df):
        """Changing label columns (incl. true load_scale) must not change any feature."""
        from ml.features import build_feature_matrix, get_feature_cols
        d2 = dataset.copy()
        d2["load_scale"] = 0.5
        d2["true_km"] = 0.0
        d2["magnitude_mw"] = 99.0
        d2["label"] = "normal"
        a = build_feature_matrix(dataset, baseline_df)
        b = build_feature_matrix(d2, baseline_df)
        fc = get_feature_cols(a)
        pd.testing.assert_frame_equal(a[fc], b[fc])

    def test_unmetered_tap_channels_ignored(self, dataset, baseline_df):
        """Tap-segment flows are not metered under major_only → must not influence features."""
        from ml.features import build_feature_matrix, get_feature_cols, get_grid_meta, metered_channels
        keep = set(metered_channels(get_grid_meta("major_only")))
        d2 = dataset.copy()
        for c in d2.columns:
            if c.startswith("m_LINE") and c not in keep:
                d2[c] = np.nan
        fc = get_feature_cols(build_feature_matrix(dataset, baseline_df))
        pd.testing.assert_frame_equal(build_feature_matrix(dataset, baseline_df)[fc],
                                      build_feature_matrix(d2, baseline_df)[fc])

    def test_sections(self):
        from ml.features import get_grid_meta
        m = get_grid_meta("major_only")
        assert [s.name for s in m.sections] == [
            "SHARAVATHI-SHIMOGA", "SHIMOGA-NELAMANGALA", "NELAMANGALA-PEENYA"]
        assert len(get_grid_meta("major_plus_mid").sections) == 4

    def test_load_scale_estimate(self, df_feat):
        n = df_feat[df_feat["label"] == "normal"]
        assert (n["f_load_scale_est"] - n["load_scale"]).abs().max() < 0.03

    def test_residual_tracks_theft_mw(self, df_feat):
        t = df_feat[(df_feat["label"] == "theft") & (df_feat["magnitude_mw"] >= 10)]
        assert np.corrcoef(t["f_r_p_total"], t["magnitude_mw"])[0, 1] > 0.8

    def test_normal_residuals_small(self, df_feat):
        n = df_feat[df_feat["label"] == "normal"]
        assert n["f_r_p_total"].abs().median() < 5.0


# ---------------------------------------------------------------------------
# 2. Baseline detector
# ---------------------------------------------------------------------------

class TestBaselineDetector:
    def test_fpr_near_target_on_train_normals(self, tvt, bd):
        n = tvt[0][tvt[0]["label"] == "normal"]
        assert bd.predict(n).mean() <= 0.02

    def test_score_in_range_and_threshold(self, tvt, bd):
        s = bd.score(tvt[2])
        assert ((s >= 0) & (s <= 1)).all()
        np.testing.assert_array_equal(bd.predict(tvt[2]), (bd.statistic(tvt[2]) > 1).astype(int))

    def test_metrics_schema(self, tvt, bd):
        te = tvt[2]
        m = bd.metrics(te, (te["label"] != "normal").astype(int).to_numpy())
        for k in ("precision", "recall", "f1", "fpr", "roc_auc", "recall_at_fpr_1pct"):
            assert k in m

    def test_save_load(self, tvt, bd, tmp_path):
        from ml.detector_baseline import BaselineDetector
        bd.save(tmp_path / "bd.joblib")
        np.testing.assert_array_equal(bd.predict(tvt[2]),
                                      BaselineDetector.load(tmp_path / "bd.joblib").predict(tvt[2]))


# ---------------------------------------------------------------------------
# 3. XGBoost classifier
# ---------------------------------------------------------------------------

class TestXGBClassifier:
    def test_shapes(self, tvt, clf):
        te = tvt[2]
        assert clf.predict(te).shape == (len(te),)
        p = clf.predict_proba_binary(te)
        assert ((p >= 0) & (p <= 1)).all()
        pm = clf.predict_proba_multi(te)
        assert pm.shape == (len(te), 6)
        np.testing.assert_allclose(pm.sum(axis=1), 1.0, atol=1e-5)

    def test_handles_nan_features(self, tvt, clf, feat_cols):
        te = tvt[2].head(5).copy()
        te[feat_cols[:5]] = np.nan
        assert np.isfinite(clf.predict_proba_binary(te)).all()

    def test_likely_cause(self, tvt, clf):
        for c in clf.likely_cause(tvt[2].head(10)):
            assert set(c) == {"extraction", "sensor_error", "other"}
            assert abs(sum(c.values()) - 1.0) < 1e-6

    def test_shap(self, tvt, clf, feat_cols):
        tops = clf.shap_top(tvt[2].head(4), n=3)
        assert len(tops) == 4 and len(tops[0]) == 3
        assert tops[0][0]["feature"] in feat_cols

    def test_metrics_schema(self, tvt, clf):
        m = clf.metrics(tvt[2])
        assert set(m) == {"detection", "classification"}
        assert np.array(m["classification"]["confusion"]).shape == (6, 6)

    def test_beats_chance(self, tvt, clf):
        assert clf.metrics(tvt[2])["detection"]["roc_auc"] > 0.8


# ---------------------------------------------------------------------------
# 4. Localization
# ---------------------------------------------------------------------------

class TestLocalization:
    def test_reg_predict(self, tvt, reg):
        th = tvt[2][tvt[2]["label"] == "theft"]
        est, lo, hi = reg.predict(th)
        assert est.shape == (len(th),)
        assert (lo <= est).all() and (est <= hi).all()
        assert ((est >= 0) & (est <= 361)).all()

    def test_reg_metrics(self, tvt, reg):
        m = reg.metrics(tvt[2])
        for k in ("mae_km", "median_km", "within_10km", "within_25km", "within_50km",
                  "interval_coverage", "by_magnitude"):
            assert k in m
        assert m["mae_km"] < 90          # far better than random (~120 km)

    def test_rank_predict(self, tvt, rank):
        th = tvt[2][tvt[2]["label"] == "theft"].head(5)
        est, lo, hi, ranked = rank.predict(th)
        assert len(ranked) == 5
        probs = [t["prob"] for t in ranked[0]]
        assert probs == sorted(probs, reverse=True) and sum(probs) <= 1.0 + 1e-6
        assert all(lo <= hi)
        secs = rank.ranked_sections(th)[0]
        assert abs(sum(s["prob"] for s in secs) - 1.0) < 1e-5

    def test_smallest_window(self):
        from ml.localize import _smallest_window
        assert _smallest_window(np.array([0.05, 0.1, 0.7, 0.1, 0.05]), 0.8) == (1, 2)
        assert _smallest_window(np.array([1.0]), 0.8) == (0, 0)

    def test_save_load(self, tvt, reg, tmp_path):
        from ml.localize import LocalizerXGBReg
        reg.save(tmp_path / "r.joblib")
        r2 = LocalizerXGBReg.load(tmp_path / "r.joblib")
        np.testing.assert_allclose(reg.predict(tvt[2])[0], r2.predict(tvt[2])[0])


# ---------------------------------------------------------------------------
# 5. Evaluation helpers
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_recall_at_fpr(self):
        from ml.evaluate import recall_at_fpr
        y = np.array([0] * 100 + [1] * 100)
        s = np.r_[np.linspace(0, 1, 100), np.linspace(2, 3, 100)]
        assert recall_at_fpr(y, s, 0.01) == 1.0

    def test_physics_localize(self, df_feat):
        from ml.evaluate import physics_localize
        th = df_feat[df_feat["label"] == "theft"]
        est, secs = physics_localize(th)
        assert est.shape == (len(th),) and len(secs) == len(th)


# ---------------------------------------------------------------------------
# 6. Infer API (schemas from 08_DATA_CONTRACTS)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle(bd, clf, reg, rank, baseline_df, feat_cols):
    from ml.infer import ModelBundle
    return ModelBundle(baseline_detector=bd, xgb_classifier=clf, localizer_reg=reg,
                       localizer_rank=rank, baseline_df=baseline_df, feat_cols=feat_cols)


def _meas(row: pd.Series) -> dict:
    return {k: (None if pd.isna(v) else float(v)) for k, v in row.items() if k.startswith("m_")}


class TestInferAPI:
    def test_detect_schema(self, bundle, dataset):
        r = bundle.detect(_meas(dataset[dataset["label"] == "theft"].iloc[0]))
        assert r["status"] in ("normal", "suspicious")
        assert 0 <= r["anomaly_score"] <= 1
        assert len(r["residuals"]) == 3
        assert set(r["residuals"][0]) == {"section", "r_p_mw", "r_q_mvar", "z"}
        assert set(r["likely_cause"]) == {"extraction", "sensor_error", "other"}
        assert all(set(t) == {"feature", "shap"} for t in r["top_contributors"])
        json.dumps(r)                           # JSON-serialisable

    def test_detect_matches_batch_features(self, bundle, df_feat, clf):
        """Raw-measurement inference must reproduce the training-time features."""
        row = df_feat.iloc[[0]]
        r = bundle.detect(_meas(row.iloc[0]))
        assert abs(r["anomaly_score"] - clf.predict_proba_binary(row)[0]) < 1e-3

    def test_detect_handles_missing(self, bundle, dataset):
        m = _meas(dataset.iloc[0])
        m["m_SHIMOGA_p"] = None
        assert bundle.detect(m)["status"] in ("normal", "suspicious")

    @pytest.mark.parametrize("method", ["xgb_reg", "section_rank"])
    def test_localize_schema(self, bundle, dataset, method):
        r = bundle.localize(_meas(dataset[dataset["label"] == "theft"].iloc[0]), method=method)
        assert r["method"] == method
        lo, hi = r["interval_km"]
        assert lo <= hi
        assert 0 <= r["est_km"] <= 361
        assert all(set(t) == {"tap_id", "km", "prob"} for t in r["ranked_taps"])
        assert all(set(s) == {"section", "prob"} for s in r["ranked_sections"])
        json.dumps(r)

    def test_gnn_availability_check(self):
        from ml.gnn.model import is_gnn_available
        assert isinstance(is_gnn_available(), bool)


# ---------------------------------------------------------------------------
# 7. Artifacts from `make train`
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def metrics():
    p = MODELS / "metrics.json"
    if not p.exists():
        pytest.skip("metrics.json not found — run `make train`")
    return json.loads(p.read_text())


class TestArtifacts:
    def test_metrics_response_keys(self, metrics):
        assert set(metrics["detection"]) >= {"precision", "recall", "f1", "fpr"}
        assert set(metrics["classification"]) >= {"confusion", "per_class_recall"}
        assert set(metrics["localization"]) >= {"mae_km", "within_10km", "within_25km", "within_50km"}
        assert set(metrics["baseline"]) >= {"precision", "recall", "f1", "fpr", "mae_km"}

    def test_beats_baseline_somewhere(self, metrics):
        """03 acceptance: beat the residual baseline on recall@FPR or localization MAE."""
        b = metrics["beats_baseline"]
        assert b.get("recall_at_fpr_1pct") or b.get("localization_mae_km"), b

    def test_version_stamp(self, metrics):
        assert metrics["artifact_meta"]["dataset_hash"]
        meta = json.loads((MODELS / "model_meta.json").read_text())
        assert meta["dataset_hash"] == metrics["artifact_meta"]["dataset_hash"]

    def test_load_models(self):
        from ml.infer import load_models
        b = load_models(MODELS)
        assert b.xgb_classifier is not None and b.localizer_reg is not None
        assert b.xgb_classifier.artifact_meta["version"]

    def test_sensor_study_schema(self):
        p = MODELS / "sensor_study.json"
        if not p.exists():
            pytest.skip("sensor_study.json not found — run `make sensor-study`")
        s = json.loads(p.read_text())
        for c in s["configs"]:
            assert {"name", "detection_recall", "fpr", "mae_km"} <= set(c)


# ---------------------------------------------------------------------------
# 8. GNN (Step 5) — skipped if torch / torch_geometric are missing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gnn(tvt, baseline_df):
    from ml.gnn.model import GNNLocalizer, is_gnn_available
    if not is_gnn_available():
        pytest.skip("torch / torch_geometric not installed")
    tr, vl, _ = tvt
    return GNNLocalizer(epochs=3, hidden=16, n_threads=2).fit(tr, vl, baseline_df, verbose=False)


class TestGNN:
    def test_node_tensor(self, dataset, baseline_df):
        from ml.gnn.data import FEATURES, build_node_tensor
        X = build_node_tensor(dataset.head(20), baseline_df)
        assert X.shape == (20, 39, len(FEATURES))
        assert np.isfinite(X).all()
        unmetered = X[:, [i for i in range(39) if X[0, i, 9] == 0], :9]
        assert (unmetered == 0).all()                 # taps carry no measurement info

    def test_predict_schema(self, gnn, tvt):
        th = tvt[2][tvt[2]["label"] == "theft"].head(5)
        est, lo, hi, ranked = gnn.predict(th)
        assert est.shape == (5,) and (lo <= hi).all()
        assert gnn.predict_proba(th).sum(axis=1) == pytest.approx(1.0, abs=1e-4)
        assert all(t["tap_id"].startswith("TAP_") for t in ranked[0])
        assert {"mae_km", "top3_within_10km"} <= set(gnn.metrics(tvt[2]))

    def test_unseen_taps_are_candidates(self, gnn):
        """Unlike the XGB tap classifier, the GNN can rank held-out taps (e.g. TAP_40)."""
        assert "TAP_40" in [gnn.bus_names[i] for i in gnn.cand_idx]

    def test_save_load(self, gnn, tvt, baseline_df, tmp_path):
        from ml.gnn.model import GNNLocalizer
        gnn.save(tmp_path / "g.pt")
        g2 = GNNLocalizer.load(tmp_path / "g.pt", baseline_df)
        np.testing.assert_allclose(gnn.predict(tvt[2].head(8))[0], g2.predict(tvt[2].head(8))[0],
                                   rtol=1e-5)

    def test_infer_gnn_method(self, bundle, gnn, dataset):
        bundle.gnn = gnn
        r = bundle.localize(_meas(dataset[dataset["label"] == "theft"].iloc[0]), method="gnn")
        assert r["method"] == "gnn" and len(r["ranked_taps"]) == 10
        json.dumps(r)
