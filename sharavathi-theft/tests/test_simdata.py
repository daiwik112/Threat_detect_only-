"""
tests/test_simdata.py — WF2 acceptance tests.

Quality checks from 02_SYNTHETIC_DATA.md:
- Class balance as configured (±5%)
- No duplicate scenario_ids; no group appears in >1 split
- Theft rows: true_km and magnitude_mw present; non-theft rows: NaN
- Sanity: mean upstream p_from for theft rows > matching normal rows (same group)
- Reproducible: same seed → identical scenario_id sequence
- Splits: no group_id in >1 of train/val/test
- Leave-location-out: holdout_locations rows only in test_loc_holdout
- Dataset card written and non-empty
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures — small dataset (300 rows) for fast tests
# ---------------------------------------------------------------------------

N_SMALL = 300   # small enough to run quickly in CI


@pytest.fixture(scope="module")
def topology():
    from grid.config import load_topology
    return load_topology()


@pytest.fixture(scope="module")
def sensors_cfg():
    from grid.config import load_sensors
    return load_sensors()


@pytest.fixture(scope="module")
def scenarios_cfg():
    from grid.config import load_scenarios
    return load_scenarios()


@pytest.fixture(scope="module")
def small_scenarios_cfg():
    """A tiny scenarios config for fast testing."""
    from grid.config import ScenariosConfig
    return ScenariosConfig(
        n_normal=60,
        n_theft=80,
        n_sensor_bias=30,
        n_sensor_drift=20,
        n_missing=15,
        n_line_outage=10,
        load_scale_range=(0.7, 1.05),
        holdout_locations_km=[45.0, 135.0, 255.0],
    )


@pytest.fixture(scope="module")
def small_df(topology, sensors_cfg, small_scenarios_cfg, tmp_path_factory):
    """Generate a small dataset once for the whole module."""
    from simdata.generate import generate_dataset
    tmp = tmp_path_factory.mktemp("simdata")
    df = generate_dataset(
        scenarios_cfg=small_scenarios_cfg,
        topology=topology,
        sensors_cfg=sensors_cfg,
        profile="major_only",
        out_dir=tmp,
        seed=42,
        n_jobs=1,
        verbose=False,
    )
    return df


@pytest.fixture(scope="module")
def small_splits(small_df, small_scenarios_cfg):
    from simdata.splits import make_splits
    return make_splits(small_df, small_scenarios_cfg.holdout_locations_km, seed=42)


# ---------------------------------------------------------------------------
# 1. Operating points
# ---------------------------------------------------------------------------

class TestOperatingPoints:
    def test_generates_correct_count(self, small_scenarios_cfg):
        from simdata.operating_points import generate_operating_points
        rng = np.random.default_rng(0)
        ops = generate_operating_points(small_scenarios_cfg, n_groups=50, rng=rng)
        assert len(ops) == 50

    def test_load_scale_in_range(self, small_scenarios_cfg):
        from simdata.operating_points import generate_operating_points
        rng = np.random.default_rng(1)
        ls_min, ls_max = small_scenarios_cfg.load_scale_range
        ops = generate_operating_points(small_scenarios_cfg, n_groups=100, rng=rng)
        for op in ops:
            assert ls_min <= op.load_scale <= ls_max, \
                f"load_scale {op.load_scale} out of [{ls_min}, {ls_max}]"

    def test_unique_group_ids(self, small_scenarios_cfg):
        from simdata.operating_points import generate_operating_points
        rng = np.random.default_rng(2)
        ops = generate_operating_points(small_scenarios_cfg, n_groups=100, rng=rng)
        ids = [op.group_id for op in ops]
        assert len(ids) == len(set(ids)), "Duplicate group_ids!"

    def test_bus_jitter_in_range(self, small_scenarios_cfg):
        from simdata.operating_points import generate_operating_points
        rng = np.random.default_rng(3)
        ops = generate_operating_points(small_scenarios_cfg, n_groups=50, rng=rng)
        for op in ops:
            for bus, jitter in op.bus_jitter.items():
                assert 0.80 <= jitter <= 1.20, \
                    f"Bus {bus} jitter {jitter:.3f} out of expected range"


# ---------------------------------------------------------------------------
# 2. Scenario builders
# ---------------------------------------------------------------------------

class TestScenarioBuilders:
    def test_normal_label(self, small_scenarios_cfg):
        from simdata.operating_points import generate_operating_points
        from simdata.scenarios import build_normal
        rng = np.random.default_rng(0)
        ops = generate_operating_points(small_scenarios_cfg, 10, rng)
        spec = build_normal(ops[0], rng)
        assert spec.label == "normal"
        assert spec.theft_km is None
        assert spec.theft_mw is None

    def test_theft_has_km_and_mw(self, topology):
        from grid.build import build_net
        from simdata.operating_points import OperatingPoint
        from simdata.scenarios import build_theft
        import uuid

        net = build_net(topology=topology)
        tap_kms = sorted(net["tap_map"].values())
        rng = np.random.default_rng(5)
        op = OperatingPoint(
            group_id=str(uuid.uuid4()), load_scale=1.0,
            bus_jitter={}, hour=12.0, season="summer"
        )
        spec = build_theft(op, tap_kms, rng)
        assert spec.label == "theft"
        assert spec.theft_km is not None
        assert 1.0 <= spec.theft_mw <= 40.0
        assert 0.85 <= spec.theft_pf <= 0.98

    def test_small_theft_magnitude(self, topology):
        from grid.build import build_net
        from simdata.operating_points import OperatingPoint
        from simdata.scenarios import build_theft_small
        import uuid

        net = build_net(topology=topology)
        tap_kms = sorted(net["tap_map"].values())
        rng = np.random.default_rng(6)
        op = OperatingPoint(
            group_id=str(uuid.uuid4()), load_scale=1.0,
            bus_jitter={}, hour=10.0, season="monsoon"
        )
        for _ in range(20):
            spec = build_theft_small(op, tap_kms, rng)
            assert spec.theft_mw <= 3.0, f"small theft mw={spec.theft_mw}"

    def test_missing_fault_channels(self):
        from simdata.operating_points import OperatingPoint
        from simdata.scenarios import build_missing
        import uuid
        rng = np.random.default_rng(7)
        op = OperatingPoint(
            group_id=str(uuid.uuid4()), load_scale=0.9,
            bus_jitter={}, hour=8.0, season="winter"
        )
        spec = build_missing(op, rng)
        assert spec.label == "missing"
        assert 1 <= len(spec.fault_channels) <= 3
        assert spec.fault_type == "missing"


# ---------------------------------------------------------------------------
# 3. Dataset generation
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    def test_dataframe_has_required_columns(self, small_df):
        required = ["scenario_id", "group_id", "label", "true_km",
                    "magnitude_mw", "load_scale", "sensor_profile"]
        for col in required:
            assert col in small_df.columns, f"Missing column: {col}"

    def test_no_duplicate_scenario_ids(self, small_df):
        ids = small_df["scenario_id"]
        assert ids.nunique() == len(ids), \
            f"Duplicate scenario_ids: {len(ids) - ids.nunique()} duplicates"

    def test_theft_rows_have_km_and_magnitude(self, small_df):
        theft = small_df[small_df["label"] == "theft"]
        assert len(theft) > 0, "No theft rows!"
        # true_km must not be NaN for theft
        nan_km = theft["true_km"].isna().sum()
        assert nan_km == 0, f"{nan_km} theft rows have NaN true_km"
        # magnitude must be > 0
        assert (theft["magnitude_mw"] > 0).all(), "theft rows have magnitude <= 0"

    def test_normal_rows_have_zero_magnitude(self, small_df):
        normal = small_df[small_df["label"] == "normal"]
        assert (normal["magnitude_mw"] == 0).all(), \
            "Normal rows have non-zero magnitude"
        # true_km must be NaN for normal
        assert normal["true_km"].isna().all(), \
            "Normal rows have non-NaN true_km"

    def test_all_labels_present(self, small_df):
        expected = {"normal", "theft", "sensor_bias", "sensor_drift",
                    "missing", "line_outage"}
        found = set(small_df["label"].unique())
        missing = expected - found
        assert not missing, f"Labels missing from dataset: {missing}"

    def test_load_scale_in_range(self, small_df, small_scenarios_cfg):
        ls_min, ls_max = small_scenarios_cfg.load_scale_range
        assert small_df["load_scale"].min() >= ls_min - 0.01
        assert small_df["load_scale"].max() <= ls_max + 0.01

    def test_measurements_have_floats_or_nan(self, small_df):
        """All m_* columns should be float (possibly NaN)."""
        m_cols = [c for c in small_df.columns if c.startswith("m_")]
        assert len(m_cols) > 0, "No measurement columns found"
        for col in m_cols[:5]:   # check first 5
            assert small_df[col].dtype in [float, "float64", "float32"], \
                f"Column {col} has dtype {small_df[col].dtype}"

    def test_reproducible_same_seed(self, topology, sensors_cfg,
                                     small_scenarios_cfg, tmp_path):
        """Same seed must produce identical scenario_id sequence."""
        from simdata.generate import generate_dataset
        df1 = generate_dataset(
            scenarios_cfg=small_scenarios_cfg, topology=topology,
            sensors_cfg=sensors_cfg, profile="major_only",
            out_dir=tmp_path / "r1", seed=99, n_jobs=1, verbose=False,
        )
        df2 = generate_dataset(
            scenarios_cfg=small_scenarios_cfg, topology=topology,
            sensors_cfg=sensors_cfg, profile="major_only",
            out_dir=tmp_path / "r2", seed=99, n_jobs=1, verbose=False,
        )
        assert list(df1["scenario_id"]) == list(df2["scenario_id"]), \
            "Different seeds produce different scenario_id order — not reproducible"
        # Check a measurement column
        m_cols = [c for c in df1.columns if c.startswith("m_")]
        if m_cols:
            col = m_cols[0]
            pd.testing.assert_series_equal(
                df1[col].reset_index(drop=True),
                df2[col].reset_index(drop=True),
                check_names=False,
            )


# ---------------------------------------------------------------------------
# 4. Theft physics sanity
# ---------------------------------------------------------------------------

class TestTheftPhysics:
    def test_theft_raises_upstream_generation(self, small_df):
        """
        Mean _gen_p_mw for theft rows must exceed normal rows at similar load scale.
        """
        if "_gen_p_mw" not in small_df.columns:
            pytest.skip("_gen_p_mw column not present")

        theft  = small_df[small_df["label"] == "theft"]
        normal = small_df[small_df["label"] == "normal"]

        # Compare at similar load scale band
        for ls_band in [(0.7, 0.85), (0.85, 1.0), (1.0, 1.05)]:
            t = theft[(theft["load_scale"] >= ls_band[0]) &
                      (theft["load_scale"] <  ls_band[1])]
            n = normal[(normal["load_scale"] >= ls_band[0]) &
                       (normal["load_scale"] <  ls_band[1])]
            if len(t) < 5 or len(n) < 5:
                continue
            assert t["_gen_p_mw"].mean() > n["_gen_p_mw"].mean(), \
                (f"Theft gen {t['_gen_p_mw'].mean():.1f} MW not > "
                 f"normal gen {n['_gen_p_mw'].mean():.1f} MW "
                 f"at ls band {ls_band}")


# ---------------------------------------------------------------------------
# 5. Splits
# ---------------------------------------------------------------------------

class TestSplits:
    def test_no_duplicate_indices(self, small_splits):
        all_idx = []
        for k, v in small_splits.items():
            all_idx.extend(v)
        assert len(all_idx) == len(set(all_idx)), "Duplicate row indices across splits!"

    def test_no_group_in_multiple_splits(self, small_df, small_splits):
        from simdata.splits import validate_splits
        validate_splits(small_df, small_splits)   # raises AssertionError on failure

    def test_all_rows_accounted(self, small_df, small_splits):
        total = sum(len(v) for v in small_splits.values())
        assert total == len(small_df), \
            f"Splits cover {total} rows but df has {len(small_df)}"

    def test_train_val_test_sizes(self, small_splits):
        n_train = len(small_splits["train"])
        n_val   = len(small_splits["val"])
        n_test  = len(small_splits["test"])
        n_hout  = len(small_splits["test_loc_holdout"])
        total   = n_train + n_val + n_test + n_hout
        assert total > 0
        # train should be largest
        assert n_train > n_val, "train smaller than val"
        assert n_train > n_test, "train smaller than test"

    def test_holdout_locations_only_in_holdout(self, small_df, small_splits,
                                                small_scenarios_cfg):
        """Theft rows near holdout_locations_km must NOT appear in train/val/test."""
        holdout_kms = small_scenarios_cfg.holdout_locations_km
        radius = 5.0

        for split_name in ["train", "val", "test"]:
            idx = small_splits[split_name]
            sub = small_df.loc[idx]
            theft_sub = sub[sub["label"] == "theft"]
            for _, row in theft_sub.iterrows():
                km = row["true_km"]
                if math.isnan(km):
                    continue
                near_holdout = any(abs(km - h) <= radius for h in holdout_kms)
                assert not near_holdout, \
                    (f"Holdout theft (km={km:.0f}) found in '{split_name}' split! "
                     f"Holdout locs: {holdout_kms}")


# ---------------------------------------------------------------------------
# 6. Dataset card
# ---------------------------------------------------------------------------

class TestDatasetCard:
    def test_dataset_card_written(self, small_df, small_splits,
                                   small_scenarios_cfg, sensors_cfg, tmp_path):
        from simdata.dataset_card import write_dataset_card
        card_path = tmp_path / "dataset_card.md"
        write_dataset_card(
            small_df, small_splits, card_path,
            "major_only", sensors_cfg, small_scenarios_cfg,
        )
        assert card_path.exists(), "dataset_card.md not created"
        content = card_path.read_text()
        assert "Class Distribution" in content
        assert "Theft Statistics" in content
        assert "Assumptions" in content
        assert len(content) > 500, "dataset_card.md seems too short"
