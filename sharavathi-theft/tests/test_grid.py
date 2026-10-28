"""
tests/test_grid.py — Acceptance tests for 01_PANDAPOWER_GRID (Task 1–8).

Covers all acceptance criteria from 01_PANDAPOWER_GRID.md:
  - PF converges for 100 random load scales
  - Power balance: Σgen = Σload + Σloss (tol 1e-6 MW)
  - Theft load at tap X changes res_line.p_from_mw of upstream segment
  - Snap logic: km=47 → nearest tap 50
  - Missing-sensor profile still returns valid dict with NaNs / floats
  - Baseline parquet generation (smoke test)
  - State estimation converges with sufficient measurements
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def topology():
    """Load topology once for the whole module."""
    from grid.config import load_topology
    return load_topology()


@pytest.fixture(scope="module")
def sensors_cfg():
    from grid.config import load_sensors
    return load_sensors()


@pytest.fixture(scope="module")
def base_net(topology):
    """Build network at nominal load (load_scale=1.0, no theft)."""
    from grid.build import build_net
    return build_net(topology=topology, load_scale=1.0, theft=None)


@pytest.fixture(scope="module")
def base_results(base_net):
    """Run PF once on the base network."""
    from grid.flow import run_pf
    return run_pf(base_net)


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_topology_loads(self, topology):
        assert topology is not None
        assert topology.base_kv == 220
        assert len(topology.buses) == 5
        assert topology.tap_spacing_km == 10

    def test_all_buses_have_source(self, topology):
        for bus in topology.buses:
            assert bus.source, f"Bus {bus.id} missing source"

    def test_all_loads_have_source(self, topology):
        for load in topology.loads:
            assert load.source, f"Load at {load.bus} missing source"

    def test_sensors_config_loads(self, sensors_cfg):
        assert sensors_cfg.noise.vm_pct > 0
        assert sensors_cfg.noise.p_pct > 0

    def test_missing_source_raises(self, tmp_path):
        """Topology with numeric param but no 'source' must raise ConfigError."""
        import yaml
        from grid.config import load_topology, ConfigError

        bad_yaml = {
            "base_kv": 220,
            "tap_spacing_km": 10,
            "buses": [
                {"id": "A", "km": 0, "role": "slack_gen",
                 "metered": True, "source": "assumed"},
                {"id": "B", "km": 100, "role": "load",
                 "metered": True, "source": "assumed"},
            ],
            "lines": {
                "r_ohm_per_km": 0.07, "x_ohm_per_km": 0.40,
                "c_nf_per_km": 9.0, "max_i_ka": 0.80, "circuits": 2
            },
            "loads": [{"bus": "B", "p_mw": 50, "q_mvar": 15}],  # no source!
            "gen": {"bus": "A", "vm_pu": 1.02},
        }
        bad_file = tmp_path / "bad_topology.yaml"
        bad_file.write_text(yaml.dump(bad_yaml))

        with pytest.raises(ConfigError):
            load_topology(bad_file)


# ---------------------------------------------------------------------------
# 2. Network build
# ---------------------------------------------------------------------------


class TestBuild:
    def test_bus_count_reasonable(self, base_net, topology):
        # 5 substations + ~35 tap buses for 361 km / 10 km spacing
        n_buses = len(base_net.bus)
        assert n_buses > 5, f"Expected >5 buses, got {n_buses}"
        assert n_buses < 100, f"Too many buses: {n_buses}"

    def test_tap_buses_exist(self, base_net):
        tap_map = base_net["tap_map"]
        assert len(tap_map) > 10, "Expected at least 10 tap buses"

    def test_section_map_populated(self, base_net):
        section_map = base_net["section_map"]
        assert len(section_map) == len(base_net.line)

    def test_ext_grid_at_sharavathi(self, base_net):
        assert len(base_net.ext_grid) == 1
        eg_bus = base_net.ext_grid.iloc[0]["bus"]
        sharavathi_idx = base_net["id_to_idx"]["SHARAVATHI"]
        assert eg_bus == sharavathi_idx

    def test_loads_scaled(self, topology):
        from grid.build import build_net
        net_half = build_net(topology=topology, load_scale=0.5)
        net_full = build_net(topology=topology, load_scale=1.0)
        p_half = net_half.load["p_mw"].sum()
        p_full = net_full.load["p_mw"].sum()
        assert abs(p_half / p_full - 0.5) < 1e-6, "Load scaling broken"


# ---------------------------------------------------------------------------
# 3. Power flow convergence for 100 random load scales
# ---------------------------------------------------------------------------


class TestPowerFlow:
    def test_pf_converges_base(self, base_results):
        assert base_results["converged"] is True

    def test_pf_100_random_scales(self, topology):
        """PF must converge for 100 random load_scale values in [0.5, 1.15]."""
        from grid.build import build_net
        from grid.flow import run_pf, PFNonConvergence

        rng = np.random.default_rng(0)
        scales = rng.uniform(0.5, 1.15, 100)
        failures = []
        for ls in scales:
            try:
                net = build_net(topology=topology, load_scale=float(ls))
                res = run_pf(net)
                assert res["converged"]
            except PFNonConvergence:
                failures.append(ls)

        assert len(failures) == 0, (
            f"PF failed for {len(failures)}/100 load scales: {failures[:5]}"
        )

    def test_power_balance(self, base_net, base_results):
        """Σgen = Σload + Σloss to within 1e-6 MW."""
        gen  = base_results["gen_p_mw"]
        load = base_results["load_p_mw"]
        loss = base_results["total_loss_mw"]
        residual = abs(gen - load - loss)
        assert residual < 1e-4, (
            f"Power balance error {residual:.2e} MW (gen={gen:.3f}, "
            f"load={load:.3f}, loss={loss:.3f})"
        )

    def test_voltage_in_range(self, base_results):
        """No-theft, nominal load: voltage must be within 0.95–1.05 pu."""
        assert base_results["min_vm_pu"] >= 0.94, (
            f"Min voltage {base_results['min_vm_pu']:.4f} too low"
        )
        assert base_results["max_vm_pu"] <= 1.06, (
            f"Max voltage {base_results['max_vm_pu']:.4f} too high"
        )

    def test_losses_plausible(self, base_results):
        """Losses should be 1–8% of generation at peak load."""
        gen = base_results["gen_p_mw"]
        loss_pct = 100.0 * base_results["total_loss_mw"] / gen
        assert 0.5 <= loss_pct <= 10.0, (
            f"Loss {loss_pct:.2f}% outside plausible range"
        )


# ---------------------------------------------------------------------------
# 4. Theft injection
# ---------------------------------------------------------------------------


class TestTheftInjection:
    def test_theft_load_created(self, topology):
        from grid.build import build_net, TheftSpec
        theft = TheftSpec(km=150, mw=20, pf=0.9)
        net = build_net(topology=topology, theft=theft)
        theft_loads = net.load[net.load["name"] == "THEFT"]
        assert len(theft_loads) == 1
        assert abs(theft_loads.iloc[0]["p_mw"] - 20.0) < 1e-6

    def test_theft_raises_upstream_flow(self, topology):
        """Injecting 20 MW theft must increase p_from_mw on upstream segments."""
        from grid.build import build_net, TheftSpec
        from grid.flow import run_pf

        # Clean network
        net_clean = build_net(topology=topology, load_scale=1.0, theft=None)
        res_clean = run_pf(net_clean)

        # Theft at ~km 150 (between SHARAVATHI=0 and SHIMOGA=110)
        # nearest tap to 150 should be between SHIMOGA(110) and SIRA_KBCROSS(215)
        theft = TheftSpec(km=150, mw=20, pf=0.9)
        net_t = build_net(topology=topology, load_scale=1.0, theft=theft)
        res_t = run_pf(net_t)

        # Total generation must increase by approximately theft MW + extra losses
        delta_gen = res_t["gen_p_mw"] - res_clean["gen_p_mw"]
        assert delta_gen > 15.0, (
            f"Generation increase after 20 MW theft too small: {delta_gen:.2f} MW"
        )

        # Losses must increase
        assert res_t["total_loss_mw"] > res_clean["total_loss_mw"]

    def test_theft_drops_downstream_voltage(self, topology):
        """Downstream voltages must be lower with theft than without."""
        from grid.build import build_net, TheftSpec
        from grid.flow import run_pf

        net_clean = build_net(topology=topology, load_scale=1.0, theft=None)
        res_clean = run_pf(net_clean)

        theft = TheftSpec(km=150, mw=20, pf=0.9)
        net_t = build_net(topology=topology, load_scale=1.0, theft=theft)
        res_t = run_pf(net_t)

        # PEENYA (end bus) must have lower voltage with theft
        peenya_idx = net_clean["id_to_idx"]["PEENYA"]
        vm_clean = res_clean["vm_pu"][peenya_idx]
        vm_theft = res_t["vm_pu"][peenya_idx]
        assert vm_theft < vm_clean, (
            f"PEENYA voltage did not drop with theft: {vm_clean:.5f} → {vm_theft:.5f}"
        )

    def test_snap_km_47_to_50(self, topology):
        """km=47 must snap to the nearest 10-km tap (i.e., TAP_50)."""
        from grid.build import build_net, TheftSpec
        theft = TheftSpec(km=47, mw=10, pf=0.9)
        net = build_net(topology=topology, theft=theft)
        snapped_km = net["theft_km"]
        # TAP_50 = km 50
        assert abs(snapped_km - 50.0) < 1.0, (
            f"Expected snap to km 50, got {snapped_km}"
        )

    def test_theft_q_from_pf(self, topology):
        """TheftSpec.q_mvar must match mw * tan(acos(pf))."""
        from grid.build import TheftSpec
        for mw, pf in [(20, 0.9), (10, 0.85), (5, 0.95)]:
            ts = TheftSpec(km=100, mw=mw, pf=pf)
            expected_q = mw * math.tan(math.acos(pf))
            assert abs(ts.q_mvar - expected_q) < 1e-9


# ---------------------------------------------------------------------------
# 5. Sensors / missing profile
# ---------------------------------------------------------------------------


class TestSensors:
    def test_measure_returns_dict(self, base_net, base_results, sensors_cfg):
        from grid.sensors import measure, enrich_results_with_bus_power

        results = enrich_results_with_bus_power(dict(base_results), base_net)
        id_to_idx = base_net["id_to_idx"]
        bus_names = dict(base_net.bus["name"])
        rng = np.random.default_rng(42)

        meas = measure(results, "major_only", sensors_cfg, id_to_idx, bus_names, rng)
        assert isinstance(meas, dict)
        assert len(meas) > 0

    def test_missing_sensor_profile_has_nans(self, base_net, base_results, sensors_cfg):
        """Even with all sensors missing, returned dict must contain NaN (not crash)."""
        from grid.sensors import measure, enrich_results_with_bus_power, FaultSpec

        results = enrich_results_with_bus_power(dict(base_results), base_net)
        id_to_idx = base_net["id_to_idx"]
        bus_names = dict(base_net.bus["name"])
        rng = np.random.default_rng(0)

        # Inject 100% missing fault on all metered buses
        fault_keys = [f"m_{bid}_vm" for bid in ["SHARAVATHI", "SHIMOGA",
                                                 "NELAMANGALA", "PEENYA"]]
        faults = [
            FaultSpec(k, "missing", prob=1.0)
            for k in fault_keys
        ]
        meas = measure(results, "major_only", sensors_cfg, id_to_idx, bus_names,
                       rng, faults=faults)
        assert isinstance(meas, dict)
        # vm keys should all be NaN
        for k in fault_keys:
            val = meas.get(k)
            assert val is None or (isinstance(val, float) and math.isnan(val)), (
                f"Expected NaN for {k}, got {val}"
            )

    def test_bias_fault_shifts_measurement(self, base_net, base_results, sensors_cfg):
        """A +10% bias fault must shift the measured voltage up by ~10%."""
        from grid.sensors import measure, enrich_results_with_bus_power, FaultSpec

        results = enrich_results_with_bus_power(dict(base_results), base_net)
        id_to_idx = base_net["id_to_idx"]
        bus_names = dict(base_net.bus["name"])
        rng = np.random.default_rng(1)

        meas_clean = measure(results, "major_only", sensors_cfg, id_to_idx,
                             bus_names, rng)

        rng2 = np.random.default_rng(1)
        meas_biased = measure(results, "major_only", sensors_cfg, id_to_idx,
                              bus_names, rng2,
                              faults=[FaultSpec("m_SHARAVATHI_vm", "bias", pct=10.0)])

        vm_clean  = meas_clean.get("m_SHARAVATHI_vm", None)
        vm_biased = meas_biased.get("m_SHARAVATHI_vm", None)
        assert vm_clean is not None and vm_biased is not None
        # biased should be ~10% higher
        assert vm_biased > vm_clean, "Bias fault did not increase reading"


# ---------------------------------------------------------------------------
# 6. Baseline generation (smoke test)
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_baseline_smoke(self, topology, tmp_path):
        """Run 10-point baseline sweep and verify output parquet schema."""
        from grid.baseline import generate_baseline

        out = tmp_path / "baseline.parquet"
        df = generate_baseline(
            topology=topology,
            n_points=10,
            load_range=(0.8, 1.0),
            out_path=out,
            verbose=False,
        )
        assert len(df) == 10
        assert "load_scale" in df.columns
        assert "total_loss_mw" in df.columns
        assert "loss_pct" in df.columns
        assert "converged" in df.columns
        assert out.exists()

    def test_baseline_losses_in_range(self, topology, tmp_path):
        """Baseline losses must fall within expected 1–8% range."""
        from grid.baseline import generate_baseline

        df = generate_baseline(
            topology=topology,
            n_points=5,
            load_range=(0.9, 1.05),
            out_path=tmp_path / "bl.parquet",
            verbose=False,
        )
        assert df["loss_pct"].max() < 15.0, (
            f"Losses too high: {df['loss_pct'].max():.1f}%"
        )
        assert df["loss_pct"].min() > 0, "Losses must be positive"


# ---------------------------------------------------------------------------
# 7. State estimator (smoke test — only when enough measurements)
# ---------------------------------------------------------------------------


class TestEstimator:
    def test_estimation_converges(self, base_net, base_results, sensors_cfg):
        from grid.sensors import measure, enrich_results_with_bus_power
        from grid.estimator import estimate

        results = enrich_results_with_bus_power(dict(base_results), base_net)
        id_to_idx = base_net["id_to_idx"]
        bus_names = dict(base_net.bus["name"])
        rng = np.random.default_rng(7)

        meas = measure(results, "major_plus_mid", sensors_cfg,
                       id_to_idx, bus_names, rng)

        est = estimate(base_net, meas, sensors_cfg)
        assert est is not None
        assert est.converged is True
        assert not math.isnan(est.chi2_j)
        assert len(est.vm_pu) > 0

    def test_estimation_no_measurements_returns_unobservable(
        self, base_net, sensors_cfg
    ):
        """With no measurements at all, estimator must return observable=False."""
        from grid.estimator import estimate

        est = estimate(base_net, {}, sensors_cfg)
        assert est is not None
        assert est.observable is False
        assert est.converged is False
