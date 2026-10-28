"""
grid/ — pandapower network model for Sharavathi 220 kV corridor.

Workflow 01 — PANDAPOWER_GRID (all 8 tasks complete)
------------------------------------------------------
1. config.py    : load & validate YAML configs with pydantic (raises ConfigError if source missing)
2. build.py     : build_net(topology, load_scale, theft) -> pp.pandapowerNet
                  Creates HV buses, tap buses, line segments, loads, optional theft injection
3. flow.py      : run_pf(net) -> dict  (Newton-Raphson AC PF, raises PFNonConvergence cleanly)
4. sensors.py   : measure(results, profile, sensors_cfg, ...) -> dict
                  Gaussian noise + bias / drift / missing / stuck faults
5. estimator.py : estimate(net, measurements, sensors_cfg) -> EstimationResult | None
                  WLS state estimation; returns chi-square J, normalised residuals
6. baseline.py  : generate_baseline(topology, n_points=200) -> pd.DataFrame
                  Sweeps load_scale in [0.5, 1.15], saves data/raw/baseline_states.parquet
7. viz_check.py : plot_all(net, results, ...) — voltage profile, loading, losses vs load
8. CLI          : python -m grid.build --check  (prints convergence, losses, voltages)
"""
