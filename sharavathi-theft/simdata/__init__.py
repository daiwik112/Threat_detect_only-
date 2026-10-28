"""
simdata/ — Synthetic scenario dataset generator (WF2).

Modules
-------
operating_points : generate base load-scale operating points with group IDs
scenarios        : ScenarioSpec + one builder function per class
generate         : parallel joblib loop → data/processed/dataset.parquet
splits           : group-split + leave-location-out → splits.json
profiles         : regenerate under all sensor profiles for placement study
dataset_card     : write dataset_card.md with counts / histograms / assumptions
"""
