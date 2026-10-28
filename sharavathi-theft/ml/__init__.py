"""
ml/ — Machine Learning package (WF3).

Modules
-------
features          : physics-based feature engineering (f_* columns)
detector_baseline : residual threshold + chi-square benchmark detector
xgb_classifier    : XGBoost 6-class + binary anomaly classifier
localize          : XGBRegressor (km) + section-ranking localizer
gnn/              : optional PyTorch Geometric GNN (feature_flags: gnn_enabled)
sensor_study      : run classify+localize under all profiles/noise levels
evaluate          : compute metrics.json (detection, classification, localization)
infer             : load_models(), detect(), localize() — API-ready interface
"""
