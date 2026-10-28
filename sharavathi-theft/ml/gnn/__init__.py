"""
ml/gnn/ — GraphSAGE theft detector/localizer (Step 5). Needs torch + torch_geometric.

data.py   scenario → fixed-topology node tensor (39 nodes, 12 features)
model.py  GNNLocalizer (3 SAGE layers + pooled graph context; node + graph heads)
train.py  3-seed training → data/models/gnn.pt, gnn_results.json

Keep/drop vs XGBoost is decided in ml/evaluate.py (metrics.json → gnn_decision).
"""
