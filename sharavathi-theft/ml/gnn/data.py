"""
ml/gnn/data.py — Turn scenarios into fixed-topology graph tensors.

Graph: 39 nodes (substations + tap buses, pandapower order), 38 line segments as
bidirectional edges. Same topology for every scenario, so node features are stored
as one dense array X[n_scenarios, n_nodes, F] and batched by offsetting edge_index.

Node features (F = 12). Unmetered nodes get zeros + metered_flag = 0, so the model
must infer theft location through message passing / the pooled graph context.
  0 vm_dev        measured − expected voltage (pu ×100)
  1 va_dev        measured − expected angle (deg)
  2 p_inj_resid   KCL residual: bus P injection + incident metered line-end P (MW)
  3 q_inj_resid   same for Q
  4 dp_up         upstream line-end P (into this bus) minus expected   (MW)
  5 dp_dn         downstream line-end P (out of this bus) minus expected (MW)
  6 dq_up, 7 dq_dn   same for Q
  8 missing_frac  fraction of this node's channels that are NaN
  9 metered_flag  1 if the profile meters this bus
 10 km_norm       km / 361
 11 is_junction   1 for substations (load / junction buses)
Expected values: baseline_states interpolated at the estimated load scale (ml.features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["vm_dev", "va_dev", "p_inj_resid", "q_inj_resid", "dp_up", "dp_dn",
            "dq_up", "dq_dn", "missing_frac", "metered_flag", "km_norm", "is_junction"]
N_STANDARDIZE = 8          # first 8 features are standardized with train statistics


def edge_index(n_nodes: int) -> np.ndarray:
    src = np.arange(n_nodes - 1)
    return np.vstack([np.r_[src, src + 1], np.r_[src + 1, src]])


def build_node_tensor(df: pd.DataFrame, baseline_df: pd.DataFrame,
                      profile: str = "major_only") -> np.ndarray:
    """Return X[n, n_nodes, F] (float32, NaN → 0). Uses only metered channels."""
    from ml.features import _col, _expected, estimate_load_scale, get_grid_meta

    meta = get_grid_meta(profile)
    n, nn_ = len(df), len(meta.bus_names)
    X = np.zeros((n, nn_, len(FEATURES)), dtype=np.float32)
    ls, _ = estimate_load_scale(df, meta)
    exp = lambda c: _expected(baseline_df, c, ls)

    for i, b in enumerate(meta.bus_names):
        X[:, i, 10] = meta.bus_km[b] / 361.0
        X[:, i, 11] = 0.0 if b.startswith("TAP_") else 1.0
        if b not in meta.metered:
            continue
        X[:, i, 9] = 1.0
        chans = [_col(df, f"m_{b}_{k}") for k in ("vm", "va", "p", "q")]
        vals = {
            0: (chans[0] - exp(f"vm_{i}")) * 100.0,
            1: chans[1] - exp(f"va_{i}"),
        }
        kcl_p, kcl_q = chans[2].copy(), chans[3].copy()
        if i > 0:                                   # line i-1 ends here (p_to < 0 when delivering)
            up_p, up_q = _col(df, f"m_LINE{i-1}_p_to"), _col(df, f"m_LINE{i-1}_q_to")
            vals[4] = -(up_p - exp(f"p_to_line{i-1}"))
            vals[6] = -(up_q - exp(f"q_to_line{i-1}"))
            kcl_p, kcl_q = kcl_p + up_p, kcl_q + up_q
            chans += [up_p, up_q]
        if i < nn_ - 1:                             # line i starts here
            dn_p, dn_q = _col(df, f"m_LINE{i}_p_from"), _col(df, f"m_LINE{i}_q_from")
            vals[5] = dn_p - exp(f"p_from_line{i}")
            vals[7] = dn_q - exp(f"q_from_line{i}")
            kcl_p, kcl_q = kcl_p + dn_p, kcl_q + dn_q
            chans += [dn_p, dn_q]
        vals[2], vals[3] = kcl_p, kcl_q
        for j, v in vals.items():
            X[:, i, j] = np.nan_to_num(v, nan=0.0)
        X[:, i, 8] = np.mean([np.isnan(c) for c in chans], axis=0)
    return X


def node_targets(df: pd.DataFrame, profile: str = "major_only") -> np.ndarray:
    """Index of the theft node per row (−1 if not theft)."""
    from ml.features import get_grid_meta
    idx = get_grid_meta(profile).bus_idx
    tap = df["true_tap_id"].where(df["label"] == "theft")
    return np.array([idx.get(t, -1) if isinstance(t, str) else -1 for t in tap], dtype=np.int64)
