"""
ml/gnn/model.py — GraphSAGE theft detector + localizer (Step 5).

Architecture
------------
  x (12 node features, see ml/gnn/data.py)
  → Linear(12, H)
  → 3 × [SAGEConv(H, H) + LayerNorm + ReLU + Dropout]   (residual connections)
  → graph context g = [mean-pool ‖ max-pool]  (2H)
  → node head : MLP([h_node ‖ g ‖ km_norm]) → logit, softmax over candidate taps
  → graph head: MLP(g) → anomaly logit

Why the pooled context: in the 39-node chain the metered buses are up to 22 hops
apart, so 3 hops of message passing alone cannot reach mid-section taps. The pooled
context (a "virtual node") gives every node the whole-graph evidence while keeping
the GraphSAGE layers topology-aware. Held-out taps are ordinary candidates here
(unlike the XGB tap classifier, which never saw them).

Loss = BCE(graph anomaly)
     + soft-target CE over candidate taps (Gaussian around the true km, σ = label_sigma_km;
       without it the held-out taps are pure negatives and get suppressed)
     + λ·|E[km] − true_km| / 361   (directly optimizes localization MAE)   — theft rows only.

Public API
----------
is_gnn_available() -> bool
GNNLocalizer(seed, profile, ...)
    fit(train_df, val_df, baseline_df)
    predict(df) -> (est_km, lo, hi, ranked_taps)        # LocalizerSectionRank-compatible
    predict_proba_binary(df) -> anomaly score
    ranked_sections(df), metrics(df)
    save(path.pt) / GNNLocalizer.load(path.pt, baseline_df)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

_TORCH_AVAILABLE = False
_PYG_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
    from torch_geometric.nn import SAGEConv
    _PYG_AVAILABLE = True
except ImportError:
    pass


def is_gnn_available() -> bool:
    return _TORCH_AVAILABLE and _PYG_AVAILABLE


if is_gnn_available():
    class _Net(nn.Module):
        def __init__(self, in_ch: int, hidden: int, layers: int, dropout: float):
            super().__init__()
            self.inp = nn.Linear(in_ch, hidden)
            self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.drop = nn.Dropout(dropout)
            self.node_head = nn.Sequential(nn.Linear(3 * hidden + 1, hidden), nn.ReLU(),
                                           nn.Linear(hidden, 1))
            self.graph_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                            nn.Linear(hidden, 1))

        def forward(self, x, ei, B: int, N: int, km):
            h = F.relu(self.inp(x))
            for conv, norm in zip(self.convs, self.norms):
                h = h + self.drop(F.relu(norm(conv(h, ei))))
            H = h.view(B, N, -1)
            g = torch.cat([H.mean(1), H.max(1).values], dim=-1)            # (B, 2H)
            node_in = torch.cat([H, g.unsqueeze(1).expand(B, N, -1),
                                 km.view(1, N, 1).expand(B, N, 1)], dim=-1)
            return self.node_head(node_in).squeeze(-1), self.graph_head(g).squeeze(-1)


class GNNLocalizer:
    method = "gnn"

    def __init__(self, seed: int = 42, profile: str = "major_only", hidden: int = 64,
                 layers: int = 3, dropout: float = 0.1, lr: float = 2e-3,
                 weight_decay: float = 1e-4, epochs: int = 150, patience: int = 15,
                 batch_size: int = 256, coverage: float = 0.80, label_sigma_km: float = 10.0,
                 km_loss_weight: float = 2.0, n_threads: int | None = None):
        if not is_gnn_available():
            raise ImportError("GNN requires: pip install torch torch_geometric")
        self.cfg = dict(seed=seed, profile=profile, hidden=hidden, layers=layers,
                        dropout=dropout, lr=lr, weight_decay=weight_decay, epochs=epochs,
                        patience=patience, batch_size=batch_size, coverage=coverage,
                        label_sigma_km=label_sigma_km, km_loss_weight=km_loss_weight)
        self.n_threads = n_threads
        self.mu = self.sd = None
        self.net = None
        self.baseline_df: pd.DataFrame | None = None
        self.history: list[dict] = []
        self._init_graph()

    # ------------------------------------------------------------------
    def _init_graph(self):
        from ml.features import get_grid_meta
        from ml.gnn.data import edge_index
        meta = get_grid_meta(self.cfg["profile"])
        self.bus_names = meta.bus_names
        self.N = len(meta.bus_names)
        self.km = np.array([meta.bus_km[b] for b in meta.bus_names], dtype=np.float32)
        self.cand = np.array([b.startswith("TAP_") for b in meta.bus_names])
        self.cand_idx = np.where(self.cand)[0]           # sorted by km (chain order)
        self._ei = torch.as_tensor(edge_index(self.N), dtype=torch.long)
        self._ei_cache: dict[int, torch.Tensor] = {}

    def _batched_ei(self, B: int) -> "torch.Tensor":
        if B not in self._ei_cache:
            off = (torch.arange(B) * self.N).repeat_interleave(self._ei.shape[1])
            self._ei_cache[B] = self._ei.repeat(1, B) + off
        return self._ei_cache[B]

    def _tensor(self, df: pd.DataFrame) -> np.ndarray:
        from ml.gnn.data import N_STANDARDIZE, build_node_tensor
        X = build_node_tensor(df, self.baseline_df, self.cfg["profile"])
        X[:, :, :N_STANDARDIZE] = (X[:, :, :N_STANDARDIZE] - self.mu) / self.sd
        return X

    def _forward(self, X: np.ndarray, train: bool = False):
        B = X.shape[0]
        x = torch.as_tensor(X.reshape(B * self.N, -1))
        node, graph = self.net(x, self._batched_ei(B), B, self.N,
                               torch.as_tensor(self.km / 361.0))
        node = node.masked_fill(~torch.as_tensor(self.cand).view(1, -1), -1e9)
        return node, graph

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            baseline_df: pd.DataFrame, verbose: bool = True) -> "GNNLocalizer":
        from ml.gnn.data import FEATURES, N_STANDARDIZE, build_node_tensor, node_targets
        c = self.cfg
        if self.n_threads:
            torch.set_num_threads(self.n_threads)
        torch.manual_seed(c["seed"])
        np.random.seed(c["seed"])
        self.baseline_df = baseline_df

        Xtr_raw = build_node_tensor(train_df, baseline_df, c["profile"])
        met = Xtr_raw[:, :, 9] > 0
        vals = Xtr_raw[:, :, :N_STANDARDIZE][met]
        self.mu = vals.mean(0).astype(np.float32)
        self.sd = (vals.std(0) + 1e-6).astype(np.float32)
        Xtr = Xtr_raw.copy()
        Xtr[:, :, :N_STANDARDIZE] = (Xtr[:, :, :N_STANDARDIZE] - self.mu) / self.sd
        Xvl = self._tensor(val_df)
        ytr_g = (train_df["label"] != "normal").to_numpy(np.float32)
        yvl_g = (val_df["label"] != "normal").to_numpy(np.float32)
        ytr_n, yvl_n = node_targets(train_df, c["profile"]), node_targets(val_df, c["profile"])
        pos_w = torch.tensor((1 - ytr_g).sum() / max(ytr_g.sum(), 1))

        self.net = _Net(len(FEATURES), c["hidden"], c["layers"], c["dropout"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

        # soft targets: row t = Gaussian over candidate-tap km centred on node t
        d = self.km[:, None] - self.km[None, :]
        K = np.exp(-0.5 * (d / c["label_sigma_km"]) ** 2) * self.cand[None, :]
        K = torch.as_tensor(K / K.sum(1, keepdims=True), dtype=torch.float32)
        km_t = torch.as_tensor(self.km)

        def loss_fn(X, yg, yn, train):
            node, graph = self._forward(X, train)
            l = F.binary_cross_entropy_with_logits(graph, torch.as_tensor(yg), pos_weight=pos_w)
            t = torch.as_tensor(yn)
            m = t >= 0
            if m.any():
                logp = F.log_softmax(node[m], dim=-1)
                l = l - (K[t[m]] * logp).sum(-1).mean()
                e_km = (logp.exp() * km_t).sum(-1)
                l = l + c["km_loss_weight"] * (e_km - km_t[t[m]]).abs().mean() / 361.0
            return l

        best, best_state, wait = np.inf, None, 0
        rng = np.random.default_rng(c["seed"])
        t0 = time.time()
        for ep in range(c["epochs"]):
            self.net.train()
            perm = rng.permutation(len(Xtr))
            tot = 0.0
            for s in range(0, len(perm), c["batch_size"]):
                b = perm[s:s + c["batch_size"]]
                opt.zero_grad()
                l = loss_fn(Xtr[b], ytr_g[b], ytr_n[b], True)
                l.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += float(l) * len(b)
            self.net.eval()
            with torch.no_grad():
                vl = sum(float(loss_fn(Xvl[s:s + 1024], yvl_g[s:s + 1024], yvl_n[s:s + 1024], False))
                         * len(Xvl[s:s + 1024]) for s in range(0, len(Xvl), 1024)) / len(Xvl)
            sched.step(vl)
            self.history.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_loss": vl})
            if vl < best - 1e-4:
                best, wait = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= c["patience"]:
                    break
            if verbose and ep % 10 == 0:
                print(f"    ep {ep:3d} train {tot / len(Xtr):.4f} val {vl:.4f} ({time.time() - t0:.0f}s)")
        self.net.load_state_dict(best_state)
        self.best_val_loss = best
        if verbose:
            print(f"    stopped at epoch {ep}, best val loss {best:.4f} ({time.time() - t0:.0f}s)")
        return self

    # ------------------------------------------------------------------
    def _infer(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._tensor(df)
        self.net.eval()
        P, G = [], []
        with torch.no_grad():
            for s in range(0, len(X), 1024):
                node, graph = self._forward(X[s:s + 1024])
                P.append(torch.softmax(node, dim=-1).numpy())
                G.append(torch.sigmoid(graph).numpy())
        return np.concatenate(P), np.concatenate(G)

    def predict_proba_binary(self, df: pd.DataFrame) -> np.ndarray:
        return self._infer(df)[1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """(n, n_candidates) theft-location probabilities over candidate taps (km order)."""
        return self._infer(df)[0][:, self.cand_idx]

    def predict(self, df: pd.DataFrame):
        from ml.localize import _smallest_window
        P = self.predict_proba(df)
        kms = self.km[self.cand_idx]
        est = P @ kms
        lo, hi, ranked = [], [], []
        for p in P:
            a, b = _smallest_window(p, self.cfg["coverage"])
            lo.append(kms[a]); hi.append(kms[b])
            ranked.append([{"tap_id": self.bus_names[self.cand_idx[i]], "km": float(kms[i]),
                            "prob": float(p[i])} for i in np.argsort(-p)[:10]])
        return est, np.array(lo, float), np.array(hi, float), ranked

    def ranked_sections(self, df: pd.DataFrame) -> list[list[dict]]:
        from ml.features import get_grid_meta
        meta = get_grid_meta(self.cfg["profile"])
        sec = np.array([meta.section_of_km(k) for k in self.km[self.cand_idx]])
        return [sorted(({"section": s.name, "prob": float(p[sec == s.name].sum())}
                        for s in meta.sections), key=lambda d: -d["prob"])
                for p in self.predict_proba(df)]

    def metrics(self, df: pd.DataFrame) -> dict:
        from ml.features import get_grid_meta
        from ml.localize import localization_metrics
        sub = df[df["true_km"].notna()]
        if len(sub) == 0:
            return {}
        meta = get_grid_meta(self.cfg["profile"])
        est, lo, hi, ranked = self.predict(sub)
        true = sub["true_km"].to_numpy()
        hits = [any(abs(t["km"] - k) <= 10 for t in r[:3]) for r, k in zip(ranked, true)]
        return localization_metrics(true, est, lo, hi, sub["magnitude_mw"],
                                    [meta.section_of_km(k) for k in true],
                                    [r[0]["section"] for r in self.ranked_sections(sub)],
                                    top_k_hit=hits, method=self.method)

    # ------------------------------------------------------------------
    def save(self, path: Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg, "state_dict": self.net.state_dict(), "mu": self.mu,
                    "sd": self.sd, "history": self.history,
                    "artifact_meta": getattr(self, "artifact_meta", {}), **(extra or {})}, path)

    @staticmethod
    def load(path: Path, baseline_df: pd.DataFrame) -> "GNNLocalizer":
        from ml.gnn.data import FEATURES
        ck = torch.load(path, weights_only=False)
        m = GNNLocalizer(**ck["cfg"])
        m.net = _Net(len(FEATURES), ck["cfg"]["hidden"], ck["cfg"]["layers"], ck["cfg"]["dropout"])
        m.net.load_state_dict(ck["state_dict"])
        m.net.eval()
        m.mu, m.sd, m.history = ck["mu"], ck["sd"], ck.get("history", [])
        m.artifact_meta = ck.get("artifact_meta", {})
        m.baseline_df = baseline_df
        return m
