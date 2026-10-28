"""
vectordb/store.py — Minimal persisted vector store.

    vs = VectorStore(dim=128)
    vs.add(ids, vectors, metadata)                    # vectors: (n, dim) array
    hits = vs.search(query_vec, k=5, where={"label": "theft"})
    vs.save(path)   /   VectorStore.load(path)

Vectors are L2-normalised on insert, so search is a matrix–vector dot product (exact cosine).
Brute force is ~1 ms for tens of thousands of rows at these dimensions, which is all this project
needs; swap in FAISS/HNSW behind the same interface if the corpus grows by orders of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np


@dataclass
class VectorHit:
    id: str
    score: float           # cosine similarity in [-1, 1]
    metadata: dict


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


class VectorStore:
    def __init__(self, dim: int, name: str = "store"):
        self.dim, self.name = dim, name
        self.ids: list[str] = []
        self.metadata: list[dict] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self._pos: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict] | None = None) -> None:
        """Insert or overwrite (upsert) rows."""
        vectors = _normalize(np.atleast_2d(vectors))
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        metadata = metadata or [{} for _ in ids]
        new_rows, new_meta, new_ids = [], [], []
        for i, (id_, md) in enumerate(zip(ids, metadata)):
            if id_ in self._pos:                         # overwrite in place
                j = self._pos[id_]
                self.vectors[j], self.metadata[j] = vectors[i], md
            else:
                new_ids.append(id_)
                new_rows.append(vectors[i])
                new_meta.append(md)
        if new_ids:
            start = len(self.ids)
            self.vectors = np.vstack([self.vectors, np.stack(new_rows)])
            self.ids += new_ids
            self.metadata += new_meta
            self._pos.update({id_: start + k for k, id_ in enumerate(new_ids)})

    def _mask(self, where: dict | Callable[[dict], bool] | None) -> np.ndarray | None:
        if where is None:
            return None
        if callable(where):
            return np.array([bool(where(m)) for m in self.metadata])
        def ok(m: dict) -> bool:
            for k, v in where.items():
                if isinstance(v, (list, tuple, set)):
                    if m.get(k) not in v:
                        return False
                elif m.get(k) != v:
                    return False
            return True
        return np.array([ok(m) for m in self.metadata])

    def scores(self, query: np.ndarray) -> np.ndarray:
        return self.vectors @ _normalize(query).ravel()

    def search(self, query: np.ndarray, k: int = 5,
               where: dict[str, Any] | Callable[[dict], bool] | None = None,
               exclude_ids: set[str] | None = None) -> list[VectorHit]:
        if len(self) == 0:
            return []
        s = self.scores(query)
        mask = self._mask(where)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        if exclude_ids:
            for id_ in exclude_ids:
                if id_ in self._pos:
                    s[self._pos[id_]] = -np.inf
        top = np.argsort(-s)[:k]
        return [VectorHit(self.ids[i], float(s[i]), self.metadata[i]) for i in top if np.isfinite(s[i])]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"dim": self.dim, "name": self.name, "ids": self.ids,
                     "metadata": self.metadata, "vectors": self.vectors}, path)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        d = joblib.load(path)
        vs = cls(d["dim"], d.get("name", "store"))
        vs.ids, vs.metadata, vs.vectors = d["ids"], d["metadata"], d["vectors"]
        vs._pos = {id_: i for i, id_ in enumerate(vs.ids)}
        return vs
