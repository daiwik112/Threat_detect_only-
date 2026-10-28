"""
rag/retriever.py — Hybrid lexical retrieval over the advisor corpus.

score = (0.45·cos_word + 0.25·cos_char + 0.30·cos_dense) · source_weight + 0.08·[tags ∩ boost_tags]
cos_dense comes from the LSA passage vectors in the local vector DB (vectordb.VectorStore).
TF-IDF vectors are L2-normalised by TfidfVectorizer, so a dot product is the cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

W_WORD, W_CHAR, W_DENSE, TAG_BOOST = 0.45, 0.25, 0.30, 0.08


@dataclass
class Hit:
    chunk: dict
    score: float


class Retriever:
    def __init__(self, index: dict):
        self.index = index
        self.chunks: list[dict] = index["chunks"]
        weights = index["config"].get("source_weight", {})
        self._prior = np.array([weights.get(c["source_type"], 1.0) for c in self.chunks])

    def scores(self, query: str, boost_tags: set[str] | None = None) -> np.ndarray:
        ix = self.index
        qw = ix["word"].transform([query])
        s = W_WORD * (ix["Xw"] @ qw.T).toarray().ravel() \
            + W_CHAR * (ix["Xc"] @ ix["char"].transform([query]).T).toarray().ravel()
        if "dense" in ix:                                   # vector-DB similarity (LSA space)
            s = s + W_DENSE * np.clip(ix["dense"].scores(ix["svd"].transform(qw)), 0, None)
        s = s * self._prior
        if boost_tags:
            s = s + TAG_BOOST * np.array([bool(boost_tags & set(c["tags"])) for c in self.chunks])
        return s

    def search(self, query: str, k: int = 6, boost_tags: set[str] | None = None,
               max_per_doc: int = 2, source_types: set[str] | None = None) -> list[Hit]:
        s = self.scores(query, boost_tags)
        hits, per_doc = [], {}
        for i in np.argsort(-s):
            if s[i] <= 0:
                break
            if source_types and self.chunks[i]["source_type"] not in source_types:
                continue
            d = self.chunks[i]["doc_id"]
            if per_doc.get(d, 0) >= max_per_doc:          # keep sources diverse
                continue
            per_doc[d] = per_doc.get(d, 0) + 1
            hits.append(Hit(self.chunks[i], float(s[i])))
            if len(hits) == k:
                break
        return hits

    def best_with_tag(self, query: str, tag: str) -> Hit | None:
        s = self.scores(query)
        cand = [i for i, c in enumerate(self.chunks) if tag in c["tags"]]
        if not cand:
            return None
        i = max(cand, key=lambda j: s[j])
        return Hit(self.chunks[i], float(s[i]))

    @property
    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(c["doc_id"], {"doc_id": c["doc_id"], "title": c["title"],
                                              "source_type": c["source_type"], "n_chunks": 0})
            d["n_chunks"] += 1
        return list(docs.values())
