"""tests/test_vectordb.py — local vector store (vectordb/)."""
import numpy as np
import pytest

from vectordb.store import VectorStore


def test_add_search_upsert_filter_roundtrip(tmp_path):
    vs = VectorStore(3)
    vs.add(["a", "b", "c"], np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]]),
           [{"label": "x"}, {"label": "y"}, {"label": "x"}])
    hits = vs.search(np.array([1, 0.1, 0]), k=2)
    assert [h.id for h in hits] == ["a", "c"] and hits[0].score == pytest.approx(0.995, abs=1e-3)
    assert [h.id for h in vs.search(np.array([1, 0, 0]), k=3, where={"label": "y"})] == ["b"]
    assert "a" not in [h.id for h in vs.search(np.array([1, 0, 0]), k=3, exclude_ids={"a"})]
    vs.add(["a"], np.array([[0, 0, 1]]), [{"label": "z"}])          # upsert overwrites
    assert len(vs) == 3 and vs.search(np.array([0, 0, 1]), k=1)[0].metadata == {"label": "z"}
    vs.save(tmp_path / "v.joblib")
    vs2 = VectorStore.load(tmp_path / "v.joblib")
    assert [h.id for h in vs2.search(np.array([0, 1, 0]), k=1)] == ["b"]
    with pytest.raises(ValueError):
        vs.add(["d"], np.ones((1, 4)))


def test_rag_index_has_dense_vectors():
    from rag.ingest import build_index
    idx = build_index(save=False)
    assert len(idx["dense"]) == len(idx["chunks"]) and idx["dense"].dim >= 2
