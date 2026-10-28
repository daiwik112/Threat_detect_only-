"""
rag/ingest.py — Build the advisor's knowledge-base index.

    python -m rag.ingest            # (re)build data/models/rag_index.joblib

Sources come from config/rag.yaml. Markdown is split at headings (the heading path is kept
as `section`); long sections are split into ~chunk_chars pieces at paragraph boundaries
with a one-paragraph overlap. Two TF-IDF vectorizers (word 1–2-grams, char_wb 3–5-grams)
are fitted on the chunk texts. The index stores a hash of the corpus contents, so a changed
corpus is detected and rebuilt automatically.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml

ROOT = Path(__file__).resolve().parents[1]          # sharavathi-theft/
CONFIG = ROOT / "config" / "rag.yaml"


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    source_type: str
    path: str
    text: str
    tags: list[str] = field(default_factory=list)


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def collect_sources(cfg: dict) -> list[tuple[Path, str]]:
    seen, out = set(), []
    for src in cfg["sources"]:
        for p in sorted(ROOT.glob(src["glob"])):
            rp = p.resolve()
            if p.is_file() and rp not in seen:
                seen.add(rp)
                out.append((rp, src["type"]))
    return out


def _front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    return (yaml.safe_load(m.group(1)) or {}), text[m.end():]


def _split_long(text: str, limit: int) -> list[str]:
    paras = []
    for p in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not p:
            continue
        if len(p) <= limit:
            paras.append(p)
            continue
        group = ""                                   # oversized paragraph: split at sentences
        for sent in re.split(r"(?<=[.!?])\s+", p):
            if group and len(group) + len(sent) > limit:
                paras.append(group.strip())
                group = ""
            group += sent + " "
        if group.strip():
            paras.append(group.strip())
    pieces, cur = [], []
    for p in paras:
        if cur and sum(len(x) for x in cur) + len(p) > limit:
            pieces.append("\n\n".join(cur))
            cur = [cur[-1]] if len(cur[-1]) < limit // 2 else []      # one-paragraph overlap
        cur.append(p)
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces


def chunk_markdown(text: str, doc_id: str, source_type: str, path: str, limit: int = 900) -> list[Chunk]:
    meta, body = _front_matter(text.replace("\r\n", "\n"))
    title = meta.get("title")
    tags = [str(t) for t in meta.get("tags", [])]
    heads: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    cur: list[str] = []
    in_code = False
    for line in body.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            if "\n".join(cur).strip():
                sections.append((" › ".join(heads), cur))
            level, name = len(m.group(1)), m.group(2).strip()
            if level == 1 and title is None:
                title = name
            heads = heads[: level - 1] + [name]
            cur = []
        else:
            cur.append(line)
    if "\n".join(cur).strip():
        sections.append((" › ".join(heads), cur))
    title = title or doc_id

    chunks: list[Chunk] = []
    for section, lines in sections:
        body_text = "\n".join(lines).strip()
        if len(body_text) < 40:          # skip near-empty fragments
            continue
        for piece in _split_long(body_text, limit):
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, section or title,
                                source_type, path, piece, tags))
    return chunks


CHUNKER_VERSION = "3"         # bump when chunking / indexing logic changes → forces a rebuild


def corpus_hash(sources: list[tuple[Path, str]], cfg: dict) -> str:
    h = hashlib.sha256(f"chunker-v{CHUNKER_VERSION}|".encode() + yaml.safe_dump(cfg, sort_keys=True).encode())
    for p, t in sources:
        h.update(f"{p.name}|{t}|".encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_index(cfg: dict | None = None, save: bool = True) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    cfg = cfg or load_config()
    sources = collect_sources(cfg)
    chunks: list[Chunk] = []
    for p, stype in sources:
        doc_id = p.stem
        rel = str(p.relative_to(ROOT.parent)) if ROOT.parent in p.parents else str(p)
        chunks += chunk_markdown(p.read_text(encoding="utf-8"), doc_id, stype, rel, cfg["chunk_chars"])
    if not chunks:
        raise RuntimeError("RAG corpus is empty — check config/rag.yaml sources")

    texts = [f"{c.section}. {c.text}" for c in chunks]      # section path already starts with the title
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    Xw = word.fit_transform(texts)
    # Dense "semantic" vectors: LSA (truncated SVD of the word TF-IDF) stored in the local vector DB.
    # LSA links passages that share related vocabulary even without exact term overlap.
    from sklearn.decomposition import TruncatedSVD
    from vectordb.store import VectorStore
    dims = max(2, min(128, len(chunks) - 1, Xw.shape[1] - 1))
    svd = TruncatedSVD(n_components=dims, random_state=0)
    dense = VectorStore(dims, "rag_passages")
    dense.add([c.id for c in chunks], svd.fit_transform(Xw),
              [{"doc_id": c.doc_id, "source_type": c.source_type} for c in chunks])
    index = {
        "chunks": [asdict(c) for c in chunks],
        "word": word, "Xw": Xw,
        "char": char, "Xc": char.fit_transform(texts),
        "svd": svd, "dense": dense,
        "hash": corpus_hash(sources, cfg),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
    }
    if save:
        out = ROOT / cfg["index_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(index, out)
    return index


def load_or_build(cfg: dict | None = None) -> dict:
    """Load the saved index; rebuild if missing or the corpus changed."""
    cfg = cfg or load_config()
    path = ROOT / cfg["index_path"]
    current = corpus_hash(collect_sources(cfg), cfg)
    if path.exists():
        idx = joblib.load(path)
        if idx.get("hash") == current:
            return idx
    return build_index(cfg)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    idx = build_index()
    docs = {}
    for c in idx["chunks"]:
        docs.setdefault((c["source_type"], c["doc_id"]), 0)
        docs[(c["source_type"], c["doc_id"])] += 1
    print(f"RAG index: {len(idx['chunks'])} chunks from {len(docs)} documents (hash {idx['hash']})")
    for (t, d), n in sorted(docs.items()):
        print(f"  {t:<12} {d:<40} {n:>3} chunks")
