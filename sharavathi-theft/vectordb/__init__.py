"""
vectordb/ — Small local vector database (no server, no network).

store.VectorStore   persisted float32 matrix + metadata, cosine top-k with metadata filters
cases.CaseIndex     "similar past cases": embeds a scenario as its standardized physics-feature
                    vector (ml.features) and returns the nearest labelled dataset scenarios

Used by rag/ (dense LSA passage vectors) and the backend (/api/similar-cases).
The VectorStore API (add / search / save / load) is deliberately minimal so a hosted vector DB
(Chroma, FAISS, pgvector) can replace it without touching callers.
"""
