"""Embeddings + ChromaDB wrapper (Phase 3)."""
import chromadb

from .config import settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_model = None


def embedder():
    global _model
    if _model is None:                          # lazy-load: first call downloads ~130 MB
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.embed_model)
    return _model


def is_embedder_loaded() -> bool:
    return _model is not None


def embed_docs(texts):
    return embedder().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def embed_query(q):
    return embedder().encode(QUERY_PREFIX + q, normalize_embeddings=True).tolist()


def embed_queries(queries):
    return embedder().encode(
        [QUERY_PREFIX + query for query in queries],
        normalize_embeddings=True,
        batch_size=32,
    ).tolist()


def _where(paper_ids):
    return {"paper_id": {"$in": paper_ids}} if paper_ids else None


class PaperStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.col = self.client.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})

    def has_paper(self, paper_id: str) -> bool:
        return len(self.col.get(where={"paper_id": paper_id}, limit=1)["ids"]) > 0

    def add_chunks(self, chunks):
        self.col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embed_docs([c.text for c in chunks]),
            metadatas=[{"paper_id": c.paper_id, "title": c.title,
                        "section": c.section, "page": c.page} for c in chunks],
        )

    def search(self, query: str, k: int = 6, paper_ids=None):
        return self.search_many([query], k=k, paper_ids=paper_ids)[0]

    def search_many(self, queries, k: int = 6, paper_ids=None):
        if not queries:
            return []

        res = self.col.query(
            query_embeddings=embed_queries(queries),
            n_results=k,
            where=_where(paper_ids),
        )
        return [
            [
                {
                    "id": res["ids"][query_index][i],
                    "text": res["documents"][query_index][i],
                    "meta": res["metadatas"][query_index][i],
                    "score": 1 - res["distances"][query_index][i],
                }
                for i in range(len(res["ids"][query_index]))
            ]
            for query_index in range(len(queries))
        ]

    def get_chunks(self, paper_ids=None):
        r = self.col.get(where=_where(paper_ids), include=["documents", "metadatas"])
        return [{"id": i, "text": t, "meta": m, "score": 0.0}
                for i, t, m in zip(r["ids"], r["documents"], r["metadatas"])]

    def list_papers(self):
        seen = {}
        for m in self.col.get(include=["metadatas"])["metadatas"]:
            p = seen.setdefault(m["paper_id"], {"paper_id": m["paper_id"], "title": m["title"], "chunks": 0})
            p["chunks"] += 1
        return list(seen.values())

    def delete_paper(self, paper_id: str):
        self.col.delete(where={"paper_id": paper_id})
