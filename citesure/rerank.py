"""Cross-encoder reranking of a retrieved shortlist (Phase 12)."""
from sentence_transformers import CrossEncoder

_reranker = None


def rerank(query, hits, top_n=5):
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    scores = _reranker.predict([(query, h["text"]) for h in hits])
    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    return [h for _, h in ranked[:top_n]]
