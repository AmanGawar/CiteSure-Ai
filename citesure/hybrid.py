"""Hybrid BM25 + vector search with Reciprocal Rank Fusion (Phase 12)."""
from rank_bm25 import BM25Okapi


def rrf(rank_lists, k=60):
    scores = {}
    for ranks in rank_lists:
        for pos, doc_id in enumerate(ranks, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + pos)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(store, query, paper_ids=None, k=6):
    docs = store.get_chunks(paper_ids)
    if not docs:
        return []
    bm25 = BM25Okapi([d["text"].lower().split() for d in docs])
    sc = bm25.get_scores(query.lower().split())
    bm_rank = [docs[i]["id"] for i in sorted(range(len(docs)), key=lambda i: -sc[i])[:20]]
    vec_rank = [h["id"] for h in store.search(query, k=20, paper_ids=paper_ids)]
    by_id = {d["id"]: d for d in docs}
    return [by_id[i] for i in rrf([bm_rank, vec_rank])[:k]]
