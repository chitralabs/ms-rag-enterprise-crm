"""
Reciprocal Rank Fusion (RRF) for combining ranked result lists.

Reference: Cormack, Clarke & Buettcher (SIGIR 2009).

Usage:
    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=60, top_m=10)
"""

from __future__ import annotations

from typing import List, Dict, Any


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_m: int = 10,
    id_field: str = "id",
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for document d = Σ_r  1 / (k + rank_r(d))
    where rank_r(d) is the 1-based position of d in ranked list r.

    Parameters
    ----------
    ranked_lists : list of list of dict
        Each inner list is a ranked sequence of documents (rank 1 = best).
        Documents must share a common identifier under `id_field`.
    k : int
        RRF constant controlling the contribution of lower-ranked items (default 60).
    top_m : int
        Number of top-scoring documents to return after fusion.
    id_field : str
        Key used to identify documents across lists (default 'id').

    Returns
    -------
    list of dict
        Top-m documents sorted by descending RRF score, each enriched with
        'rrf_score', 'fused_rank', and 'source_ranks' (per-list ranks).
    """
    scores: Dict[str, float] = {}
    source_ranks: Dict[str, Dict[int, int]] = {}
    doc_store: Dict[str, Dict[str, Any]] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = str(doc[id_field])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            source_ranks.setdefault(doc_id, {})[list_idx] = rank
            if doc_id not in doc_store:
                doc_store[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)

    results = []
    for fused_rank, doc_id in enumerate(sorted_ids[:top_m], start=1):
        result = dict(doc_store[doc_id])
        result["rrf_score"] = scores[doc_id]
        result["fused_rank"] = fused_rank
        result["source_ranks"] = source_ranks[doc_id]
        results.append(result)

    return results
