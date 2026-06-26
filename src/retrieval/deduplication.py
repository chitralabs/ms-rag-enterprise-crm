"""
SHA-256 content deduplication for retrieved passages.

Duplicate passages (identical or near-identical text) inflate LLM context
without adding information. This module removes exact duplicates before
the fused candidate set is passed to the generation layer.

Usage:
    clean = deduplicate(passages)
    clean = deduplicate(passages, similarity_threshold=0.9)  # near-duplicate removal
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    """Return the SHA-256 hex digest of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate(
    passages: List[Dict[str, Any]],
    text_field: str = "text",
    similarity_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Remove duplicate passages from a candidate list.

    Exact deduplication is performed via SHA-256 hashing of the passage text.
    Optionally, near-duplicate removal is applied using cosine similarity of
    TF-IDF vectors when `similarity_threshold` is provided.

    Parameters
    ----------
    passages : list of dict
        Ranked passages, each containing at least a `text_field` key.
    text_field : str
        Key in each passage dict that holds the text content.
    similarity_threshold : float or None
        If set (0–1), passages with pairwise cosine similarity above this
        threshold are considered near-duplicates; only the higher-ranked
        one is kept. Adds sklearn dependency when used.

    Returns
    -------
    list of dict
        Deduplicated passages preserving original rank order, each enriched
        with a 'content_hash' field.
    """
    seen_hashes: set = set()
    unique_passages: List[Dict[str, Any]] = []

    for passage in passages:
        text = passage.get(text_field, "")
        digest = _sha256(text)
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            result = dict(passage)
            result["content_hash"] = digest
            unique_passages.append(result)

    n_removed = len(passages) - len(unique_passages)
    if n_removed:
        logger.debug("Exact deduplication: removed %d duplicate passage(s).", n_removed)

    if similarity_threshold is not None:
        unique_passages = _near_duplicate_filter(unique_passages, text_field, similarity_threshold)

    return unique_passages


def _near_duplicate_filter(
    passages: List[Dict[str, Any]],
    text_field: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    """Remove near-duplicate passages using TF-IDF cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if len(passages) <= 1:
        return passages

    texts = [p[text_field] for p in passages]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf)

    keep = []
    dropped = set()
    for i in range(len(passages)):
        if i in dropped:
            continue
        keep.append(passages[i])
        for j in range(i + 1, len(passages)):
            if sim_matrix[i, j] >= threshold:
                dropped.add(j)

    logger.debug(
        "Near-duplicate filter (threshold=%.2f): removed %d passage(s).",
        threshold,
        len(passages) - len(keep),
    )
    return keep
