"""
BM25-based sparse keyword retriever using rank-bm25.

Usage:
    retriever = BM25Retriever(corpus)
    results = retriever.retrieve("customer support query", top_k=10)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenization with basic punctuation removal."""
    import re
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


class BM25Retriever:
    """
    Wraps BM25Okapi from rank-bm25 for document retrieval.

    Parameters
    ----------
    documents : list of dict
        Each document must have at least a 'text' field and an 'id' field.
    k1 : float
        BM25 k1 term-frequency saturation parameter (default 1.5).
    b : float
        BM25 b length-normalization parameter (default 0.75).
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        tokenized_corpus = [_tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        logger.info("BM25Retriever indexed %d documents.", len(documents))

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a query.

        Returns
        -------
        list of dict
            Each item contains the original document fields plus 'bm25_score' and 'rank'.
        """
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            result = dict(self.documents[idx])
            result["bm25_score"] = float(scores[idx])
            result["rank"] = rank
            result["retriever"] = "bm25"
            results.append(result)

        return results

    @classmethod
    def from_texts(cls, texts: List[str], **kwargs) -> "BM25Retriever":
        """Convenience constructor from a plain list of text strings."""
        documents = [{"id": str(i), "text": t} for i, t in enumerate(texts)]
        return cls(documents, **kwargs)
