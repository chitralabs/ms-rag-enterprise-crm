"""
Dense vector retriever using sentence-transformers + FAISS.

Usage:
    retriever = DenseRetriever(documents, model_name="all-MiniLM-L6-v2")
    results = retriever.retrieve("customer support query", top_k=10)
"""

from __future__ import annotations

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Bi-encoder dense retrieval: encodes documents into a FAISS index,
    then retrieves nearest neighbours for a query embedding.

    Parameters
    ----------
    documents : list of dict
        Each document must have at least 'text' and 'id' fields.
    model_name : str
        HuggingFace sentence-transformers model identifier.
    batch_size : int
        Encoding batch size (reduce if OOM on large corpora).
    normalize : bool
        If True, L2-normalize embeddings (enables cosine similarity via inner product).
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        import faiss

        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

        logger.info("Encoding %d documents with %s ...", len(documents), model_name)
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        if normalize:
            faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine after normalization
        self.index.add(embeddings)
        logger.info("FAISS index built: %d vectors, dim=%d.", self.index.ntotal, dim)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a query.

        Returns
        -------
        list of dict
            Each item contains original document fields plus 'dense_score' and 'rank'.
        """
        import faiss

        query_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        if self.normalize:
            faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)
        scores, indices = scores[0], indices[0]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0:  # FAISS returns -1 for padded results when ntotal < top_k
                continue
            result = dict(self.documents[idx])
            result["dense_score"] = float(score)
            result["rank"] = rank
            result["retriever"] = "dense"
            results.append(result)

        return results

    def save(self, path: str) -> None:
        """Persist the FAISS index to disk."""
        import faiss
        faiss.write_index(self.index, path)
        logger.info("Saved FAISS index to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved FAISS index."""
        import faiss
        self.index = faiss.read_index(path)
        logger.info("Loaded FAISS index from %s (%d vectors)", path, self.index.ntotal)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs,
    ) -> "DenseRetriever":
        """Convenience constructor from a plain list of text strings."""
        documents = [{"id": str(i), "text": t} for i, t in enumerate(texts)]
        return cls(documents, model_name=model_name, **kwargs)
