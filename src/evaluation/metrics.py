"""
Evaluation metrics used in the MS-RAG ablation study.

Metrics
-------
- ROUGE-L  : longest common subsequence F1 (generation quality)
- MRR      : mean reciprocal rank (retrieval quality)
- P@k      : precision at k (retrieval quality)
- Hallucination Rate : proportion of responses flagged by human annotators
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute the length of the longest common subsequence of token lists."""
    m, n = len(a), len(b)
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L F1 between hypothesis and reference strings.

    Parameters
    ----------
    hypothesis : str
        The generated answer.
    reference : str
        The gold reference answer.

    Returns
    -------
    float
        ROUGE-L F1 in [0, 1].
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """Macro-average ROUGE-L over parallel hypothesis/reference lists."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length.")
    scores = [rouge_l(h, r) for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# MRR — Mean Reciprocal Rank
# ---------------------------------------------------------------------------

def reciprocal_rank(ranked_doc_ids: List[str], relevant_ids: Sequence[str]) -> float:
    """
    Reciprocal rank of the first relevant document in a ranked list.

    Parameters
    ----------
    ranked_doc_ids : list of str
        Document IDs ordered from most to least relevant (rank 1 = first item).
    relevant_ids : sequence of str
        Set of ground-truth relevant document IDs.

    Returns
    -------
    float
        1 / rank of first relevant doc, or 0 if no relevant doc found.
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    all_ranked_lists: List[List[str]],
    all_relevant_ids: List[Sequence[str]],
) -> float:
    """MRR across multiple queries."""
    if len(all_ranked_lists) != len(all_relevant_ids):
        raise ValueError("all_ranked_lists and all_relevant_ids must have the same length.")
    rrs = [
        reciprocal_rank(ranked, relevant)
        for ranked, relevant in zip(all_ranked_lists, all_relevant_ids)
    ]
    return sum(rrs) / len(rrs) if rrs else 0.0


# ---------------------------------------------------------------------------
# P@k — Precision at k
# ---------------------------------------------------------------------------

def precision_at_k(ranked_doc_ids: List[str], relevant_ids: Sequence[str], k: int) -> float:
    """
    Proportion of relevant documents in the top-k retrieved results.

    Parameters
    ----------
    ranked_doc_ids : list of str
    relevant_ids : sequence of str
    k : int

    Returns
    -------
    float in [0, 1]
    """
    relevant_set = set(relevant_ids)
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k if k > 0 else 0.0


def mean_precision_at_k(
    all_ranked_lists: List[List[str]],
    all_relevant_ids: List[Sequence[str]],
    k: int,
) -> float:
    """Mean P@k across multiple queries."""
    scores = [
        precision_at_k(ranked, relevant, k)
        for ranked, relevant in zip(all_ranked_lists, all_relevant_ids)
    ]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Hallucination Rate
# ---------------------------------------------------------------------------

def hallucination_rate(labels: Sequence[int]) -> float:
    """
    Proportion of responses labelled as hallucinated (label == 1).

    Parameters
    ----------
    labels : sequence of int
        Binary labels: 1 = hallucination present, 0 = factually grounded.

    Returns
    -------
    float in [0, 1]
    """
    if not labels:
        return 0.0
    return sum(labels) / len(labels)


def cohens_kappa(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """
    Cohen's kappa inter-annotator agreement for binary labels.

    Parameters
    ----------
    labels_a, labels_b : sequences of int (0 or 1)

    Returns
    -------
    float
        Kappa coefficient in [-1, 1].
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label sequences must have the same length.")
    n = len(labels_a)
    if n == 0:
        return 0.0

    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    p_o = agree / n  # observed agreement

    p_a1 = sum(labels_a) / n
    p_b1 = sum(labels_b) / n
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)  # expected agreement by chance

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)
