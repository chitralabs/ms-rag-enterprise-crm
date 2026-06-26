"""
Confidence-based escalation logic for the MS-RAG agent layer.

When the agent's response confidence falls below a configurable threshold,
the query is escalated to a human support queue rather than returning a
potentially unreliable generated answer.

Confidence estimation is heuristic: it combines the top retrieved passage
score with a simple token-overlap measure between the query and the
highest-ranked passage. In production, this can be replaced by a
calibrated classifier or an LLM self-assessment step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EscalationDecision:
    """Result of a confidence check."""

    should_escalate: bool
    confidence: float
    reason: str
    top_passage: Optional[Dict[str, Any]] = None


def estimate_confidence(
    query: str,
    passages: List[Dict[str, Any]],
    score_field: str = "rrf_score",
    score_weight: float = 0.7,
    overlap_weight: float = 0.3,
) -> float:
    """
    Estimate response confidence from retrieval scores and query–passage overlap.

    Parameters
    ----------
    query : str
        The user query.
    passages : list of dict
        Retrieved and fused passages (must contain `score_field`).
    score_field : str
        Name of the numerical relevance score in each passage dict.
    score_weight : float
        Weight assigned to the normalized top-passage retrieval score.
    overlap_weight : float
        Weight assigned to the query–passage token overlap ratio.

    Returns
    -------
    float
        Estimated confidence in [0, 1].
    """
    if not passages:
        return 0.0

    top = passages[0]
    raw_score = float(top.get(score_field, 0.0))
    # Normalize RRF scores (which are in roughly [0, 1] already) via sigmoid-like clamping
    normalized_score = min(raw_score * 10.0, 1.0)  # RRF scores are small; scale up

    query_tokens = set(query.lower().split())
    passage_tokens = set(top.get("text", "").lower().split())
    overlap = (
        len(query_tokens & passage_tokens) / len(query_tokens)
        if query_tokens
        else 0.0
    )

    confidence = score_weight * normalized_score + overlap_weight * overlap
    return round(min(confidence, 1.0), 4)


class ConfidenceEscalator:
    """
    Decides whether to return a generated answer or escalate to human support.

    Parameters
    ----------
    threshold : float
        Confidence values below this trigger escalation (default 0.6).
    empty_passage_escalates : bool
        If True, escalate immediately when the retrieval result set is empty.
    """

    def __init__(self, threshold: float = 0.6, empty_passage_escalates: bool = True) -> None:
        self.threshold = threshold
        self.empty_passage_escalates = empty_passage_escalates

    def check(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        score_field: str = "rrf_score",
    ) -> EscalationDecision:
        """
        Evaluate confidence and return an escalation decision.

        Parameters
        ----------
        query : str
            The user query.
        passages : list of dict
            Top-m retrieved passages after deduplication.
        score_field : str
            Key used to read retrieval scores.

        Returns
        -------
        EscalationDecision
        """
        if self.empty_passage_escalates and not passages:
            return EscalationDecision(
                should_escalate=True,
                confidence=0.0,
                reason="No relevant passages retrieved; escalating to human support.",
            )

        confidence = estimate_confidence(query, passages, score_field=score_field)
        should_escalate = confidence < self.threshold

        reason = (
            f"Confidence {confidence:.3f} below threshold {self.threshold}; escalating."
            if should_escalate
            else f"Confidence {confidence:.3f} meets threshold {self.threshold}; proceeding."
        )

        if should_escalate:
            logger.info("Escalation triggered: %s", reason)
        else:
            logger.debug("No escalation: %s", reason)

        return EscalationDecision(
            should_escalate=should_escalate,
            confidence=confidence,
            reason=reason,
            top_passage=passages[0] if passages else None,
        )
