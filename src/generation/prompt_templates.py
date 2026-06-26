"""
Structured prompt templates for the MS-RAG generation layer.

Templates enforce retrieval grounding: the LLM is instructed to answer
only from the provided context passages and to explicitly state when the
passages do not contain sufficient information.

IMPORTANT: LLM generation results may vary unless the exact model snapshot,
temperature (0.0 for deterministic output), and decoding settings are matched.
See configs/prompt_config.yaml for the settings used in the paper.

API keys must be set as environment variables — never hard-code credentials.
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = (
    "You are a helpful enterprise knowledge assistant. "
    "Answer the user's question using ONLY the information provided in the context passages below. "
    "If the passages do not contain enough information to answer the question, "
    "respond with: 'I don't have enough information in the available knowledge base to answer this question.' "
    "Do not fabricate information or draw on knowledge outside the provided passages."
)

_CONTEXT_BLOCK_TEMPLATE = "Context Passage {rank} (Source: {source}):\n{text}"

_USER_TEMPLATE = (
    "{context_block}\n\n"
    "Question: {query}\n\n"
    "Answer (based only on the passages above):"
)

_PUBLIC_SYSTEM_ADDENDUM = (
    " Note: you are serving a public (unauthenticated) user. "
    "Restrict your answer to information available in public knowledge articles only."
)


TEMPLATES: Dict[str, str] = {
    "authenticated_agent": _SYSTEM_TEMPLATE,
    "public_agent": _SYSTEM_TEMPLATE + _PUBLIC_SYSTEM_ADDENDUM,
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    query: str,
    passages: List[Dict[str, Any]],
    template_key: str = "authenticated_agent",
    text_field: str = "text",
    source_field: str = "source",
    max_passage_chars: int = 1000,
) -> Dict[str, str]:
    """
    Build a system+user prompt pair for the retrieval-grounded generation step.

    Parameters
    ----------
    query : str
        The user's query.
    passages : list of dict
        Top-m retrieved passages after RRF fusion and deduplication.
    template_key : str
        Key into the TEMPLATES dict selecting the system prompt variant.
    text_field : str
        Key in each passage dict holding the passage text.
    source_field : str
        Key in each passage dict holding the source identifier.
    max_passage_chars : int
        Maximum characters to include from each individual passage.

    Returns
    -------
    dict with 'system' and 'user' keys ready for a chat-completion API call.
    """
    system_prompt = TEMPLATES.get(template_key, _SYSTEM_TEMPLATE)

    context_blocks = []
    for rank, passage in enumerate(passages, start=1):
        text = passage.get(text_field, "")[:max_passage_chars]
        source = passage.get(source_field, "unknown")
        context_blocks.append(
            _CONTEXT_BLOCK_TEMPLATE.format(rank=rank, source=source, text=text)
        )

    context_block = "\n\n".join(context_blocks) if context_blocks else "(No passages retrieved)"
    user_prompt = _USER_TEMPLATE.format(context_block=context_block, query=query)

    return {"system": system_prompt, "user": user_prompt}


# ---------------------------------------------------------------------------
# Optional: thin LLM call wrapper (API key from environment only)
# ---------------------------------------------------------------------------

def call_llm(
    prompt: Dict[str, str],
    model: str = "gpt-4o-2024-08-06",
    temperature: float = 0.0,
    max_tokens: int = 512,
    api_key_env: str = "LLM_API_KEY",
) -> Optional[str]:
    """
    Send the prompt to an OpenAI-compatible chat endpoint.

    Requires the LLM_API_KEY environment variable to be set.
    Never pass API keys as function arguments or hard-code them in source.

    Returns
    -------
    str or None
        The model's response text, or None on failure.
    """
    try:
        import openai
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return None

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set. "
            "Export your API key before running generation: "
            f"export {api_key_env}='your-key-here'"
        )

    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return None
