"""
Access-controlled query router for dual-agent CRM orchestration.

The MS-RAG agent layer distinguishes two user classes:
  - Authenticated users  (e.g., registered partner portal users)
  - Public users         (e.g., unauthenticated visitors)

Each class is routed to a different agent configuration that controls
which knowledge sources can be queried and which prompt template is used.

This implementation is intentionally generic and uses no proprietary CRM data.
Replace the placeholder source lists with your own public data-source identifiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class UserClass(str, Enum):
    AUTHENTICATED = "authenticated"
    PUBLIC = "public"


@dataclass
class AgentConfig:
    """Configuration for a single agent role."""

    user_class: UserClass
    allowed_sources: List[str]
    prompt_template_key: str
    max_context_passages: int = 5
    confidence_threshold: float = 0.6


# Default source sets referencing only public dataset identifiers.
# In a real deployment these would map to knowledge article indices,
# CMS collection names, etc. — replace with your own source IDs.
_DEFAULT_AUTHENTICATED_SOURCES = [
    "public_knowledge_articles",
    "public_cms_content",
    "public_pdf_documents",
    "public_cloud_files",
]

_DEFAULT_PUBLIC_SOURCES = [
    "public_knowledge_articles",
    "public_cms_content",
]


class AccessRouter:
    """
    Routes incoming queries to the appropriate agent configuration
    based on user authentication status.

    Parameters
    ----------
    authenticated_config : AgentConfig or None
        Configuration for authenticated portal users. Falls back to a
        sensible default if not provided.
    public_config : AgentConfig or None
        Configuration for unauthenticated public users.
    """

    def __init__(
        self,
        authenticated_config: Optional[AgentConfig] = None,
        public_config: Optional[AgentConfig] = None,
    ) -> None:
        self.configs = {
            UserClass.AUTHENTICATED: authenticated_config or AgentConfig(
                user_class=UserClass.AUTHENTICATED,
                allowed_sources=_DEFAULT_AUTHENTICATED_SOURCES,
                prompt_template_key="authenticated_agent",
                max_context_passages=5,
                confidence_threshold=0.6,
            ),
            UserClass.PUBLIC: public_config or AgentConfig(
                user_class=UserClass.PUBLIC,
                allowed_sources=_DEFAULT_PUBLIC_SOURCES,
                prompt_template_key="public_agent",
                max_context_passages=3,
                confidence_threshold=0.65,
            ),
        }

    def route(self, is_authenticated: bool) -> AgentConfig:
        """
        Return the agent configuration appropriate for this user.

        Parameters
        ----------
        is_authenticated : bool
            True if the user has a valid authenticated session.

        Returns
        -------
        AgentConfig
        """
        user_class = UserClass.AUTHENTICATED if is_authenticated else UserClass.PUBLIC
        config = self.configs[user_class]
        logger.debug("Routing query to %s agent (sources: %s).", user_class, config.allowed_sources)
        return config

    def filter_results(
        self,
        results: List[dict],
        config: AgentConfig,
        source_field: str = "source",
    ) -> List[dict]:
        """
        Filter retrieved passages to only those from allowed sources.

        Parameters
        ----------
        results : list of dict
            Retrieved passages, each containing a `source_field` key.
        config : AgentConfig
            The agent configuration whose allowed_sources list governs filtering.
        source_field : str
            Key in each result dict that identifies the knowledge source.

        Returns
        -------
        list of dict
            Filtered and potentially truncated passage list.
        """
        filtered = [
            r for r in results
            if r.get(source_field, "public_knowledge_articles") in config.allowed_sources
        ]
        return filtered[: config.max_context_passages]
