"""
Web Search Tool for SyllAIq using Tavily API.
Fallback mechanism when knowledge base does not contain relevant content.
"""

from __future__ import annotations

import os
from typing import List, Optional
from urllib.parse import urlparse

from config.settings import SOURCE_TRUST_TIERS
from models.documents import Document, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_trust_tier(url: str) -> int:
    """Classifies domain into trust tiers (1-5)."""
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        for tier, domains in SOURCE_TRUST_TIERS.items():
            if any(trusted in domain for trusted in domains):
                return tier
    except Exception:
        pass
    return 5


class WebSearchTool:
    """Tavily web search integration returning standardized Document objects."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self._client = None
        if self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
                logger.info("TavilyClient initialized successfully")
            except Exception as exc:
                logger.warning("Failed to initialize TavilyClient: %s", exc)
        else:
            logger.info("TAVILY_API_KEY not configured. Web search will run in mock/disabled fallback.")

    def search(self, query: str, max_results: int = 3) -> List[Document]:
        """
        Executes a search via Tavily and maps results to SyllAIq Document models.
        """
        if not self._client:
            logger.warning("[WebSearchTool] No client available (missing API key or init failure)")
            return []

        try:
            logger.info("[WebSearchTool] Searching Tavily for query: %r (max=%d)", query[:80], max_results)
            response = self._client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_raw_content=False,
            )

            results = response.get("results", [])
            documents: List[Document] = []

            for idx, res in enumerate(results):
                title = res.get("title", "")
                content = res.get("content", "")
                url = res.get("url", "")
                score = float(res.get("score", 0.5))

                doc_text = f"Title: {title}\nURL: {url}\n\nContent: {content}"
                doc = Document(
                    chunk_id=f"web_{idx}_{abs(hash(url)) % 100000}",
                    text=doc_text,
                    source_type=SourceType.WEB,
                    subject="Operating Systems",
                    university="Web",
                    topic=title or "Web Search Result",
                    book=url,  # store url in book or topic for citation mapping
                    char_count=len(doc_text),
                    reranker_score=score,
                    nli_score=score,
                )
                documents.append(doc)

            logger.info("[WebSearchTool] Found %d results from Tavily", len(documents))
            return documents

        except Exception as exc:
            logger.error("[WebSearchTool] Error searching Tavily: %s", exc)
            return []
