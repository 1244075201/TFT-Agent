"""Semantic knowledge retrieval contract, separate from ingestion."""

from typing import Protocol

from tft_agent.tft_mcp_server.schemas import KnowledgeChunk


class KnowledgeRepository(Protocol):
    """Retrieve evidence chunks; implementations must not synthesize strategy answers."""

    async def search(self, query: str, patch: str | None, limit: int) -> list[KnowledgeChunk]: ...
