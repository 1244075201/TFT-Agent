"""Semantic knowledge persistence and retrieval contracts."""

from collections.abc import Sequence
from typing import Protocol

from tft_agent.tft_mcp_server.schemas import KnowledgeChunk, KnowledgeChunkRecord, KnowledgeDocument


class KnowledgeRepository(Protocol):
    async def replace_document(
        self, document: KnowledgeDocument, chunks: Sequence[KnowledgeChunkRecord]
    ) -> None: ...

    async def search(
        self,
        query: str,
        limit: int,
        game_patch: str | None = None,
        set_name: str | None = None,
        document_type: str | None = None,
    ) -> list[KnowledgeChunk]: ...
