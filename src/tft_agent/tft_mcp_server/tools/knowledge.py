"""Evidence retrieval tool; generation and ingestion remain outside this module."""

from typing import Any

from tft_agent.core.schemas import ToolMetadata, ToolResult
from tft_agent.tft_mcp_server.repositories.knowledge_repository import KnowledgeRepository
from tft_agent.tft_mcp_server.tools.game_data import _optional_string, _required_string


class KnowledgeTools:
    """Search semantic evidence without producing a final strategy answer."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self._repository = repository

    async def search_game_knowledge(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        query = _required_string(arguments, "query")
        patch = _optional_string(arguments, "patch")
        limit = arguments.get("limit", 5)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
            raise ValueError("limit must be an integer between 1 and 20")
        chunks = await self._repository.search(query, patch, limit)
        return ToolResult(
            tool_call_id=call_id,
            success=True,
            data={"chunks": [chunk.model_dump() for chunk in chunks]},
            metadata=ToolMetadata(game_patch=patch, source_type="knowledge_repository"),
        )
