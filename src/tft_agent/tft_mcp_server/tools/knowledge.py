"""Evidence retrieval tool; generation and ingestion remain outside this module."""

from typing import Any

from tft_agent.core.schemas import ToolError, ToolMetadata, ToolResult
from tft_agent.tft_mcp_server.repositories.knowledge_repository import KnowledgeRepository
from tft_agent.tft_mcp_server.tools.game_data import _optional_string, _required_string


class KnowledgeTools:
    """Search semantic evidence without producing a final strategy answer."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self._repository = repository

    async def search_game_knowledge(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        query = _required_string(arguments, "query")
        patch = _optional_string(arguments, "game_patch")
        set_name = _optional_string(arguments, "set_name")
        document_type = _optional_string(arguments, "document_type")
        source_type = _optional_string(arguments, "source_type")
        strategy_comp_name = _optional_string(arguments, "strategy_comp_name")
        strategy_topic = _optional_string(arguments, "strategy_topic")
        limit = arguments.get("top_k", 4)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 8:
            raise ValueError("top_k must be an integer between 1 and 8")
        chunks = await self._repository.search(
            query,
            limit,
            patch,
            set_name,
            document_type,
            source_type,
            strategy_comp_name,
            strategy_topic,
        )
        if not chunks:
            return ToolResult(
                tool_call_id=call_id,
                success=False,
                metadata=ToolMetadata(game_patch=patch, source_type="knowledge_repository"),
                error=ToolError(
                    code="knowledge_not_found",
                    message=(
                        "The local versioned knowledge does not establish an answer; "
                        "do not substitute model memory."
                    ),
                ),
            )
        return ToolResult(
            tool_call_id=call_id,
            success=True,
            data={"results": [chunk.model_dump(mode="json") for chunk in chunks]},
            metadata=ToolMetadata(game_patch=patch, source_type="knowledge_repository"),
        )
