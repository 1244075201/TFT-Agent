"""Fast in-process implementation of the MCPClient contract for the desktop app."""

from __future__ import annotations

from tft_agent.core.config import Settings
from tft_agent.core.schemas import ToolCall, ToolResult
from tft_agent.embeddings.local_e5 import LocalE5EmbeddingClient
from tft_agent.mcp_client.client import MCPToolDefinition
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.repositories.sqlite_knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from tft_agent.tft_mcp_server.schemas import (
    CompStatsRecord,
    DataSnapshotMetadata,
    StaticEntityRecord,
)
from tft_agent.tft_mcp_server.tools.game_data import GameDataTools
from tft_agent.tft_mcp_server.tools.knowledge import KnowledgeTools


class LocalMCPClient:
    """Reuse repositories and the lazy embedding model for the app lifetime."""

    def __init__(self, settings: Settings) -> None:
        database = settings.database_path.resolve()
        self._game_repository = SQLiteGameDataRepository(database)
        self._game_repository.initialize_sync()
        embeddings = LocalE5EmbeddingClient(settings.embedding_model, settings.embedding_device)
        self._knowledge_repository = SQLiteKnowledgeRepository(
            database, embeddings, settings.retrieval_threshold
        )
        self._knowledge_repository.initialize_sync()
        self._game = GameDataTools(self._game_repository)
        self._knowledge = KnowledgeTools(self._knowledge_repository)

    async def list_tools(self) -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="get_champion_info",
                description="Retrieve champion identity, cost, image metadata, and data version.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "champion_id": {"type": ["string", "null"]},
                        "data_version": {"type": ["string", "null"]},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            ),
            MCPToolDefinition(
                name="get_shop_odds",
                description="Retrieve base per-slot shop cost-tier probabilities at one level.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "level": {"type": "integer", "minimum": 1, "maximum": 10},
                        "game_patch": {"type": ["string", "null"]},
                    },
                    "required": ["level"],
                    "additionalProperties": False,
                },
            ),
            MCPToolDefinition(
                name="search_datatft_comps",
                description=(
                    "Search synchronized DataTFT CN composition tiers, units, item identifiers, "
                    "and operation guides for an exact game patch."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "game_patch": {"type": "string"},
                        "query": {"type": ["string", "null"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "required": ["game_patch"],
                    "additionalProperties": False,
                },
            ),
            MCPToolDefinition(
                name="search_game_knowledge",
                description=self._knowledge_repository.capability_summary(),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
                        **{
                            name: {"type": ["string", "null"]}
                            for name in (
                                "game_patch",
                                "set_name",
                                "document_type",
                                "source_type",
                                "strategy_comp_name",
                                "strategy_topic",
                            )
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        ]

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        arguments = tool_call.arguments
        if tool_call.name == "get_champion_info":
            return await self._game.get_champion_info(tool_call.id, arguments)
        if tool_call.name == "get_shop_odds":
            return await self._game.get_shop_odds(tool_call.id, arguments)
        if tool_call.name == "search_datatft_comps":
            patch = arguments.get("game_patch")
            if not isinstance(patch, str):
                raise ValueError("game_patch is required")
            query = arguments.get("query")
            return await self._game.search_datatft_comps(
                tool_call.id,
                query if isinstance(query, str) else None,
                patch,
                int(arguments.get("limit", 5)),
            )
        if tool_call.name == "search_game_knowledge":
            return await self._knowledge.search_game_knowledge(tool_call.id, arguments)
        raise ValueError(f"Unknown local tool: {tool_call.name}")

    async def latest_datatft_snapshot(self) -> tuple[DataSnapshotMetadata, int] | None:
        return await self._game_repository.latest_datatft_snapshot()

    async def search_datatft_comps(
        self, query: str | None, game_patch: str, limit: int = 50
    ) -> list[CompStatsRecord]:
        return await self._game_repository.search_comps(query, game_patch, limit)

    async def search_static_entities(self, names: list[str]) -> list[StaticEntityRecord]:
        return await self._game_repository.search_static_entities(names)

    async def load_session_state(self, session_id: str) -> tuple[str, str] | None:
        return await self._game_repository.load_session_state(session_id)

    async def save_session_state(
        self, session_id: str, state_json: str, field_sources_json: str
    ) -> None:
        await self._game_repository.save_session_state(
            session_id, state_json, field_sources_json
        )

    async def clear_session_state(self, session_id: str) -> None:
        await self._game_repository.clear_session_state(session_id)
