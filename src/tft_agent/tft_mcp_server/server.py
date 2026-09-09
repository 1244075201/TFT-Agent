"""Official MCP SDK v2 server composition for the TFT tool process."""

import argparse
from pathlib import Path

from mcp.server import MCPServer

from tft_agent.core.config import Settings
from tft_agent.core.schemas import ToolResult
from tft_agent.embeddings.local_e5 import LocalE5EmbeddingClient
from tft_agent.tft_mcp_server.repositories.game_data_repository import GameDataRepository
from tft_agent.tft_mcp_server.repositories.knowledge_repository import KnowledgeRepository
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.repositories.sqlite_knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from tft_agent.tft_mcp_server.tools.game_data import GameDataTools
from tft_agent.tft_mcp_server.tools.knowledge import KnowledgeTools


def create_server(
    repository: GameDataRepository, knowledge_repository: KnowledgeRepository | None = None
) -> MCPServer:
    """Create a server with dependencies injected and no import-time global state."""

    server = MCPServer(
        "tft-decision-agent",
        instructions="Official structured TFT data tools. Strategic advice is out of scope.",
    )
    tools = GameDataTools(repository)

    @server.tool()
    async def get_champion_info(
        name: str,
        champion_id: str | None = None,
        data_version: str | None = None,
    ) -> ToolResult:
        """Retrieve authoritative champion identity, cost, image metadata, and Riot Data Dragon
        data version. Use only when those fields directly help answer the request. Do not use for
        items, builds, traits, abilities, positioning, strategy, meta strength, set membership,
        base stats, or matchup advice.
        """

        return await tools.lookup_champion(
            call_id="mcp-transport",
            name=name,
            champion_id=champion_id,
            data_version=data_version,
        )

    @server.tool()
    async def get_shop_odds(level: int, game_patch: str | None = None) -> ToolResult:
        """ONLY retrieve base per-shop-slot probabilities for cost tiers 1 through 5 at a player
        level. It does not recommend rolling or leveling and does not calculate an exact champion
        hit chance, pool depletion, opponent holdings, or meta strategy. Omit game_patch unless the
        user or GameState supplies a verified TFT game patch; a Data Dragon data_version is not a
        game_patch.
        """

        return await tools.get_shop_odds(
            call_id="mcp-transport",
            arguments={"level": level, "game_patch": game_patch},
        )

    @server.tool()
    async def search_datatft_comps(
        game_patch: str, query: str | None = None, limit: int = 5
    ) -> ToolResult:
        """Search synchronized DataTFT CN composition statistics for one exact game patch.
        Use for meta strength, composition, item, augment, and level-plan questions. Returned
        numeric values and recommendations must cite the record source URL and retrieval time.
        Never use a mismatched patch or invent a missing statistic.
        """

        return await tools.search_datatft_comps("mcp-transport", query, game_patch, limit)

    if knowledge_repository is not None:
        knowledge = KnowledgeTools(knowledge_repository)
        capability_summary = knowledge_repository.capability_summary()

        @server.tool(
            description=(
                "Search the limited local corpus for official patch/set facts and community "
                "strategy guides. Community strategy is opinion, never Riot authority. Use "
                "strategy only for a named covered comp/topic, not global-best claims or "
                "unsupported authors. Never replace structured champion data or shop odds. "
                "Copy the user's complete question into query; use grounded filters only. For an "
                "explicitly named comp, strategy_comp_name is required. Use strategy_topic only "
                "for an explicit supported topic; omit both for broad strategy questions. "
                f"{capability_summary}"
            )
        )
        async def search_game_knowledge(
            query: str,
            top_k: int = 4,
            game_patch: str | None = None,
            set_name: str | None = None,
            document_type: str | None = None,
            source_type: str | None = None,
            strategy_comp_name: str | None = None,
            strategy_topic: str | None = None,
        ) -> ToolResult:
            """Search the local versioned official and community knowledge corpus."""

            return await knowledge.search_game_knowledge(
                "mcp-transport",
                {
                    "query": query,
                    "top_k": top_k,
                    "game_patch": game_patch,
                    "set_name": set_name,
                    "document_type": document_type,
                    "source_type": source_type,
                    "strategy_comp_name": strategy_comp_name,
                    "strategy_topic": strategy_topic,
                },
            )

    return server


def main() -> None:
    """Initialize SQLite and run the official stdio transport."""

    parser = argparse.ArgumentParser(description="Run the TFT MCP server over stdio")
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    repository = SQLiteGameDataRepository(args.database.resolve())
    repository.initialize_sync()
    settings = Settings()
    embeddings = LocalE5EmbeddingClient(settings.embedding_model, settings.embedding_device)
    knowledge = SQLiteKnowledgeRepository(
        args.database.resolve(), embeddings, settings.retrieval_threshold
    )
    knowledge.initialize_sync()
    create_server(repository, knowledge).run()


if __name__ == "__main__":
    main()
