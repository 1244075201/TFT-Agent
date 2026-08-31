"""Official MCP SDK v2 server composition for the TFT tool process."""

import argparse
from pathlib import Path

from mcp.server import MCPServer

from tft_agent.core.schemas import ToolResult
from tft_agent.tft_mcp_server.repositories.game_data_repository import GameDataRepository
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.tools.game_data import GameDataTools


def create_server(repository: GameDataRepository) -> MCPServer:
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
        hit chance, pool depletion, opponent holdings, or meta strategy.
        """

        return await tools.get_shop_odds(
            call_id="mcp-transport",
            arguments={"level": level, "game_patch": game_patch},
        )

    return server


def main() -> None:
    """Initialize SQLite and run the official stdio transport."""

    parser = argparse.ArgumentParser(description="Run the TFT MCP server over stdio")
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    repository = SQLiteGameDataRepository(args.database.resolve())
    repository.initialize_sync()
    create_server(repository).run()


if __name__ == "__main__":
    main()
