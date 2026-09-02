"""Opt-in real stdio retrieval smoke for local development."""

import asyncio
import json
import sys
from pathlib import Path

from tft_agent.core.config import Settings
from tft_agent.core.schemas import ToolCall
from tft_agent.mcp_client.stdio_client import StdioMCPClient


async def run() -> None:
    settings = Settings()
    client = StdioMCPClient(
        sys.executable,
        [
            "-m",
            "tft_agent.tft_mcp_server.server",
            "--database",
            str(settings.database_path.resolve()),
        ],
        Path.cwd(),
    )
    async with client:
        tools = [tool.name for tool in await client.list_tools()]
        cases = {
            "official_wisp": {
                "query": "Wisp mechanic",
                "game_patch": "18.1",
                "source_type": "riot_official",
                "top_k": 3,
            },
            "blossom": {
                "query": "Blossom Ahri核心质量不足时D还是升级？",
                "game_patch": "18.1",
                "source_type": "community_strategy",
                "document_type": "strategy_guide",
                "strategy_comp_name": "Blossom Ahri",
                "strategy_topic": "core_quality",
                "top_k": 3,
            },
            "solar": {
                "query": "Solar Kayle现在怎么玩？",
                "game_patch": "18.1",
                "source_type": "community_strategy",
                "document_type": "strategy_guide",
                "strategy_comp_name": "Solar Kayle",
                "top_k": 3,
            },
            "wisp_positioning": {
                "query": "Wisp怎么站位？",
                "game_patch": "18.1",
                "source_type": "community_strategy",
                "document_type": "strategy_guide",
                "strategy_topic": "positioning",
                "top_k": 3,
            },
        }
        results = {
            name: await client.call_tool(
                ToolCall(id=f"rag-smoke-{name}", name="search_game_knowledge", arguments=arguments)
            )
            for name, arguments in cases.items()
        }
    print(
        json.dumps(
            {
                "tools": tools,
                "results": {
                    name: result.model_dump(mode="json") for name, result in results.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
