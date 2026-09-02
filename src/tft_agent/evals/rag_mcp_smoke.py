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
        result = await client.call_tool(
            ToolCall(
                id="rag-smoke",
                name="search_game_knowledge",
                arguments={
                    "query": "Wisp mechanic",
                    "game_patch": "18.1",
                    "source_type": "riot_official",
                    "top_k": 3,
                },
                # Explicit authority filter prevents community strategy chunks from
                # competing with an official mechanics question.
            )
        )
    print(
        json.dumps(
            {"tools": tools, "result": result.model_dump(mode="json")},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
