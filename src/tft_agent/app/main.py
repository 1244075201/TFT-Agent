"""One-shot CLI composition root; business logic belongs in lower layers."""

import asyncio
import logging
import sys
from pathlib import Path

from tft_agent.agent.response_formatter import ResponseFormatter
from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.config import Settings
from tft_agent.core.schemas import AgentStatus, GameState
from tft_agent.llm.deepseek_client import DeepSeekLLMClient
from tft_agent.mcp_client.stdio_client import StdioMCPClient


def main() -> None:
    """Run one natural-language task through DeepSeek and local MCP."""

    settings = Settings()
    logging.basicConfig(level=settings.log_level)
    asyncio.run(_run_once(settings))


async def _run_once(settings: Settings) -> None:
    question = input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required")
    database_path = settings.database_path.resolve()
    if not database_path.is_file():
        raise SystemExit(
            f"Structured data database does not exist: {database_path}. "
            "Run tft-sync-game-data first."
        )
    llm_client = DeepSeekLLMClient.from_settings(settings)
    server_args = [
        "-m",
        "tft_agent.tft_mcp_server.server",
        "--database",
        str(database_path),
    ]
    state = AgentState(
        game_state=GameState(user_question=question),
        goal=question,
        max_steps=settings.max_steps,
    )
    try:
        async with StdioMCPClient(
            command=sys.executable,
            args=server_args,
            cwd=Path.cwd(),
        ) as mcp_client:
            result = await AgentRuntime(llm_client, mcp_client).run(state)
    finally:
        await llm_client.close()
    if result.status == AgentStatus.COMPLETED and result.final_response is not None:
        print(ResponseFormatter().format_text(result.final_response))
        return
    if result.status == AgentStatus.WAITING_FOR_CONTEXT:
        print("More context is required: " + ", ".join(result.missing_context))
        return
    raise SystemExit(
        f"Agent stopped with status={result.status.value}: {result.failure_reason or ''}"
    )


if __name__ == "__main__":
    main()
