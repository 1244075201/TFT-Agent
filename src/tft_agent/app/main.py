"""One-shot CLI composition root; business logic belongs in lower layers."""

import asyncio
import logging

from tft_agent.agent.response_formatter import ResponseFormatter
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatRequest


def main() -> None:
    """Run one natural-language task through DeepSeek and local MCP."""

    settings = Settings()
    logging.basicConfig(level=settings.log_level)
    asyncio.run(_run_once(settings))


async def _run_once(settings: Settings) -> None:
    question = input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required")
    service = AgentService(settings)
    async for event in service.ask(ChatRequest(text=question, session_id="cli")):
        if event.type == "answer" and event.response is not None:
            print(ResponseFormatter().format_text(event.response))
        elif event.type == "error":
            raise SystemExit(event.message)


if __name__ == "__main__":
    main()
