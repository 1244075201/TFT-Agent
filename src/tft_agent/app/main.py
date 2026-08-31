"""CLI composition placeholder; business logic belongs in lower layers."""

import logging

from tft_agent.core.config import Settings


def main() -> None:
    """Configure the process and explain unavailable concrete adapters."""

    settings = Settings()
    logging.basicConfig(level=settings.log_level)
    raise SystemExit(
        "No concrete LLM or MCP transport is configured yet. "
        "Use the typed interfaces to compose adapters in the next phase."
    )


if __name__ == "__main__":
    main()
