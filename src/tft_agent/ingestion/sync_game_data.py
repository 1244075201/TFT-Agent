"""Atomic command for synchronizing official structured TFT data."""

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from tft_agent.ingestion.sources.riot_data_dragon import RiotDataDragonSource
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)

logger = logging.getLogger(__name__)


async def sync_game_data(database_path: Path, requested_version: str | None = None) -> str:
    """Fetch, normalize, and atomically replace one Data Dragon version."""

    source = RiotDataDragonSource()
    repository = SQLiteGameDataRepository(database_path)
    await repository.initialize()
    logger.info("Resolving Riot Data Dragon version")
    data_version = await asyncio.to_thread(source.resolve_data_version, requested_version)
    logger.info("Downloading tft-champion.json for data version %s", data_version)
    document = await asyncio.to_thread(source.get_champions, data_version)
    retrieved_at = datetime.now(UTC)
    records = source.normalize_champions(document, retrieved_at)
    if not records:
        raise RuntimeError("Refusing to replace champion data with an empty payload")
    await repository.replace_champions(records, data_version, retrieved_at)
    logger.info("Synchronized %d champion records for %s", len(records), data_version)
    return data_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize official TFT structured data")
    parser.add_argument("--database", type=Path, default=Path("data/tft.sqlite3"))
    parser.add_argument("--data-version")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(sync_game_data(args.database, args.data_version))


if __name__ == "__main__":
    main()
