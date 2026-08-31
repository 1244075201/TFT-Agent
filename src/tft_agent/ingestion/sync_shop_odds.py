"""Offline command for loading reviewed shop odds into SQLite."""

import argparse
import asyncio
import logging
from pathlib import Path

from tft_agent.ingestion.sources.shop_odds import FileShopOddsSource
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)

logger = logging.getLogger(__name__)


async def sync_shop_odds(database_path: Path, snapshot_path: Path) -> int:
    """Validate and atomically replace one patch/source-version table."""

    records = FileShopOddsSource(snapshot_path).load()
    repository = SQLiteGameDataRepository(database_path)
    await repository.initialize()
    await repository.replace_shop_odds(records)
    logger.info(
        "Synchronized %d shop-odds rows for game patch %s from %s",
        len(records),
        records[0].game_patch,
        records[0].source,
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize reviewed TFT shop odds")
    parser.add_argument("--database", type=Path, default=Path("data/tft.sqlite3"))
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data_sources/shop_odds_set18_16.17.json"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(sync_shop_odds(args.database, args.snapshot))


if __name__ == "__main__":
    main()
