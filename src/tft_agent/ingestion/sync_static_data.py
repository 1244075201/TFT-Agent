"""Synchronize Chinese TFT static definitions from CommunityDragon."""

import asyncio

from tft_agent.core.config import Settings
from tft_agent.ingestion.sources.communitydragon import CommunityDragonSource
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


async def sync() -> int:
    settings = Settings()
    source = CommunityDragonSource(
        str(settings.communitydragon_base_url),
        settings.static_data_version,
        str(settings.communitydragon_tft_url),
    )
    records = await asyncio.to_thread(source.fetch)
    repository = SQLiteGameDataRepository(settings.database_path.resolve())
    await repository.initialize()
    await repository.replace_static_entities(records)
    return len(records)


def main() -> None:
    print(f"已同步 {asyncio.run(sync())} 条中文静态定义")


if __name__ == "__main__":
    main()
