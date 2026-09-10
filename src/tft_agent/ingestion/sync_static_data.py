"""Synchronize Chinese TFT static definitions from CommunityDragon."""

import asyncio

from tft_agent.app.entity_resolution import explicit_aliases
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
        current_set_number=settings.current_set_number,
    )
    records = await asyncio.to_thread(source.fetch)
    repository = SQLiteGameDataRepository(settings.database_path.resolve())
    await repository.initialize()
    await repository.replace_static_entities(records)
    await repository.rebuild_canonical_catalog(
        settings.current_game_patch,
        settings.current_set_number,
        settings.current_set_name,
        explicit_aliases(),
        active_static_ids=source.active_entity_ids,
    )
    return len(records)


def main() -> None:
    print(f"已同步 {asyncio.run(sync())} 条中文静态定义")


if __name__ == "__main__":
    main()
