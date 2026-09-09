"""Import a DataTFT snapshot from a public page or user-provided export."""

import argparse
import asyncio
from pathlib import Path

from tft_agent.core.config import Settings
from tft_agent.ingestion.sources.datatft import DataTFTSource
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


async def sync(input_path: Path | None, database: Path, settings: Settings) -> tuple[str, int]:
    source = DataTFTSource(str(settings.datatft_url), settings.datatft_server)
    if input_path is None:
        payload, etag = await asyncio.to_thread(
            source.fetch_current,
            settings.current_game_patch,
            settings.datatft_request_interval_seconds,
        )
        source_url = str(settings.datatft_url)
    else:
        payload, etag = input_path.read_text(encoding="utf-8"), None
        source_url = input_path.resolve().as_uri()
    metadata, records = source.parse(payload, source_url, etag)
    repository = SQLiteGameDataRepository(database.resolve())
    await repository.initialize()
    await repository.replace_datatft_snapshot(metadata, records, payload)
    return metadata.snapshot_id, len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize a versioned DataTFT snapshot")
    parser.add_argument("--input", type=Path, help="User-exported canonical JSON snapshot")
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    settings = Settings()
    snapshot_id, count = asyncio.run(
        sync(args.input, args.database or settings.database_path, settings)
    )
    print(f"snapshot_id={snapshot_id} comps={count}")


if __name__ == "__main__":
    main()
