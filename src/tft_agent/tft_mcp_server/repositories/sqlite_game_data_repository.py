"""SQLite implementation of structured TFT game-data persistence."""

import asyncio
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from tft_agent.tft_mcp_server.repositories.migrations import apply_migrations
from tft_agent.tft_mcp_server.schemas import (
    ChampionImage,
    ChampionRecord,
    ItemInfo,
    ShopOddsRecord,
    TraitInfo,
)


class SQLiteGameDataRepository:
    """Persist and query normalized records without exposing SQL to callers."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def initialize_sync(self) -> None:
        """Initialize schema before starting a synchronous process entry point."""

        self._initialize_sync()

    async def replace_champions(
        self,
        records: Sequence[ChampionRecord],
        data_version: str,
        retrieved_at: datetime,
    ) -> None:
        await asyncio.to_thread(
            self._replace_champions_sync, list(records), data_version, retrieved_at
        )

    async def find_champions(
        self,
        name: str,
        champion_id: str | None = None,
        data_version: str | None = None,
    ) -> list[ChampionRecord]:
        return await asyncio.to_thread(self._find_champions_sync, name, champion_id, data_version)

    async def get_item(self, name: str, patch: str | None) -> ItemInfo | None:
        return None

    async def get_trait(self, name: str, patch: str | None) -> TraitInfo | None:
        return None

    async def replace_shop_odds(self, records: Sequence[ShopOddsRecord]) -> None:
        await asyncio.to_thread(self._replace_shop_odds_sync, list(records))

    async def get_shop_odds(self, level: int, game_patch: str | None) -> ShopOddsRecord | None:
        return await asyncio.to_thread(self._get_shop_odds_sync, level, game_patch)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_sync(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            apply_migrations(connection)

    def _replace_champions_sync(
        self, records: list[ChampionRecord], data_version: str, retrieved_at: datetime
    ) -> None:
        if any(record.data_version != data_version for record in records):
            raise ValueError("all records must match the synchronized data_version")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM champion_syncs WHERE data_version = ?", (data_version,))
            source = records[0].source if records else "riot-data-dragon"
            connection.execute(
                "INSERT INTO champion_syncs VALUES (?, ?, ?, ?)",
                (data_version, source, retrieved_at.isoformat(), len(records)),
            )
            connection.executemany(
                "INSERT INTO champions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        record.data_version,
                        record.champion_id,
                        record.name,
                        record.cost,
                        record.image.full,
                        record.image.sprite,
                        record.image.group,
                        record.image.x,
                        record.image.y,
                        record.image.width,
                        record.image.height,
                        record.source,
                        record.retrieved_at.isoformat(),
                    )
                    for record in records
                ],
            )

    def _find_champions_sync(
        self, name: str, champion_id: str | None, data_version: str | None
    ) -> list[ChampionRecord]:
        clauses = ["name = ? COLLATE NOCASE"]
        parameters: list[object] = [name]
        if champion_id is not None:
            clauses.append("champion_id = ?")
            parameters.append(champion_id)
        if data_version is not None:
            clauses.append("data_version = ?")
            parameters.append(data_version)
        else:
            clauses.append(
                "data_version = (SELECT data_version FROM champion_syncs "
                "ORDER BY retrieved_at DESC LIMIT 1)"
            )
        query = "SELECT * FROM champions WHERE " + " AND ".join(clauses)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _replace_shop_odds_sync(self, records: list[ShopOddsRecord]) -> None:
        if not records:
            raise ValueError("shop odds synchronization requires records")
        patch = records[0].game_patch
        source_version = records[0].source_version
        if any(
            record.game_patch != patch or record.source_version != source_version
            for record in records
        ):
            raise ValueError("shop odds records must share patch and source_version")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM shop_odds WHERE game_patch = ? AND source_version = ?",
                (patch, source_version),
            )
            connection.executemany(
                "INSERT INTO shop_odds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        record.game_patch,
                        record.source_version,
                        record.level,
                        record.cost_1_probability,
                        record.cost_2_probability,
                        record.cost_3_probability,
                        record.cost_4_probability,
                        record.cost_5_probability,
                        record.source,
                        record.source_type,
                        record.retrieved_at.isoformat(),
                    )
                    for record in records
                ],
            )

    def _get_shop_odds_sync(self, level: int, game_patch: str | None) -> ShopOddsRecord | None:
        clauses = ["level = ?"]
        parameters: list[object] = [level]
        if game_patch is not None:
            clauses.append("game_patch = ?")
            parameters.append(game_patch)
        query = "SELECT * FROM shop_odds WHERE " + " AND ".join(clauses)
        query += " ORDER BY retrieved_at DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return ShopOddsRecord.model_validate(dict(row)) if row is not None else None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ChampionRecord:
        return ChampionRecord(
            champion_id=row["champion_id"],
            name=row["name"],
            cost=row["cost"],
            image=ChampionImage(
                full=row["image_full"],
                sprite=row["image_sprite"],
                group=row["image_group"],
                x=row["image_x"],
                y=row["image_y"],
                w=row["image_width"],
                h=row["image_height"],
            ),
            data_version=row["data_version"],
            source=row["source"],
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
        )
