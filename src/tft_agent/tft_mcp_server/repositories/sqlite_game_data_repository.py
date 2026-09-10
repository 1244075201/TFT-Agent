"""SQLite implementation of structured TFT game-data persistence."""

import asyncio
import json
import re
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from tft_agent.tft_mcp_server.repositories.migrations import apply_migrations
from tft_agent.tft_mcp_server.schemas import (
    ChampionImage,
    ChampionRecord,
    CompStatsRecord,
    DataSnapshotMetadata,
    ItemInfo,
    ShopOddsRecord,
    StaticEntityRecord,
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

    async def replace_datatft_snapshot(
        self, metadata: DataSnapshotMetadata, records: Sequence[CompStatsRecord], raw_payload: str
    ) -> None:
        await asyncio.to_thread(
            self._replace_datatft_snapshot_sync, metadata, list(records), raw_payload
        )

    async def search_comps(
        self, query: str | None, game_patch: str, limit: int = 5
    ) -> list[CompStatsRecord]:
        return await asyncio.to_thread(self._search_comps_sync, query, game_patch, limit)

    async def latest_datatft_snapshot(self) -> tuple[DataSnapshotMetadata, int] | None:
        return await asyncio.to_thread(self._latest_datatft_snapshot_sync)

    async def replace_static_entities(self, records: Sequence[StaticEntityRecord]) -> None:
        await asyncio.to_thread(self._replace_static_entities_sync, list(records))

    async def search_static_entities(
        self, names: Sequence[str], data_version: str | None = None
    ) -> list[StaticEntityRecord]:
        return await asyncio.to_thread(
            self._search_static_entities_sync, list(names), data_version
        )

    async def load_session_state(self, session_id: str) -> tuple[str, str] | None:
        return await asyncio.to_thread(self._load_session_state_sync, session_id)

    async def save_session_state(
        self, session_id: str, state_json: str, field_sources_json: str
    ) -> None:
        await asyncio.to_thread(
            self._save_session_state_sync, session_id, state_json, field_sources_json
        )

    async def clear_session_state(self, session_id: str) -> None:
        await asyncio.to_thread(self._clear_session_state_sync, session_id)

    async def load_session_messages(
        self, session_id: str, limit: int = 12
    ) -> list[tuple[str, str]]:
        return await asyncio.to_thread(self._load_session_messages_sync, session_id, limit)

    async def append_session_messages(
        self, session_id: str, messages: Sequence[tuple[str, str]]
    ) -> None:
        await asyncio.to_thread(self._append_session_messages_sync, session_id, list(messages))

    async def clear_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._clear_session_sync, session_id)

    async def search_knowledge_text(
        self,
        query: str,
        game_patch: str | None,
        limit: int = 6,
        strategy_comp_name: str | None = None,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self._search_knowledge_text_sync,
            query,
            game_patch,
            limit,
            strategy_comp_name,
        )

    async def rebuild_canonical_catalog(
        self,
        game_patch: str,
        set_number: int,
        set_name: str,
        explicit_aliases: Sequence[dict[str, str]] = (),
        active_static_ids: Sequence[str] = (),
    ) -> None:
        await asyncio.to_thread(
            self._rebuild_canonical_catalog_sync,
            game_patch,
            set_number,
            set_name,
            list(explicit_aliases),
            list(active_static_ids),
        )

    async def canonical_catalog(
        self, game_patch: str, set_number: int
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._canonical_catalog_sync, game_patch, set_number)

    async def find_static_entities_by_ids(
        self, canonical_ids: Sequence[str]
    ) -> list[StaticEntityRecord]:
        return await asyncio.to_thread(self._find_static_entities_by_ids_sync, list(canonical_ids))

    async def find_comp_by_id(
        self, canonical_id: str, game_patch: str
    ) -> CompStatsRecord | None:
        records = await asyncio.to_thread(self._find_comp_by_id_sync, canonical_id, game_patch)
        return records

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

    def _replace_datatft_snapshot_sync(
        self, metadata: DataSnapshotMetadata, records: list[CompStatsRecord], raw_payload: str
    ) -> None:
        if any(record.snapshot.snapshot_id != metadata.snapshot_id for record in records):
            raise ValueError("all comp records must reference the synchronized snapshot")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE datatft_snapshots SET status = 'stale' "
                "WHERE server = ? AND snapshot_id <> ?",
                (metadata.server, metadata.snapshot_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO datatft_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*metadata.model_dump(mode="json").values(), raw_payload),
            )
            connection.executemany(
                "INSERT OR REPLACE INTO datatft_comps "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        metadata.snapshot_id, record.comp_id, record.name, record.tier,
                        record.average_place, record.play_rate, record.top4_rate, record.win_rate,
                        json.dumps(record.core_units, ensure_ascii=False),
                        json.dumps(record.recommended_items, ensure_ascii=False),
                        json.dumps(record.augments, ensure_ascii=False),
                        json.dumps(record.level_plan, ensure_ascii=False),
                    )
                    for record in records
                ],
            )

    def _search_comps_sync(
        self, query: str | None, game_patch: str, limit: int
    ) -> list[CompStatsRecord]:
        parameters: list[object] = [game_patch]
        name_clause = ""
        if query:
            name_clause = " AND c.name LIKE ?"
            parameters.append(f"%{query}%")
        parameters.append(limit)
        sql = (
            "SELECT c.*, s.* FROM datatft_comps c JOIN datatft_snapshots s USING(snapshot_id) "
            "WHERE s.game_patch = ? AND s.status IN ('valid','stale')" + name_clause
            + " ORDER BY s.retrieved_at DESC, CASE c.tier "
            "WHEN 'S' THEN 0 WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 ELSE 4 END, "
            "c.average_place ASC, c.name LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        output: list[CompStatsRecord] = []
        for row in rows:
            snapshot = DataSnapshotMetadata.model_validate(
                {key: row[key] for key in DataSnapshotMetadata.model_fields}
            )
            output.append(
                CompStatsRecord(
                    comp_id=row["comp_id"], name=row["name"], tier=row["tier"],
                    average_place=row["average_place"], play_rate=row["play_rate"],
                    top4_rate=row["top4_rate"], win_rate=row["win_rate"],
                    core_units=json.loads(row["core_units"]),
                    recommended_items=json.loads(row["recommended_items"]),
                    augments=json.loads(row["augments"]), level_plan=json.loads(row["level_plan"]),
                    snapshot=snapshot,
                )
            )
        return output

    def _latest_datatft_snapshot_sync(self) -> tuple[DataSnapshotMetadata, int] | None:
        sql = (
            "SELECT s.*, COUNT(c.comp_id) AS comp_count FROM datatft_snapshots s "
            "LEFT JOIN datatft_comps c USING(snapshot_id) WHERE s.status = 'valid' "
            "GROUP BY s.snapshot_id ORDER BY s.retrieved_at DESC LIMIT 1"
        )
        with self._connect() as connection:
            row = connection.execute(sql).fetchone()
        if row is None:
            return None
        metadata = DataSnapshotMetadata.model_validate(
            {key: row[key] for key in DataSnapshotMetadata.model_fields}
        )
        return metadata, int(row["comp_count"])

    def _replace_static_entities_sync(self, records: list[StaticEntityRecord]) -> None:
        if not records:
            raise ValueError("static entity synchronization requires records")
        versions = {record.data_version for record in records}
        if len(versions) != 1:
            raise ValueError("static entities must share one data version")
        version = records[0].data_version
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM static_entities WHERE data_version = ?", (version,))
            connection.executemany(
                "INSERT INTO static_entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        record.data_version, record.entity_id, record.entity_type, record.name,
                        record.description, json.dumps(record.traits, ensure_ascii=False),
                        json.dumps(record.components, ensure_ascii=False),
                        json.dumps(record.tags, ensure_ascii=False), record.source,
                        record.source_url, record.retrieved_at.isoformat(),
                    )
                    for record in records
                ],
            )

    def _search_static_entities_sync(
        self, names: list[str], data_version: str | None
    ) -> list[StaticEntityRecord]:
        cleaned = list(dict.fromkeys(name.strip() for name in names if name.strip()))
        if not cleaned:
            return []
        clauses: list[str] = []
        parameters: list[object] = []
        for name in cleaned:
            clauses.append("name = ? COLLATE NOCASE")
            parameters.append(name)
            if name.endswith("转"):
                clauses.append("(entity_type = 'emblem' AND name LIKE ?)")
                parameters.append(f"%{name[:-1]}%纹章%")
            if name == "铲子":
                clauses.append("name = '金铲铲'")
        sql = "SELECT * FROM static_entities WHERE (" + " OR ".join(clauses) + ")"
        if data_version is not None:
            sql += " AND data_version = ?"
            parameters.append(data_version)
        else:
            sql += " ORDER BY retrieved_at DESC"
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [
            StaticEntityRecord(
                entity_id=row["entity_id"], entity_type=row["entity_type"], name=row["name"],
                description=row["description"], traits=json.loads(row["traits"]),
                components=json.loads(row["components"]), tags=json.loads(row["tags"]),
                data_version=row["data_version"], source=row["source"],
                source_url=row["source_url"], retrieved_at=row["retrieved_at"],
            )
            for row in rows
        ]

    def _load_session_state_sync(self, session_id: str) -> tuple[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json, field_sources_json FROM session_game_states WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return (str(row[0]), str(row[1])) if row is not None else None

    def _save_session_state_sync(
        self, session_id: str, state_json: str, field_sources_json: str
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO session_game_states VALUES "
                "(?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now')) "
                "ON CONFLICT(session_id) DO UPDATE SET state_json=excluded.state_json, "
                "field_sources_json=excluded.field_sources_json, updated_at=excluded.updated_at",
                (session_id, state_json, field_sources_json),
            )

    def _clear_session_state_sync(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM session_game_states WHERE session_id=?", (session_id,))

    def _load_session_messages_sync(
        self, session_id: str, limit: int
    ) -> list[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM session_messages WHERE session_id=? "
                "ORDER BY message_id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [(str(row["role"]), str(row["content"])) for row in reversed(rows)]

    def _append_session_messages_sync(
        self, session_id: str, messages: list[tuple[str, str]]
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO session_messages(session_id, role, content, created_at) "
                "VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                [(session_id, role, content) for role, content in messages],
            )

    def _clear_session_sync(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM session_messages WHERE session_id=?", (session_id,))
            connection.execute("DELETE FROM session_game_states WHERE session_id=?", (session_id,))

    def _search_knowledge_text_sync(
        self,
        query: str,
        game_patch: str | None,
        limit: int,
        strategy_comp_name: str | None,
    ) -> list[dict[str, object]]:
        tokens: list[str] = []
        for token in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]{3,}", query):
            if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 2:
                tokens.extend(token[index:index + 2] for index in range(len(token) - 1))
            else:
                tokens.append(token)
        tokens = list(dict.fromkeys(tokens))
        rows = self._search_knowledge_rows(
            query, tokens, game_patch, limit, strategy_comp_name, "exact"
        )
        base_patch = re.sub(r"[a-z]+$", "", game_patch or "", flags=re.I)
        if not rows and game_patch and base_patch != game_patch:
            rows = self._search_knowledge_rows(
                query, tokens, base_patch, limit, strategy_comp_name, "family_fallback"
            )
        if not rows:
            rows = self._search_knowledge_rows(
                query, tokens, None, limit, strategy_comp_name, "unversioned"
            )
        return rows

    def _search_knowledge_rows(
        self,
        query: str,
        tokens: list[str],
        source_patch: str | None,
        limit: int,
        strategy_comp_name: str | None,
        patch_match: str,
    ) -> list[dict[str, object]]:
        clauses = ["d.game_patch IS NULL"] if source_patch is None else ["d.game_patch = ?"]
        params: list[object] = [] if source_patch is None else [source_patch]
        if strategy_comp_name:
            clauses.append("d.strategy_comp_names LIKE ?")
            params.append(f"%{strategy_comp_name}%")
        if tokens:
            token_clauses = " OR ".join(
                "(c.text LIKE ? OR d.title LIKE ?)" for _ in tokens
            )
            clauses.append("(" + token_clauses + ")")
            for token in tokens:
                params.extend((f"%{token}%", f"%{token}%"))
        params.append(max(limit * 5, limit))
        sql = (
            "SELECT c.chunk_id, c.heading, c.text, d.title, d.source, d.source_type, "
            "d.document_type, d.game_patch, d.url, d.retrieved_at "
            "FROM knowledge_chunks c JOIN knowledge_documents d USING(document_id) WHERE "
            + " AND ".join(clauses)
            + " ORDER BY d.retrieved_at DESC, c.chunk_index LIMIT ?"
        )
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(sql, params).fetchall()]
        ranked: list[dict[str, object]] = []
        for row in rows:
            haystack = f"{row['title']} {row['heading']} {row['text']}".casefold()
            score = sum(token.casefold() in haystack for token in tokens) / max(len(tokens), 1)
            if score <= 0 and tokens:
                continue
            row["relevance_score"] = round(score, 4)
            row["patch_match"] = patch_match
            ranked.append(row)
        return sorted(
            ranked, key=lambda item: float(str(item["relevance_score"])), reverse=True
        )[:limit]

    def _rebuild_canonical_catalog_sync(
        self,
        game_patch: str,
        set_number: int,
        set_name: str,
        explicit_aliases: list[dict[str, str]],
        active_static_ids: list[str],
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM entity_aliases WHERE game_patch=? AND set_number=?",
                (game_patch, set_number),
            )
            connection.execute(
                "DELETE FROM canonical_entities WHERE game_patch=? AND set_number=?",
                (game_patch, set_number),
            )
            comps = connection.execute(
                "SELECT DISTINCT c.comp_id, c.name FROM datatft_comps c "
                "JOIN datatft_snapshots s USING(snapshot_id) "
                "WHERE s.game_patch=? AND s.status IN ('valid','stale')",
                (game_patch,),
            ).fetchall()
            static_sql = "SELECT entity_id, entity_type, name FROM static_entities"
            static_params: list[object] = []
            if active_static_ids:
                marks = ",".join("?" for _ in active_static_ids)
                static_sql += f" WHERE entity_id IN ({marks})"
                static_params.extend(active_static_ids)
            else:
                # Compatibility bootstrap for databases synchronized before active-set metadata.
                from tft_agent.app.aliases import AliasRegistry

                registry_ids = sorted({
                    entry.canonical_id
                    for entry in AliasRegistry.load_default().entries
                    if entry.canonical_id is not None
                    and (entry.scope == "global" or entry.set_number == set_number)
                })
                marks = ",".join("?" for _ in registry_ids)
                static_sql += (
                    " WHERE entity_id LIKE ? OR entity_id LIKE 'DA_Component_%'"
                    + (f" OR entity_id IN ({marks})" if marks else "")
                )
                static_params.append(f"DA%{set_number}%")
                static_params.extend(registry_ids)
            static = connection.execute(static_sql, static_params).fetchall()
            entities = [
                (str(row["comp_id"]), "composition", str(row["name"]), "datatft")
                for row in comps
            ] + [
                (
                    str(row["entity_id"]), str(row["entity_type"]),
                    str(row["name"]), "communitydragon",
                )
                for row in static
                if row["name"] and "_Name" not in str(row["name"])
            ]
            connection.executemany(
                "INSERT INTO canonical_entities VALUES (?,?,?,?,?,?,?)",
                [(identifier, kind, name, game_patch, set_number, set_name, source)
                 for identifier, kind, name, source in entities],
            )
            from tft_agent.app.aliases import AliasRegistry
            registry = AliasRegistry.load_default()
            materialized = registry.materialize(entities, set_number, game_patch)
            aliases = [(item.alias, item.canonical_id, item.entity_type, item.alias_kind)
                       for item in materialized]
            occupied = {item.alias.casefold() for item in materialized}
            for identifier, kind, name, _source in entities:
                if kind == "emblem" and name.endswith("纹章"):
                    alias = name.removesuffix("纹章") + "转"
                    if alias.casefold() not in occupied:
                        aliases.append((alias, identifier, kind, "derived_emblem"))
            connection.executemany(
                "INSERT OR IGNORE INTO entity_aliases VALUES (?,?,?,?,?,?)",
                [(alias, identifier, kind, game_patch, set_number, alias_kind)
                 for alias, identifier, kind, alias_kind in aliases],
            )

    def _canonical_catalog_sync(
        self, game_patch: str, set_number: int
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT e.*, a.alias, a.alias_kind FROM canonical_entities e "
                "LEFT JOIN entity_aliases a ON a.canonical_id=e.canonical_id "
                "AND a.game_patch=e.game_patch AND a.set_number=e.set_number "
                "WHERE e.game_patch=? AND e.set_number=? ORDER BY e.entity_type,e.display_name",
                (game_patch, set_number),
            ).fetchall()
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = grouped.setdefault(str(row["canonical_id"]), {
                "canonical_id": row["canonical_id"], "entity_type": row["entity_type"],
                "display_name": row["display_name"], "game_patch": row["game_patch"],
                "set_number": row["set_number"], "set_name": row["set_name"],
                "source": row["source"], "aliases": [],
            })
            if row["alias"]:
                item["aliases"].append(row["alias"])
        return list(grouped.values())

    def _find_static_entities_by_ids_sync(
        self, canonical_ids: list[str]
    ) -> list[StaticEntityRecord]:
        if not canonical_ids:
            return []
        marks = ",".join("?" for _ in canonical_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM static_entities WHERE entity_id IN ({marks})", canonical_ids
            ).fetchall()
        return [StaticEntityRecord(
            entity_id=row["entity_id"], entity_type=row["entity_type"], name=row["name"],
            description=row["description"], traits=json.loads(row["traits"]),
            components=json.loads(row["components"]), tags=json.loads(row["tags"]),
            data_version=row["data_version"], source=row["source"],
            source_url=row["source_url"], retrieved_at=row["retrieved_at"],
        ) for row in rows]

    def _find_comp_by_id_sync(
        self, canonical_id: str, game_patch: str
    ) -> CompStatsRecord | None:
        records = self._search_comps_sync(None, game_patch, 100)
        return next((record for record in records if record.comp_id == canonical_id), None)

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
