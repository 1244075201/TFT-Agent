"""SQLite implementation of structured TFT game-data persistence."""

import asyncio
import json
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tft_agent.tft_mcp_server.repositories.migrations import apply_migrations
from tft_agent.tft_mcp_server.schemas import (
    ActivePatchContext,
    ChampionImage,
    ChampionRecord,
    CompStatsRecord,
    DataSnapshotMetadata,
    DataTFTSourceResponse,
    ItemInfo,
    ShopOddsRecord,
    StaticEntityRecord,
    TraitInfo,
)

_OPEN_QUERY_STOP_WORDS = {
    "我", "现在", "当前", "该", "怎么", "怎么玩", "怎么办", "干嘛", "下一步",
    "干什么", "做什么", "这把", "这局", "运营", "处理", "一下", "请问",
    "开局", "有", "构筑",
}


def _search_normalize(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value).casefold())


def _normalized_search_terms(queries: list[str]) -> list[str]:
    terms: list[str] = []
    for query in queries:
        normalized = _search_normalize(query)
        for stop_word in sorted(_OPEN_QUERY_STOP_WORDS, key=len, reverse=True):
            normalized = normalized.replace(stop_word, "")
        if len(normalized) >= 2:
            terms.append(normalized)
        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", query):
            token = _search_normalize(token)
            for stop_word in sorted(_OPEN_QUERY_STOP_WORDS, key=len, reverse=True):
                token = token.replace(stop_word, "")
            if len(token) >= 2:
                terms.append(token)
    return list(dict.fromkeys(terms))


def _json_text_values(value: object) -> list[str]:
    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(decoded, list):
        return [str(item) for item in decoded if str(item).strip()]
    if isinstance(decoded, dict):
        return [str(item) for item in decoded.values() if str(item).strip()]
    return [str(decoded)] if str(decoded).strip() else []


def _normalized_search_document(
    entity_type: object,
    canonical_id: object,
    display_name: object,
    section: object,
    semantic_content: str,
    comp_id: object,
    source_response_id: object,
    *,
    suffix: str = "",
    source_url: object = None,
) -> dict[str, object]:
    identity = ":".join(filter(None, (
        str(entity_type), str(canonical_id), str(section), str(comp_id or ""), suffix,
    )))
    return {
        "evidence_id": "normalized-search:" + identity,
        "entity_type": str(entity_type),
        "canonical_entity_id": str(canonical_id),
        "display_name": str(display_name),
        "section": str(section),
        "semantic_content": semantic_content,
        "comp_id": str(comp_id) if comp_id is not None else None,
        "source_response_id": (
            str(source_response_id) if source_response_id is not None else None
        ),
        "source_url": str(source_url) if source_url else None,
    }


def _relation_search_document(
    snapshot_id: str,
    game_patch: str,
    source_url: str,
    comp_id: object,
    comp_name: object,
    section: str,
    canonical_id: object,
    semantic_content: str | list[dict[str, object]] | dict[str, object],
    suffix: str,
    *,
    source_response_id: object = None,
) -> dict[str, object]:
    return {
        "evidence_id": "normalized-relation:" + suffix,
        "source": "DataTFT normalized relation",
        "source_patch": game_patch,
        "patch_match": "exact",
        "entity_type": "composition",
        "canonical_entity_id": str(comp_id),
        "display_name": str(comp_name),
        "section": section,
        "semantic_content": semantic_content,
        "relevance_score": 1.0,
        "url": source_url,
        "snapshot_id": snapshot_id,
        "comp_id": str(comp_id),
        "source_response_id": (
            str(source_response_id) if source_response_id is not None else None
        ),
        "resolution_status": "RESOLVED",
    }


def _candidate_has_seed(candidate: dict[str, object], seed_name: str) -> bool:
    signals = candidate.get("signals")
    return isinstance(signals, list) and any(
        isinstance(signal, dict) and signal.get("seed") == seed_name
        for signal in signals
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

    async def bootstrap_patch_registry(
        self, game_patch: str, set_number: int, set_name: str
    ) -> ActivePatchContext | None:
        return await asyncio.to_thread(
            self._bootstrap_patch_registry_sync, game_patch, set_number, set_name
        )

    async def active_patch_context(self) -> ActivePatchContext | None:
        return await asyncio.to_thread(self._active_patch_context_sync)

    async def patch_context(self, game_patch: str) -> ActivePatchContext | None:
        return await asyncio.to_thread(self._patch_context_sync, game_patch)

    async def create_sync_run(
        self,
        sync_id: str,
        requested_patch: str | None,
        server: str,
        set_number: int | None,
        set_name: str | None,
        started_at: datetime,
    ) -> None:
        await asyncio.to_thread(
            self._create_sync_run_sync,
            sync_id,
            requested_patch,
            server,
            set_number,
            set_name,
            started_at,
        )

    async def register_candidate_patch(
        self,
        game_patch: str,
        set_number: int,
        set_name: str,
        source: str,
        discovered_at: datetime,
    ) -> None:
        await asyncio.to_thread(
            self._register_candidate_patch_sync,
            game_patch,
            set_number,
            set_name,
            source,
            discovered_at,
        )

    async def store_source_responses(self, responses: Sequence[DataTFTSourceResponse]) -> None:
        await asyncio.to_thread(self._store_source_responses_sync, list(responses))

    async def update_sync_run(
        self,
        sync_id: str,
        status: str,
        *,
        detected_patch: str | None = None,
        snapshot_id: str | None = None,
        warnings: Sequence[str] = (),
        failure_stage: str | None = None,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._update_sync_run_sync,
            sync_id,
            status,
            detected_patch,
            snapshot_id,
            list(warnings),
            failure_stage,
            failure_code,
            failure_message,
        )

    async def store_candidate_snapshot(
        self,
        metadata: DataSnapshotMetadata,
        records: Sequence[CompStatsRecord],
        canonical_payload: str,
        sync_id: str,
        source_response_ids: dict[str, str],
        raw_source_status: str,
    ) -> tuple[str, bool]:
        return await asyncio.to_thread(
            self._store_candidate_snapshot_sync,
            metadata,
            list(records),
            canonical_payload,
            sync_id,
            source_response_ids,
            raw_source_status,
        )

    async def resolve_datatft_normalized_entities(
        self, snapshot_id: str, game_patch: str, set_number: int
    ) -> None:
        await asyncio.to_thread(
            self._resolve_datatft_normalized_entities_sync,
            snapshot_id,
            game_patch,
            set_number,
        )

    async def get_comp_board_units(
        self, snapshot_id: str, comp_id: str, phase: str | None = None
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._read_normalized_rows_sync,
            "datatft_comp_board_units",
            snapshot_id,
            comp_id,
            "board_phase",
            phase,
        )

    async def get_comp_item_plans(self, snapshot_id: str, comp_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._read_normalized_rows_sync,
            "datatft_comp_item_plan_items",
            snapshot_id,
            comp_id,
            None,
            None,
        )

    async def get_comp_items(self, snapshot_id: str, comp_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_comp_items_sync, snapshot_id, comp_id, None)

    async def get_comp_items_by_category(
        self, snapshot_id: str, comp_id: str, category: str
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_comp_items_sync, snapshot_id, comp_id, category)

    async def get_item_definition(
        self, snapshot_id: str, item_identity: str
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self._get_identity_definition_sync,
            "datatft_item_identities",
            snapshot_id,
            item_identity,
        )

    async def get_comp_conditions(self, snapshot_id: str, comp_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._read_normalized_rows_sync,
            "datatft_comp_conditions",
            snapshot_id,
            comp_id,
            None,
            None,
        )

    async def get_comp_hexes(self, snapshot_id: str, comp_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._read_normalized_rows_sync,
            "datatft_comp_hexes",
            snapshot_id,
            comp_id,
            None,
            None,
        )

    async def get_comp_augments(self, snapshot_id: str, comp_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_comp_augments_sync, snapshot_id, comp_id)

    async def get_augment_definition(
        self, snapshot_id: str, augment_identity: str
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self._get_identity_definition_sync,
            "datatft_augment_identities",
            snapshot_id,
            augment_identity,
        )

    async def get_comp_strategy_sections(
        self, snapshot_id: str, comp_id: str
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._read_normalized_rows_sync,
            "datatft_comp_strategy_sections",
            snapshot_id,
            comp_id,
            None,
            None,
        )

    async def activate_snapshot(
        self,
        game_patch: str,
        set_number: int,
        set_name: str,
        snapshot_id: str,
        sync_id: str,
        warnings: Sequence[str],
    ) -> ActivePatchContext:
        return await asyncio.to_thread(
            self._activate_snapshot_sync,
            game_patch,
            set_number,
            set_name,
            snapshot_id,
            sync_id,
            list(warnings),
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
        return await asyncio.to_thread(self._search_static_entities_sync, list(names), data_version)

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

    async def search_normalized_knowledge(
        self,
        snapshot_id: str,
        game_patch: str,
        queries: Sequence[str],
        limit: int = 12,
    ) -> list[dict[str, object]]:
        """Search active normalized knowledge without requiring canonical routing."""
        return await asyncio.to_thread(
            self._search_normalized_knowledge_sync,
            snapshot_id,
            game_patch,
            list(queries),
            limit,
        )

    async def expand_normalized_relations(
        self,
        snapshot_id: str,
        game_patch: str,
        seeds: Sequence[dict[str, str]],
        candidate_limit: int = 5,
    ) -> dict[str, object]:
        """Reverse active-snapshot relations for soft retrieval seeds."""
        return await asyncio.to_thread(
            self._expand_normalized_relations_sync,
            snapshot_id,
            game_patch,
            list(seeds),
            candidate_limit,
        )

    async def discover_compositions(
        self,
        snapshot_id: str,
        game_patch: str,
        *,
        main_unit_cost_min: int | None,
        main_unit_cost_max: int | None,
        strategy_terms: list[str],
        item_terms: list[str],
        require_strategy_match: bool,
        requested_fields: list[str],
        limit: int,
    ) -> dict[str, object]:
        """Execute the bounded semantic-discovery capability set."""
        return await asyncio.to_thread(
            self._discover_compositions_sync, snapshot_id, game_patch,
            main_unit_cost_min, main_unit_cost_max, strategy_terms, item_terms,
            require_strategy_match, requested_fields, limit,
        )

    async def rebuild_canonical_catalog(
        self,
        game_patch: str,
        set_number: int,
        set_name: str,
        explicit_aliases: Sequence[dict[str, str]] = (),
        active_static_ids: Sequence[str] = (),
        snapshot_id: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._rebuild_canonical_catalog_sync,
            game_patch,
            set_number,
            set_name,
            list(explicit_aliases),
            list(active_static_ids),
            snapshot_id,
        )

    async def canonical_catalog(self, game_patch: str, set_number: int) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._canonical_catalog_sync, game_patch, set_number)

    async def find_static_entities_by_ids(
        self, canonical_ids: Sequence[str]
    ) -> list[StaticEntityRecord]:
        return await asyncio.to_thread(self._find_static_entities_by_ids_sync, list(canonical_ids))

    async def find_comp_by_id(self, canonical_id: str, game_patch: str) -> CompStatsRecord | None:
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
                """INSERT OR IGNORE INTO datatft_snapshots (
                    snapshot_id, source, server, game_patch, sample_scope, source_url,
                    retrieved_at, content_hash, etag, status, raw_payload,
                    raw_source_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,'CANONICAL_INPUT_ONLY')""",
                (*metadata.model_dump(mode="json").values(), raw_payload),
            )
            connection.executemany(
                """INSERT OR REPLACE INTO datatft_comps (
                    snapshot_id, comp_id, name, tier, average_place, play_rate,
                    top4_rate, win_rate, core_units, recommended_items, augments, level_plan
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        metadata.snapshot_id,
                        record.comp_id,
                        record.name,
                        record.tier,
                        record.average_place,
                        record.play_rate,
                        record.top4_rate,
                        record.win_rate,
                        json.dumps(record.core_units, ensure_ascii=False),
                        json.dumps(record.recommended_items, ensure_ascii=False),
                        json.dumps(record.augments, ensure_ascii=False),
                        json.dumps(record.level_plan, ensure_ascii=False),
                    )
                    for record in records
                ],
            )

    def _bootstrap_patch_registry_sync(
        self, game_patch: str, set_number: int, set_name: str
    ) -> ActivePatchContext | None:
        from tft_agent.core.patches import patch_family

        with self._connect() as connection:
            active = connection.execute(
                "SELECT 1 FROM game_patches WHERE status='active' LIMIT 1"
            ).fetchone()
            if active is None:
                matches = connection.execute(
                    "SELECT snapshot_id FROM datatft_snapshots "
                    "WHERE game_patch=? AND status='valid' ORDER BY retrieved_at DESC",
                    (game_patch,),
                ).fetchall()
                if len(matches) == 1:
                    now = datetime.now(UTC).isoformat()
                    snapshots = connection.execute(
                        "SELECT game_patch,snapshot_id,status,retrieved_at "
                        "FROM datatft_snapshots ORDER BY retrieved_at DESC,snapshot_id DESC"
                    ).fetchall()
                    preferred: dict[str, sqlite3.Row] = {}
                    for row in snapshots:
                        preferred.setdefault(str(row["game_patch"]), row)
                    for row in preferred.values():
                        status = (
                            "active"
                            if row["snapshot_id"] == matches[0]["snapshot_id"]
                            else "historical"
                        )
                        connection.execute(
                            """INSERT OR IGNORE INTO game_patches (
                                game_patch,patch_family,set_number,set_name,status,
                                active_snapshot_id,source,discovered_at,synced_at,
                                validated_at,activated_at
                            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                row["game_patch"],
                                patch_family(str(row["game_patch"])),
                                set_number,
                                set_name,
                                status,
                                row["snapshot_id"],
                                "legacy",
                                row["retrieved_at"],
                                row["retrieved_at"],
                                row["retrieved_at"],
                                now if status == "active" else None,
                            ),
                        )
        return self._active_patch_context_sync()

    def _active_patch_context_sync(self) -> ActivePatchContext | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM game_patches WHERE status='active'").fetchone()
        return self._context_from_row(row)

    def _patch_context_sync(self, game_patch: str) -> ActivePatchContext | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM game_patches WHERE game_patch=? AND active_snapshot_id IS NOT NULL",
                (game_patch,),
            ).fetchone()
        return self._context_from_row(row)

    @staticmethod
    def _context_from_row(row: sqlite3.Row | None) -> ActivePatchContext | None:
        if row is None:
            return None
        return ActivePatchContext(
            game_patch=row["game_patch"],
            patch_family=row["patch_family"],
            set_number=row["set_number"],
            set_name=row["set_name"],
            active_snapshot_id=row["active_snapshot_id"],
        )

    def _create_sync_run_sync(
        self,
        sync_id: str,
        requested_patch: str | None,
        server: str,
        set_number: int | None,
        set_name: str | None,
        started_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO datatft_sync_runs "
                "(sync_id,requested_patch,set_number,set_name,server,status,started_at) "
                "VALUES (?,?,?,?,?,'running',?)",
                (sync_id, requested_patch, set_number, set_name, server, started_at.isoformat()),
            )

    def _register_candidate_patch_sync(
        self,
        game_patch: str,
        set_number: int,
        set_name: str,
        source: str,
        discovered_at: datetime,
    ) -> None:
        from tft_agent.core.patches import patch_family

        with self._connect() as connection:
            connection.execute(
                """INSERT INTO game_patches (
                    game_patch,patch_family,set_number,set_name,status,source,discovered_at
                ) VALUES (?,?,?,?,'candidate',?,?)
                ON CONFLICT(game_patch) DO UPDATE SET
                    patch_family=excluded.patch_family,set_number=excluded.set_number,
                    set_name=excluded.set_name,source=excluded.source""",
                (
                    game_patch,
                    patch_family(game_patch),
                    set_number,
                    set_name,
                    source,
                    discovered_at.isoformat(),
                ),
            )

    async def snapshot_comp_count(self, snapshot_id: str) -> int:
        return await asyncio.to_thread(self._snapshot_comp_count_sync, snapshot_id)

    def _snapshot_comp_count_sync(self, snapshot_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM datatft_comps WHERE snapshot_id=?", (snapshot_id,)
                ).fetchone()[0]
            )

    def _store_source_responses_sync(self, responses: list[DataTFTSourceResponse]) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO datatft_source_responses "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [tuple(item.model_dump(mode="json").values()) for item in responses],
            )

    def _update_sync_run_sync(
        self,
        sync_id: str,
        status: str,
        detected_patch: str | None,
        snapshot_id: str | None,
        warnings: list[str],
        failure_stage: str | None,
        failure_code: str | None,
        failure_message: str | None,
    ) -> None:
        completed = datetime.now(UTC).isoformat() if status in {"activated", "failed"} else None
        with self._connect() as connection:
            connection.execute(
                """UPDATE datatft_sync_runs SET status=?,
                    detected_patch=COALESCE(?,detected_patch),
                    canonical_snapshot_id=COALESCE(?,canonical_snapshot_id),
                    warning_json=?, failure_stage=?, failure_code=?, failure_message=?,
                    completed_at=COALESCE(?,completed_at) WHERE sync_id=?""",
                (
                    status,
                    detected_patch,
                    snapshot_id,
                    json.dumps(warnings, ensure_ascii=False),
                    failure_stage,
                    failure_code,
                    failure_message,
                    completed,
                    sync_id,
                ),
            )

    def _store_candidate_snapshot_sync(
        self,
        metadata: DataSnapshotMetadata,
        records: list[CompStatsRecord],
        canonical_payload: str,
        sync_id: str,
        source_response_ids: dict[str, str],
        raw_source_status: str,
    ) -> tuple[str, bool]:
        if not records or any(item.snapshot.game_patch != metadata.game_patch for item in records):
            raise ValueError("candidate records must be non-empty and match the snapshot patch")
        root = json.loads(canonical_payload)
        format_version = int(root.get("canonical_format_version", 1))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if raw_source_status == "AVAILABLE":
                missing = {item.comp_id for item in records} - source_response_ids.keys()
                if missing:
                    raise ValueError("every remote composition requires a detail source response")
                marks = ",".join("?" for _ in source_response_ids.values())
                matched = connection.execute(
                    f"SELECT COUNT(*) FROM datatft_source_responses "
                    f"WHERE sync_id=? AND response_id IN ({marks})",
                    (sync_id, *source_response_ids.values()),
                ).fetchone()[0]
                if matched != len(source_response_ids):
                    raise ValueError("composition source responses must belong to the creator run")
            existing = connection.execute(
                "SELECT snapshot_id FROM datatft_snapshots WHERE game_patch=? "
                "AND canonical_format_version=? AND content_hash=?",
                (metadata.game_patch, format_version, metadata.content_hash),
            ).fetchone()
            if existing is not None:
                return str(existing["snapshot_id"]), False
            connection.execute(
                """INSERT INTO datatft_snapshots (
                    snapshot_id,source,server,game_patch,sample_scope,source_url,retrieved_at,
                    content_hash,etag,status,raw_payload,sync_id,canonical_format_version,
                    raw_source_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?, ?,?,?)""",
                (
                    *metadata.model_dump(mode="json").values(),
                    canonical_payload,
                    sync_id,
                    format_version,
                    raw_source_status,
                ),
            )
            connection.executemany(
                """INSERT INTO datatft_comps (
                    snapshot_id,comp_id,name,tier,average_place,play_rate,top4_rate,win_rate,
                    core_units,recommended_items,augments,level_plan,source_response_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        metadata.snapshot_id,
                        item.comp_id,
                        item.name,
                        item.tier,
                        item.average_place,
                        item.play_rate,
                        item.top4_rate,
                        item.win_rate,
                        json.dumps(item.core_units, ensure_ascii=False),
                        json.dumps(item.recommended_items, ensure_ascii=False),
                        json.dumps(item.augments, ensure_ascii=False),
                        json.dumps(item.level_plan, ensure_ascii=False),
                        source_response_ids.get(item.comp_id),
                    )
                    for item in records
                ],
            )
            if format_version >= 2:
                self._store_normalized_knowledge(
                    connection, metadata.snapshot_id, root, source_response_ids
                )
        return metadata.snapshot_id, True

    def _store_normalized_knowledge(
        self,
        connection: sqlite3.Connection,
        snapshot_id: str,
        root: dict[str, Any],
        source_response_ids: dict[str, str],
    ) -> None:
        for comp in root.get("comps", []):
            if not isinstance(comp, dict) or not isinstance(comp.get("knowledge_v2"), dict):
                continue
            comp_id = str(comp["id"])
            source_id = source_response_ids.get(comp_id)
            knowledge = comp["knowledge_v2"]
            metadata = knowledge.get("metadata") or {}
            representative = metadata.get("representative_hero") or {}
            connection.execute(
                "INSERT INTO datatft_comp_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    snapshot_id,
                    comp_id,
                    metadata.get("difficulty"),
                    json.dumps(metadata.get("tags") or [], ensure_ascii=False),
                    metadata.get("upstream_updated_at"),
                    representative.get("external_hero_id"),
                    representative.get("external_hero_key"),
                    None,
                    representative.get("display_name"),
                    representative.get("resolution_status", "UNRESOLVED"),
                    source_id,
                ),
            )
            for phase, board in (knowledge.get("boards") or {}).items():
                for unit in (board or {}).get("units", []):
                    board_index = unit.get("board_index")
                    if board_index is None:
                        continue
                    connection.execute(
                        "INSERT INTO datatft_comp_board_units VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            snapshot_id,
                            comp_id,
                            phase,
                            board_index,
                            unit.get("x"),
                            unit.get("y"),
                            unit.get("external_hero_id"),
                            unit.get("external_hero_key"),
                            None,
                            unit.get("display_name"),
                            unit.get("stars"),
                            unit.get("is_main"),
                            unit.get("replace_id"),
                            unit.get("replace_key"),
                            json.dumps(unit.get("extra_traits") or [], ensure_ascii=False),
                            json.dumps(unit.get("powerups") or {}, ensure_ascii=False),
                            unit.get("resolution_status", "UNRESOLVED"),
                            unit.get("unresolved_raw_token"),
                            source_id,
                        ),
                    )
            for plan in knowledge.get("item_plans") or []:
                holder, item = plan.get("holder") or {}, plan.get("item") or {}
                holder_id = holder.get("external_hero_id") or holder.get("unresolved_raw_token")
                item_id = item.get("external_item_id") or item.get("unresolved_raw_token")
                if item_id:
                    connection.execute(
                        "INSERT OR IGNORE INTO datatft_item_identities VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            snapshot_id,
                            item_id,
                            item.get("external_item_key"),
                            None,
                            item.get("display_name"),
                            item.get("item_category", "OTHER"),
                            item.get("runtime_type"),
                            item.get("runtime_description"),
                            None,
                            None,
                            item.get("resolution_status", "UNRESOLVED"),
                            item.get("unresolved_raw_token"),
                        ),
                    )
                connection.execute(
                    "INSERT INTO datatft_comp_item_plan_items VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        comp_id,
                        holder_id,
                        holder.get("external_hero_key"),
                        None,
                        holder.get("display_name"),
                        holder.get("resolution_status", "UNRESOLVED"),
                        plan["plan_index"],
                        plan["slot_index"],
                        item.get("external_item_id"),
                        item.get("external_item_key"),
                        None,
                        item.get("display_name"),
                        item.get("resolution_status", "UNRESOLVED"),
                        item.get("unresolved_raw_token"),
                        source_id,
                    ),
                )
            for index, condition in enumerate(knowledge.get("conditions") or []):
                connection.execute(
                    "INSERT INTO datatft_comp_conditions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        comp_id,
                        index,
                        condition["condition_kind"],
                        condition.get("condition_type"),
                        condition.get("id"),
                        condition.get("key"),
                        condition.get("name"),
                        condition.get("level_value"),
                        condition.get("trait_level_text"),
                        condition.get("trait_level_idx"),
                        condition.get("carry_hero_key"),
                        condition.get("carry_hero_value"),
                        condition["raw_entry_json"],
                        source_id,
                    ),
                )
            for index, item in enumerate(knowledge.get("hexes") or []):
                hex_id = (
                    item.get("external_entity_id")
                    or item.get("external_hex_id")
                    or item.get("unresolved_raw_token")
                )
                hex_key = item.get("external_entity_key") or item.get("external_hex_key")
                if hex_id:
                    connection.execute(
                        "INSERT OR IGNORE INTO datatft_augment_identities VALUES "
                        "(?,?,?,?,?,?,?,?,?,?)",
                        (
                            snapshot_id,
                            hex_id,
                            hex_key,
                            None,
                            item.get("display_name"),
                            item.get("runtime_description"),
                            None,
                            None,
                            item.get("resolution_status", "UNRESOLVED"),
                            item.get("unresolved_raw_token"),
                        ),
                    )
                connection.execute(
                    "INSERT INTO datatft_comp_hexes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        comp_id,
                        index,
                        item["source_kind"],
                        hex_id,
                        hex_key,
                        None,
                        item.get("display_name"),
                        item.get("resolution_status", "UNRESOLVED"),
                        item.get("unresolved_raw_token"),
                        None,
                        item.get("rank"),
                        source_id,
                    ),
                )
            for section in knowledge.get("strategy_sections") or []:
                connection.execute(
                    "INSERT INTO datatft_comp_strategy_sections VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        comp_id,
                        section["section_type"],
                        section["ordinal"],
                        section["text"],
                        json.dumps(section.get("labels") or [], ensure_ascii=False),
                        section.get("author"),
                        section.get("upstream_updated_at"),
                        section.get("raw_entry_json"),
                        source_id,
                    ),
                )

    def _resolve_datatft_normalized_entities_sync(
        self, snapshot_id: str, game_patch: str, set_number: int
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for table, key_column, canonical_column, kinds in (
                (
                    "datatft_item_identities",
                    "external_item_key",
                    "canonical_item_id",
                    ("item", "emblem"),
                ),
                (
                    "datatft_augment_identities",
                    "external_hex_key",
                    "canonical_augment_id",
                    ("augment",),
                ),
            ):
                marks = ",".join("?" for _ in kinds)
                connection.execute(
                    f"UPDATE {table} SET {canonical_column}=(SELECT canonical_id FROM "
                    f"canonical_entities e WHERE e.canonical_id={table}.{key_column} "
                    f"AND e.game_patch=? AND e.set_number=? AND e.entity_type IN ({marks})), "
                    f"static_entity_id=(SELECT entity_id FROM static_entities s "
                    f"WHERE s.entity_id={table}.{key_column} ORDER BY s.retrieved_at DESC LIMIT 1) "
                    f"WHERE snapshot_id=? AND {key_column} IS NOT NULL",
                    (game_patch, set_number, *kinds, snapshot_id),
                )
                connection.execute(
                    f"UPDATE {table} SET definition_source=CASE "
                    "WHEN EXISTS (SELECT 1 FROM static_entities s WHERE "
                    f"s.entity_id={table}.static_entity_id AND LENGTH(s.description)>0) "
                    "THEN 'communitydragon' WHEN runtime_description IS NOT NULL "
                    "THEN 'datatft_runtime' ELSE NULL END, "
                    "resolution_status=CASE "
                    f"WHEN {canonical_column} IS NOT NULL THEN 'EXACT' "
                    "WHEN static_entity_id IS NOT NULL THEN 'STATIC_ONLY' "
                    "ELSE 'NO_STATIC_MATCH' END WHERE snapshot_id=?",
                    (snapshot_id,),
                )
            connection.execute(
                "UPDATE datatft_item_identities SET item_category=CASE "
                "WHEN EXISTS (SELECT 1 FROM static_entities s WHERE "
                "s.entity_id=static_entity_id AND s.entity_type='emblem') THEN 'EMBLEM' "
                "WHEN EXISTS (SELECT 1 FROM static_entities s WHERE "
                "s.entity_id=static_entity_id AND s.tags LIKE '%{44ace175}%') THEN 'ARTIFACT' "
                "WHEN EXISTS (SELECT 1 FROM static_entities s WHERE "
                "s.entity_id=static_entity_id AND s.tags LIKE '%{ec243f6b}%') THEN 'RADIANT' "
                "ELSE item_category END WHERE snapshot_id=? AND item_category='OTHER'",
                (snapshot_id,),
            )
            for table, key_column, canonical_column, status_column, kinds in (
                (
                    "datatft_comp_board_units",
                    "external_hero_key",
                    "canonical_champion_id",
                    "resolution_status",
                    ("champion",),
                ),
                (
                    "datatft_comp_item_plan_items",
                    "holder_external_hero_key",
                    "holder_canonical_champion_id",
                    "holder_resolution_status",
                    ("champion",),
                ),
                (
                    "datatft_comp_item_plan_items",
                    "external_item_key",
                    "canonical_item_id",
                    "resolution_status",
                    ("item", "emblem"),
                ),
                (
                    "datatft_comp_metadata",
                    "representative_external_hero_key",
                    "representative_canonical_champion_id",
                    "representative_resolution_status",
                    ("champion",),
                ),
                (
                    "datatft_comp_hexes",
                    "external_hex_key",
                    "canonical_augment_id",
                    "resolution_status",
                    ("augment",),
                ),
            ):
                marks = ",".join("?" for _ in kinds)
                connection.execute(
                    f"UPDATE {table} SET {canonical_column}=(SELECT canonical_id FROM "
                    f"canonical_entities e WHERE e.canonical_id={table}.{key_column} "
                    f"AND e.game_patch=? AND e.set_number=? AND e.entity_type IN ({marks})) "
                    f"WHERE snapshot_id=? AND {key_column} IS NOT NULL",
                    (game_patch, set_number, *kinds, snapshot_id),
                )
                connection.execute(
                    f"UPDATE {table} SET {status_column}=CASE "
                    f"WHEN {canonical_column} IS NOT NULL THEN 'EXACT' "
                    f"WHEN EXISTS (SELECT 1 FROM static_entities s "
                    f"WHERE s.entity_id={table}.{key_column}) THEN 'STATIC_ONLY' "
                    "ELSE 'NO_CANONICAL_MATCH' END "
                    f"WHERE snapshot_id=? AND {key_column} IS NOT NULL",
                    (snapshot_id,),
                )

    def _read_normalized_rows_sync(
        self,
        table: str,
        snapshot_id: str,
        comp_id: str,
        filter_column: str | None,
        filter_value: str | None,
    ) -> list[dict[str, Any]]:
        allowed = {
            "datatft_comp_board_units",
            "datatft_comp_item_plan_items",
            "datatft_comp_conditions",
            "datatft_comp_hexes",
            "datatft_comp_strategy_sections",
        }
        if table not in allowed:
            raise ValueError("unsupported normalized table")
        sql = f"SELECT * FROM {table} WHERE snapshot_id=? AND comp_id=?"
        params: list[object] = [snapshot_id, comp_id]
        if filter_column == "board_phase" and filter_value is not None:
            sql += " AND board_phase=?"
            params.append(filter_value)
        sql += " ORDER BY rowid"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def _get_comp_items_sync(
        self, snapshot_id: str, comp_id: str, category: str | None
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT p.*,i.item_category,i.runtime_type,i.static_entity_id,"
            "i.runtime_description,i.definition_source,"
            "i.resolution_status AS identity_resolution_status "
            "FROM datatft_comp_item_plan_items p "
            "JOIN datatft_item_identities i ON i.snapshot_id=p.snapshot_id "
            "AND i.external_item_id=p.external_item_id "
            "WHERE p.snapshot_id=? AND p.comp_id=?"
        )
        params: list[object] = [snapshot_id, comp_id]
        if category is not None:
            sql += " AND i.item_category=?"
            params.append(category)
        sql += " ORDER BY p.holder_external_hero_id,p.plan_index,p.slot_index"
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(sql, params).fetchall()]
            for row in rows:
                row["definition"] = self._definition_sync(
                    connection,
                    row.get("static_entity_id"),
                    row.get("runtime_description"),
                    row.get("item_display_name"),
                    row.get("definition_source"),
                )
        return rows

    def _get_comp_augments_sync(self, snapshot_id: str, comp_id: str) -> list[dict[str, Any]]:
        sql = (
            "SELECT a.*,i.static_entity_id,i.runtime_description,i.definition_source,"
            "i.resolution_status AS identity_resolution_status "
            "FROM datatft_comp_hexes a JOIN datatft_augment_identities i "
            "ON i.snapshot_id=a.snapshot_id AND i.external_hex_id=a.external_hex_id "
            "WHERE a.snapshot_id=? AND a.comp_id=? ORDER BY a.association_index"
        )
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(sql, (snapshot_id, comp_id)).fetchall()]
            for row in rows:
                row["definition"] = self._definition_sync(
                    connection,
                    row.get("static_entity_id"),
                    row.get("runtime_description"),
                    row.get("display_name"),
                    row.get("definition_source"),
                )
        return rows

    def _get_identity_definition_sync(
        self, table: str, snapshot_id: str, identity: str
    ) -> dict[str, Any] | None:
        columns = {
            "datatft_item_identities": ("external_item_id", "external_item_key"),
            "datatft_augment_identities": ("external_hex_id", "external_hex_key"),
        }
        if table not in columns:
            raise ValueError("unsupported identity table")
        id_column, key_column = columns[table]
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE snapshot_id=? "
                f"AND ({id_column}=? OR {key_column}=?) LIMIT 1",
                (snapshot_id, identity, identity),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["definition"] = self._definition_sync(
                connection,
                result.get("static_entity_id"),
                result.get("runtime_description"),
                result.get("display_name"),
                result.get("definition_source"),
            )
            return result

    @staticmethod
    def _static_definition_sync(
        connection: sqlite3.Connection, entity_id: object
    ) -> dict[str, Any] | None:
        if not entity_id:
            return None
        row = connection.execute(
            "SELECT entity_id,entity_type,name,description,tags,source,source_url,data_version "
            "FROM static_entities WHERE entity_id=? ORDER BY retrieved_at DESC LIMIT 1",
            (entity_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["tags"] = json.loads(result["tags"])
        return result

    @classmethod
    def _definition_sync(
        cls,
        connection: sqlite3.Connection,
        static_entity_id: object,
        runtime_description: object,
        display_name: object,
        definition_source: object,
    ) -> dict[str, Any] | None:
        static = cls._static_definition_sync(connection, static_entity_id)
        if static is not None and static.get("description"):
            return static
        if runtime_description:
            return {
                "entity_id": static_entity_id,
                "name": display_name,
                "description": runtime_description,
                "source": definition_source or "datatft_runtime",
            }
        return static

    def _activate_snapshot_sync(
        self,
        game_patch: str,
        set_number: int,
        set_name: str,
        snapshot_id: str,
        sync_id: str,
        warnings: list[str],
    ) -> ActivePatchContext:
        from tft_agent.core.patches import patch_family

        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            snapshot = connection.execute(
                "SELECT game_patch FROM datatft_snapshots WHERE snapshot_id=? AND status='valid'",
                (snapshot_id,),
            ).fetchone()
            if snapshot is None or snapshot["game_patch"] != game_patch:
                raise ValueError("active snapshot must be valid and match its patch")
            catalog_count = connection.execute(
                "SELECT COUNT(*) FROM canonical_entities WHERE game_patch=? AND set_number=? "
                "AND entity_type='composition'",
                (game_patch, set_number),
            ).fetchone()[0]
            comp_count = connection.execute(
                "SELECT COUNT(*) FROM datatft_comps WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()[0]
            if comp_count == 0 or catalog_count != comp_count:
                raise ValueError("candidate catalog does not match the candidate snapshot")
            connection.execute("UPDATE game_patches SET status='historical' WHERE status='active'")
            connection.execute(
                """INSERT INTO game_patches (
                    game_patch,patch_family,set_number,set_name,status,active_snapshot_id,
                    source,discovered_at,synced_at,validated_at,activated_at
                ) VALUES (?,?,?,?,'active',?,'datatft',?,?,?,?)
                ON CONFLICT(game_patch) DO UPDATE SET
                    patch_family=excluded.patch_family,set_number=excluded.set_number,
                    set_name=excluded.set_name,status='active',
                    active_snapshot_id=excluded.active_snapshot_id,synced_at=excluded.synced_at,
                    validated_at=excluded.validated_at,activated_at=excluded.activated_at""",
                (
                    game_patch,
                    patch_family(game_patch),
                    set_number,
                    set_name,
                    snapshot_id,
                    now,
                    now,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE datatft_snapshots SET validation_report=?,validated_at=? "
                "WHERE snapshot_id=?",
                (json.dumps({"warnings": warnings}, ensure_ascii=False), now, snapshot_id),
            )
            connection.execute(
                """UPDATE datatft_sync_runs SET status='activated',detected_patch=?,
                    canonical_snapshot_id=?,warning_json=?,completed_at=? WHERE sync_id=?""",
                (game_patch, snapshot_id, json.dumps(warnings, ensure_ascii=False), now, sync_id),
            )
        context = self._active_patch_context_sync()
        if context is None:
            raise RuntimeError("activation committed without an active patch")
        return context

    def _search_comps_sync(
        self, query: str | None, game_patch: str, limit: int
    ) -> list[CompStatsRecord]:
        parameters: list[object] = [game_patch]
        name_clause = ""
        with self._connect() as connection:
            registered = connection.execute(
                "SELECT active_snapshot_id FROM game_patches WHERE game_patch=?",
                (game_patch,),
            ).fetchone()
        snapshot_clause = ""
        if registered is not None and registered["active_snapshot_id"]:
            snapshot_clause = " AND s.snapshot_id = ?"
            parameters.append(registered["active_snapshot_id"])
        if query:
            name_clause = " AND c.name LIKE ?"
            parameters.append(f"%{query}%")
        parameters.append(limit)
        sql = (
            "SELECT c.*, s.* FROM datatft_comps c JOIN datatft_snapshots s USING(snapshot_id) "
            "WHERE s.game_patch = ? AND s.status IN ('valid','stale')"
            + snapshot_clause
            + name_clause
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
                    comp_id=row["comp_id"],
                    name=row["name"],
                    tier=row["tier"],
                    average_place=row["average_place"],
                    play_rate=row["play_rate"],
                    top4_rate=row["top4_rate"],
                    win_rate=row["win_rate"],
                    core_units=json.loads(row["core_units"]),
                    recommended_items=json.loads(row["recommended_items"]),
                    augments=json.loads(row["augments"]),
                    level_plan=json.loads(row["level_plan"]),
                    source_response_id=row["source_response_id"],
                    snapshot=snapshot,
                )
            )
        return output

    def _latest_datatft_snapshot_sync(self) -> tuple[DataSnapshotMetadata, int] | None:
        sql = (
            "SELECT s.*, COUNT(c.comp_id) AS comp_count FROM datatft_snapshots s "
            "JOIN game_patches p ON p.active_snapshot_id=s.snapshot_id "
            "LEFT JOIN datatft_comps c USING(snapshot_id) WHERE p.status='active' "
            "GROUP BY s.snapshot_id LIMIT 1"
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
                        record.data_version,
                        record.entity_id,
                        record.entity_type,
                        record.name,
                        record.description,
                        json.dumps(record.traits, ensure_ascii=False),
                        json.dumps(record.components, ensure_ascii=False),
                        json.dumps(record.tags, ensure_ascii=False),
                        record.source,
                        record.source_url,
                        record.retrieved_at.isoformat(),
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
                entity_id=row["entity_id"],
                entity_type=row["entity_type"],
                name=row["name"],
                description=row["description"],
                traits=json.loads(row["traits"]),
                components=json.loads(row["components"]),
                tags=json.loads(row["tags"]),
                data_version=row["data_version"],
                source=row["source"],
                source_url=row["source_url"],
                retrieved_at=row["retrieved_at"],
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

    def _load_session_messages_sync(self, session_id: str, limit: int) -> list[tuple[str, str]]:
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
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
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

    def _discover_compositions_sync(
        self,
        snapshot_id: str,
        game_patch: str,
        main_unit_cost_min: int | None,
        main_unit_cost_max: int | None,
        strategy_terms: list[str],
        item_terms: list[str],
        require_strategy_match: bool,
        requested_fields: list[str],
        limit: int,
    ) -> dict[str, object]:
        """Filter active-snapshot compositions using only normalized, existing fields."""
        strategy_terms = list(dict.fromkeys(
            _search_normalize(term) for term in strategy_terms if _search_normalize(term)
        ))
        item_terms = list(dict.fromkeys(
            _search_normalize(term) for term in item_terms if _search_normalize(term)
        ))
        with self._connect() as connection:
            snapshot = connection.execute(
                "SELECT source_url FROM datatft_snapshots WHERE snapshot_id=? AND game_patch=?",
                (snapshot_id, game_patch),
            ).fetchone()
            if snapshot is None:
                return {"evidence": [], "candidate_comps": []}
            comps = connection.execute(
                "SELECT comp_id,name,tier,source_response_id FROM datatft_comps "
                "WHERE snapshot_id=? ORDER BY name", (snapshot_id,),
            ).fetchall()
            candidates: list[dict[str, object]] = []
            evidence: list[dict[str, object]] = []
            requested = set(requested_fields)
            strategy_fields = requested & {
                "starting_conditions", "core", "early_game", "mid_game", "late_game", "tips",
            }
            section_names = {"tips": "tip", **{name: name for name in strategy_fields}}
            for comp in comps:
                comp_id = str(comp["comp_id"])
                board = connection.execute(
                    "SELECT b.display_name,b.canonical_champion_id,b.stars,b.is_main,b.board_index,"
                    "(SELECT ch.cost FROM champions ch WHERE "
                    "ch.champion_id=b.canonical_champion_id "
                    "ORDER BY ch.retrieved_at DESC LIMIT 1) cost,b.source_response_id "
                    "FROM datatft_comp_board_units b WHERE b.snapshot_id=? AND b.comp_id=? "
                    "AND b.board_phase='FINAL' ORDER BY b.board_index", (snapshot_id, comp_id),
                ).fetchall()
                mains = [row for row in board if int(row["is_main"] or 0) == 1]
                if not mains:
                    representative = connection.execute(
                        "SELECT representative_canonical_champion_id FROM datatft_comp_metadata "
                        "WHERE snapshot_id=? AND comp_id=?", (snapshot_id, comp_id),
                    ).fetchone()
                    representative_id = str(representative[0] or "") if representative else ""
                    mains = [row for row in board
                             if str(row["canonical_champion_id"] or "") == representative_id]
                main_costs = [int(row["cost"]) for row in mains if row["cost"] is not None]
                if main_unit_cost_min is not None and (
                    not main_costs or min(main_costs) < main_unit_cost_min
                ):
                    continue
                if main_unit_cost_max is not None and (
                    not main_costs or max(main_costs) > main_unit_cost_max
                ):
                    continue
                sections = connection.execute(
                    "SELECT section_type,ordinal,text,source_response_id "
                    "FROM datatft_comp_strategy_sections WHERE snapshot_id=? AND comp_id=? "
                    "ORDER BY section_type,ordinal", (snapshot_id, comp_id),
                ).fetchall()
                strategy_matches = [row for row in sections if any(
                    term in _search_normalize(row["text"]) for term in strategy_terms
                )]
                if require_strategy_match and not strategy_matches:
                    continue
                items = connection.execute(
                    "SELECT holder_display_name,item_display_name,canonical_item_id,plan_index,"
                    "slot_index,source_response_id FROM datatft_comp_item_plan_items "
                    "WHERE snapshot_id=? AND comp_id=? ORDER BY plan_index,slot_index",
                    (snapshot_id, comp_id),
                ).fetchall()
                item_matches = [row for row in items if any(
                    term in _search_normalize(
                        f"{row['holder_display_name']} {row['item_display_name']}"
                    ) for term in item_terms
                )]
                if item_terms and not item_matches:
                    continue
                signals: list[str] = []
                if main_costs:
                    signals.append("main_unit_cost")
                if strategy_matches:
                    signals.append("strategy_text_match")
                if item_matches:
                    signals.append("item_plan_match")
                candidates.append({
                    "comp_id": comp_id, "name": str(comp["name"]), "tier": comp["tier"],
                    "main_units": [{"name": str(row["display_name"]), "cost": row["cost"],
                                    "stars": row["stars"]} for row in mains],
                    "signals": signals,
                })
                common = (snapshot_id, game_patch, str(snapshot["source_url"]), comp_id,
                          str(comp["name"]))
                evidence.append(_relation_search_document(
                    *common, "candidate_composition", comp_id,
                    {"tier": comp["tier"], "main_units": candidates[-1]["main_units"],
                     "qualification_signals": signals,
                     "retrieval_order_is_not_strength_ranking": True},
                    f"semantic:candidate:{comp_id}", source_response_id=comp["source_response_id"],
                ))
                selected_sections = [
                    row for row in sections
                    if row in strategy_matches
                    or row["section_type"] in set(section_names.values())
                ]
                for row in selected_sections:
                    evidence.append(_relation_search_document(
                        *common, str(row["section_type"]), comp_id, str(row["text"]),
                        f"semantic:strategy:{comp_id}:{row['section_type']}:{row['ordinal']}",
                        source_response_id=row["source_response_id"],
                    ))
                if "roster" in requested:
                    evidence.append(_relation_search_document(
                        *common, "composition_roster", comp_id,
                        [{"name": row["display_name"], "stars": row["stars"],
                          "is_main": bool(row["is_main"])} for row in board],
                        f"semantic:roster:{comp_id}",
                    ))
                if "items" in requested and items:
                    evidence.append(_relation_search_document(
                        *common, "item_plan", comp_id,
                        [{"holder": row["holder_display_name"], "item": row["item_display_name"],
                          "plan": row["plan_index"], "slot": row["slot_index"]}
                         for row in items], f"semantic:items:{comp_id}",
                    ))
                if len(candidates) >= limit:
                    break
        selected_ids = {str(row["comp_id"]) for row in candidates}
        return {"candidate_comps": candidates,
                "evidence": [row for row in evidence if str(row["comp_id"]) in selected_ids]}

    def _search_normalized_knowledge_sync(
        self,
        snapshot_id: str,
        game_patch: str,
        queries: list[str],
        limit: int,
    ) -> list[dict[str, object]]:
        """Rank presentation-safe evidence from the active normalized snapshot."""
        cleaned_queries = list(dict.fromkeys(value.strip() for value in queries if value.strip()))
        if not cleaned_queries:
            return []
        with self._connect() as connection:
            snapshot = connection.execute(
                "SELECT source, source_url FROM datatft_snapshots "
                "WHERE snapshot_id=? AND game_patch=?",
                (snapshot_id, game_patch),
            ).fetchone()
            if snapshot is None:
                return []
            documents: list[dict[str, object]] = []

            comps = connection.execute(
                "SELECT comp_id, name, tier, core_units, level_plan, source_response_id "
                "FROM datatft_comps WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchall()
            comp_names = {str(row["comp_id"]): str(row["name"]) for row in comps}
            for row in comps:
                core_units = _json_text_values(row["core_units"])
                level_plan = _json_text_values(row["level_plan"])
                text = "；".join(filter(None, (
                    f"阵容：{row['name']}",
                    f"评级：{row['tier']}" if row["tier"] else "",
                    "核心棋子：" + "、".join(core_units) if core_units else "",
                    "等级规划：" + "；".join(level_plan) if level_plan else "",
                )))
                documents.append(_normalized_search_document(
                    "composition", row["comp_id"], row["name"], "composition",
                    text, row["comp_id"], row["source_response_id"],
                ))

            board_rows = connection.execute(
                "SELECT b.comp_id, b.display_name, b.canonical_champion_id, b.x, b.y, "
                "b.stars, b.source_response_id, s.traits FROM datatft_comp_board_units b "
                "LEFT JOIN static_entities s ON s.entity_id=b.canonical_champion_id "
                "AND s.retrieved_at=(SELECT MAX(s2.retrieved_at) FROM static_entities s2 "
                "WHERE s2.entity_id=b.canonical_champion_id) "
                "WHERE b.snapshot_id=? AND b.board_phase='FINAL' "
                "ORDER BY b.comp_id, b.board_index",
                (snapshot_id,),
            ).fetchall()
            for row in board_rows:
                name = str(row["display_name"] or "未解析棋子")
                traits = _json_text_values(row["traits"])
                position = (
                    f"，站位({row['x']},{row['y']})"
                    if row["x"] is not None and row["y"] is not None else ""
                )
                documents.append(_normalized_search_document(
                    "champion", row["canonical_champion_id"] or name,
                    name, "composition_roster",
                    f"{comp_names.get(str(row['comp_id']), '阵容')}：{name}{position}"
                    + ("，羁绊：" + "、".join(traits) if traits else ""),
                    row["comp_id"], row["source_response_id"],
                ))

            item_rows = connection.execute(
                "SELECT p.comp_id, p.holder_display_name, p.holder_canonical_champion_id, "
                "p.item_display_name, p.canonical_item_id, p.plan_index, p.slot_index, "
                "p.source_response_id, i.item_category, i.runtime_description "
                "FROM datatft_comp_item_plan_items p LEFT JOIN datatft_item_identities i "
                "ON i.snapshot_id=p.snapshot_id AND i.canonical_item_id=p.canonical_item_id "
                "WHERE p.snapshot_id=?",
                (snapshot_id,),
            ).fetchall()
            for row in item_rows:
                item_name = str(row["item_display_name"] or "未解析装备")
                holder = str(row["holder_display_name"] or "未解析棋子")
                category = str(row["item_category"] or "item").casefold()
                section = category if category in {"artifact", "radiant", "emblem"} else "item_plan"
                description = str(row["runtime_description"] or "").strip()
                documents.append(_normalized_search_document(
                    section, row["canonical_item_id"] or item_name, item_name, section,
                    f"{comp_names.get(str(row['comp_id']), '阵容')}：{holder}的第"
                    f"{int(row['plan_index']) + 1}套装备第{int(row['slot_index']) + 1}件是"
                    f"{item_name}" + (f"；效果：{description}" if description else ""),
                    row["comp_id"], row["source_response_id"],
                ))

            augment_rows = connection.execute(
                "SELECT h.comp_id, h.display_name, h.canonical_augment_id, h.source_kind, "
                "h.source_response_id, i.runtime_description "
                "FROM datatft_comp_hexes h LEFT JOIN datatft_augment_identities i "
                "ON i.snapshot_id=h.snapshot_id AND i.canonical_augment_id=h.canonical_augment_id "
                "WHERE h.snapshot_id=?",
                (snapshot_id,),
            ).fetchall()
            for row in augment_rows:
                name = str(row["display_name"] or "未解析强化")
                description = str(row["runtime_description"] or "").strip()
                documents.append(_normalized_search_document(
                    "augment", row["canonical_augment_id"] or name, name, "augment",
                    f"{comp_names.get(str(row['comp_id']), '阵容')}关联强化：{name}"
                    + (f"；效果：{description}" if description else ""),
                    row["comp_id"], row["source_response_id"],
                ))

            for row in connection.execute(
                "SELECT comp_id, condition_kind, display_name, level_value, trait_level_text, "
                "carry_hero_value, source_response_id FROM datatft_comp_conditions "
                "WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchall():
                values = [str(row[key]) for key in (
                    "display_name", "level_value", "trait_level_text", "carry_hero_value"
                ) if row[key] not in (None, "")]
                documents.append(_normalized_search_document(
                    "condition", f"{row['comp_id']}:{row['condition_kind']}",
                    comp_names.get(str(row["comp_id"]), "阵容"), "condition",
                    f"{comp_names.get(str(row['comp_id']), '阵容')}条件："
                    + "、".join(values or [str(row["condition_kind"])]),
                    row["comp_id"], row["source_response_id"],
                ))

            for row in connection.execute(
                "SELECT comp_id, section_type, ordinal, text, source_response_id "
                "FROM datatft_comp_strategy_sections WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchall():
                comp_name = comp_names.get(str(row["comp_id"]), "阵容")
                documents.append(_normalized_search_document(
                    "composition", row["comp_id"], comp_name, row["section_type"],
                    f"{comp_name}：{row['text']}", row["comp_id"], row["source_response_id"],
                    suffix=str(row["ordinal"]),
                ))

            for row in connection.execute(
                "SELECT c.canonical_id, c.entity_type, c.display_name, s.description, "
                "s.source_url FROM canonical_entities c LEFT JOIN static_entities s "
                "ON s.entity_id=c.canonical_id AND s.retrieved_at=("
                "SELECT MAX(s2.retrieved_at) FROM static_entities s2 "
                "WHERE s2.entity_id=c.canonical_id) "
                "WHERE c.game_patch=?",
                (game_patch,),
            ).fetchall():
                documents.append(_normalized_search_document(
                    row["entity_type"], row["canonical_id"], row["display_name"],
                    "definition",
                    f"{row['display_name']}：{row['description'] or '当前无详细定义'}",
                    None, None, source_url=row["source_url"],
                ))

        query_terms = _normalized_search_terms(cleaned_queries)
        ranked: list[dict[str, object]] = []
        for document in documents:
            haystack = _search_normalize(
                f"{document['display_name']} {document['semantic_content']}"
            )
            title = _search_normalize(str(document["display_name"]))
            matches = [term for term in query_terms if term and term in haystack]
            if not matches:
                continue
            title_hits = sum(term in title for term in matches)
            score = min(1.0, 0.25 + (0.5 * title_hits) + (0.25 * len(matches)))
            document.update({
                "source": (
                    "DataTFT normalized"
                    if document.get("comp_id") else "current static knowledge"
                ),
                "source_patch": game_patch,
                "patch_match": "exact",
                "url": document.pop("source_url", None) or str(snapshot["source_url"]),
                "snapshot_id": snapshot_id if document.get("comp_id") else None,
                "relevance_score": round(score, 4),
                "resolution_status": "RESOLVED",
            })
            ranked.append(document)
        ranked.sort(key=lambda item: (
            float(str(item["relevance_score"])),
            item["section"] in {"core", "tip", "composition"},
        ), reverse=True)
        if not ranked and not query_terms:
            fallbacks = [doc for doc in documents if doc["section"] == "composition"][:limit]
            for document in fallbacks:
                document.pop("source_url", None)
                document.update({
                    "source": "DataTFT normalized", "source_patch": game_patch,
                    "patch_match": "exact", "url": str(snapshot["source_url"]),
                    "snapshot_id": snapshot_id, "relevance_score": 0.1,
                    "resolution_status": "RESOLVED",
                })
            ranked = fallbacks
        selected: list[dict[str, object]] = []
        group_counts: dict[tuple[str, str], int] = {}
        for document in ranked:
            group = (str(document["section"]), str(document["display_name"]))
            maximum = 1 if document["section"] == "definition" else 2
            if group_counts.get(group, 0) >= maximum:
                continue
            group_counts[group] = group_counts.get(group, 0) + 1
            selected.append(document)
            if len(selected) >= limit:
                break
        return selected

    def _expand_normalized_relations_sync(
        self,
        snapshot_id: str,
        game_patch: str,
        seeds: list[dict[str, str]],
        candidate_limit: int,
    ) -> dict[str, object]:
        """Build reverse relations without making them a strategy ranking."""
        with self._connect() as connection:
            snapshot = connection.execute(
                "SELECT source_url FROM datatft_snapshots "
                "WHERE snapshot_id=? AND game_patch=?",
                (snapshot_id, game_patch),
            ).fetchone()
            if snapshot is None:
                return {"evidence": [], "expansions": [], "candidate_comps": []}
            source_url = str(snapshot["source_url"])
            candidate_signals: dict[str, dict[str, object]] = {}
            relation_documents: list[dict[str, object]] = []
            expansions: list[dict[str, object]] = []

            def add_candidate(
                row: sqlite3.Row,
                relation_kind: str,
                seed_name: str,
                signal: dict[str, object],
            ) -> None:
                comp_id = str(row["comp_id"])
                entry = candidate_signals.setdefault(comp_id, {
                    "comp_id": comp_id,
                    "name": str(row["name"]),
                    "tier": row["tier"],
                    "signals": [],
                    "trait_coverage_count": 0,
                })
                signals = entry["signals"]
                assert isinstance(signals, list)
                signals.append({
                    "relation_kind": relation_kind,
                    "seed": seed_name,
                    **signal,
                })
                if relation_kind == "trait_roster_overlap":
                    entry["trait_coverage_count"] = max(
                        int(str(entry["trait_coverage_count"])),
                        int(str(signal.get("overlap_count", 0))),
                    )

            for seed in seeds:
                entity_type = seed.get("entity_type", "")
                canonical_id = seed.get("canonical_id", "")
                display_name = seed.get("display_name", "")
                if not display_name:
                    continue
                if entity_type in {"trait", "emblem"}:
                    trait_name = (
                        display_name.removesuffix("纹章")
                        if entity_type == "emblem" else display_name
                    )
                    rows = connection.execute(
                        "SELECT c.comp_id,c.name,c.tier,"
                        "COUNT(DISTINCT b.board_index) overlap_count,"
                        "GROUP_CONCAT(DISTINCT b.display_name) holders "
                        "FROM datatft_comp_board_units b JOIN datatft_comps c "
                        "ON c.snapshot_id=b.snapshot_id AND c.comp_id=b.comp_id "
                        "JOIN static_entities s ON s.entity_id=b.canonical_champion_id "
                        "AND s.retrieved_at=(SELECT MAX(s2.retrieved_at) "
                        "FROM static_entities s2 WHERE s2.entity_id=b.canonical_champion_id) "
                        "WHERE b.snapshot_id=? AND b.board_phase='FINAL' AND s.traits LIKE ? "
                        "GROUP BY c.comp_id,c.name,c.tier",
                        (snapshot_id, f'%"{trait_name}"%'),
                    ).fetchall()
                    expansions.append({
                        "seed": display_name,
                        "relation": "emblem_to_trait" if entity_type == "emblem" else "trait",
                        "expanded_trait": trait_name,
                        "candidate_count": len(rows),
                    })
                    for row in rows:
                        add_candidate(row, "trait_roster_overlap", display_name, {
                            "trait": trait_name,
                            "overlap_count": int(row["overlap_count"]),
                            "champions": str(row["holders"] or "").split(","),
                        })
                elif entity_type in {"item", "artifact"}:
                    rows = connection.execute(
                        "SELECT c.comp_id,c.name,c.tier,p.holder_display_name,p.plan_index,"
                        "p.slot_index,p.item_display_name,i.item_category "
                        "FROM datatft_comp_item_plan_items p JOIN datatft_comps c "
                        "ON c.snapshot_id=p.snapshot_id AND c.comp_id=p.comp_id "
                        "JOIN datatft_item_identities i ON i.snapshot_id=p.snapshot_id "
                        "AND i.external_item_id=p.external_item_id "
                        "WHERE p.snapshot_id=? AND (p.canonical_item_id=? "
                        "OR p.item_display_name=? COLLATE NOCASE)",
                        (snapshot_id, canonical_id, display_name),
                    ).fetchall()
                    expansions.append({
                        "seed": display_name,
                        "relation": "item_to_holder_to_composition",
                        "candidate_count": len({str(row['comp_id']) for row in rows}),
                    })
                    for index, row in enumerate(rows):
                        add_candidate(row, "item_holder_plan", display_name, {
                            "holder": row["holder_display_name"],
                            "item_category": row["item_category"],
                            "plan_index": row["plan_index"],
                            "slot_index": row["slot_index"],
                        })
                        relation_documents.append(_relation_search_document(
                            snapshot_id, game_patch, source_url, row["comp_id"], row["name"],
                            "structured_relation", canonical_id or display_name,
                            f"{row['item_display_name']} → {row['holder_display_name']} → "
                            f"{row['name']}（装备方案{int(row['plan_index']) + 1}，"
                            f"槽位{int(row['slot_index']) + 1}）",
                            f"item:{canonical_id or display_name}:{index}",
                        ))
                elif entity_type == "augment":
                    rows = connection.execute(
                        "SELECT c.comp_id,c.name,c.tier,h.display_name,h.source_kind "
                        "FROM datatft_comp_hexes h JOIN datatft_comps c "
                        "ON c.snapshot_id=h.snapshot_id AND c.comp_id=h.comp_id "
                        "WHERE h.snapshot_id=? AND (h.canonical_augment_id=? "
                        "OR h.display_name=? COLLATE NOCASE)",
                        (snapshot_id, canonical_id, display_name),
                    ).fetchall()
                    expansions.append({
                        "seed": display_name,
                        "relation": "augment_to_composition",
                        "candidate_count": len({str(row['comp_id']) for row in rows}),
                    })
                    for index, row in enumerate(rows):
                        add_candidate(row, "augment_association", display_name, {
                            "source_kind": row["source_kind"],
                        })
                        relation_documents.append(_relation_search_document(
                            snapshot_id, game_patch, source_url, row["comp_id"], row["name"],
                            "structured_relation", canonical_id or display_name,
                            f"{row['display_name']} 与 {row['name']} 存在 "
                            f"{row['source_kind']} association；关联不代表排名",
                            f"augment:{canonical_id or display_name}:{index}",
                        ))

            tier_order = {"S": 0, "A": 1, "B": 2, "C": 3}
            ordered_candidates = sorted(
                candidate_signals.values(),
                key=lambda item: (
                    -int(str(item["trait_coverage_count"])),
                    tier_order.get(str(item["tier"]), 9),
                    str(item["name"]),
                ),
            )
            candidates: list[dict[str, object]] = []
            for seed_name in dict.fromkeys(seed.get("display_name", "") for seed in seeds):
                candidate = next((item for item in ordered_candidates
                                  if seed_name and _candidate_has_seed(item, seed_name)), None)
                if candidate is not None and candidate not in candidates:
                    candidates.append(candidate)
            candidates.extend(item for item in ordered_candidates if item not in candidates)
            candidates = candidates[:candidate_limit]
            selected_ids = {str(item["comp_id"]) for item in candidates}
            relation_documents = [item for item in relation_documents
                                  if str(item["comp_id"]) in selected_ids]
            for candidate in candidates:
                comp_id = str(candidate["comp_id"])
                relation_documents.append(_relation_search_document(
                    snapshot_id, game_patch, source_url, comp_id, candidate["name"],
                    "candidate_composition", comp_id,
                    {
                        "tier": candidate["tier"],
                        "candidate_source": candidate["signals"],
                        "retrieval_order_is_not_strength_ranking": True,
                    },
                    f"candidate:{comp_id}",
                ))
                sections = connection.execute(
                    "SELECT section_type,ordinal,text,source_response_id "
                    "FROM datatft_comp_strategy_sections WHERE snapshot_id=? AND comp_id=? "
                    "AND section_type IN ('starting_conditions','core','early_game',"
                    "'mid_game','late_game') ORDER BY CASE section_type "
                    "WHEN 'starting_conditions' THEN 0 WHEN 'core' THEN 1 "
                    "WHEN 'early_game' THEN 2 WHEN 'mid_game' THEN 3 ELSE 4 END, ordinal",
                    (snapshot_id, comp_id),
                ).fetchall()
                for row in sections:
                    relation_documents.append(_relation_search_document(
                        snapshot_id, game_patch, source_url, comp_id, candidate["name"],
                        str(row["section_type"]), comp_id, str(row["text"]),
                        f"strategy:{comp_id}:{row['section_type']}:{row['ordinal']}",
                        source_response_id=row["source_response_id"],
                    ))
        return {
            "evidence": relation_documents,
            "expansions": expansions,
            "candidate_comps": candidates,
        }

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
            token_clauses = " OR ".join("(c.text LIKE ? OR d.title LIKE ?)" for _ in tokens)
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
        return sorted(ranked, key=lambda item: float(str(item["relevance_score"])), reverse=True)[
            :limit
        ]

    def _rebuild_canonical_catalog_sync(
        self,
        game_patch: str,
        set_number: int,
        set_name: str,
        explicit_aliases: list[dict[str, str]],
        active_static_ids: list[str],
        snapshot_id: str | None = None,
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
            snapshot_filter = " AND s.snapshot_id=?" if snapshot_id else ""
            comp_params: list[object] = [game_patch]
            if snapshot_id:
                comp_params.append(snapshot_id)
            comps = connection.execute(
                "SELECT DISTINCT c.comp_id, c.name FROM datatft_comps c "
                "JOIN datatft_snapshots s USING(snapshot_id) "
                "WHERE s.game_patch=? AND s.status IN ('valid','stale')" + snapshot_filter,
                comp_params,
            ).fetchall()
            static_sql = "SELECT entity_id, entity_type, name FROM static_entities"
            static_params: list[object] = []
            if active_static_ids:
                marks = ",".join("?" for _ in active_static_ids)
                static_sql += (
                    f" WHERE entity_id IN ({marks})"
                    " OR (entity_id LIKE 'DA_Artifact_%' AND tags LIKE '%{44ace175}%')"
                    " OR (entity_id LIKE 'DA_%' AND tags LIKE '%{6ef5c598}%')"
                )
                static_params.extend(active_static_ids)
            else:
                # Compatibility bootstrap for databases synchronized before active-set metadata.
                from tft_agent.app.aliases import AliasRegistry

                registry_ids = sorted(
                    {
                        entry.canonical_id
                        for entry in AliasRegistry.load_default().entries
                        if entry.canonical_id is not None
                        and (entry.scope == "global" or entry.set_number == set_number)
                    }
                )
                marks = ",".join("?" for _ in registry_ids)
                static_sql += (
                    " WHERE entity_id LIKE ? OR entity_id LIKE 'DA_Component_%'"
                    " OR (entity_id LIKE 'DA_Artifact_%' AND tags LIKE '%{44ace175}%')"
                    " OR (entity_id LIKE 'DA_%' AND tags LIKE '%{6ef5c598}%')"
                    + (f" OR entity_id IN ({marks})" if marks else "")
                )
                static_params.append(f"DA%{set_number}%")
                static_params.extend(registry_ids)
            static = connection.execute(static_sql, static_params).fetchall()
            entities = [
                (str(row["comp_id"]), "composition", str(row["name"]), "datatft") for row in comps
            ] + [
                (
                    str(row["entity_id"]),
                    str(row["entity_type"]),
                    str(row["name"]),
                    "communitydragon",
                )
                for row in static
                if row["name"] and "_Name" not in str(row["name"])
            ]
            connection.executemany(
                "INSERT INTO canonical_entities VALUES (?,?,?,?,?,?,?)",
                [
                    (identifier, kind, name, game_patch, set_number, set_name, source)
                    for identifier, kind, name, source in entities
                ],
            )
            from tft_agent.app.aliases import AliasRegistry

            registry = AliasRegistry.load_default()
            materialized = registry.materialize(entities, set_number, game_patch)
            aliases = [
                (item.alias, item.canonical_id, item.entity_type, item.alias_kind)
                for item in materialized
            ]
            connection.executemany(
                "INSERT OR IGNORE INTO entity_aliases VALUES (?,?,?,?,?,?)",
                [
                    (alias, identifier, kind, game_patch, set_number, alias_kind)
                    for alias, identifier, kind, alias_kind in aliases
                ],
            )

    def _canonical_catalog_sync(self, game_patch: str, set_number: int) -> list[dict[str, Any]]:
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
            item = grouped.setdefault(
                str(row["canonical_id"]),
                {
                    "canonical_id": row["canonical_id"],
                    "entity_type": row["entity_type"],
                    "display_name": row["display_name"],
                    "game_patch": row["game_patch"],
                    "set_number": row["set_number"],
                    "set_name": row["set_name"],
                    "source": row["source"],
                    "aliases": [],
                },
            )
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
        return [
            StaticEntityRecord(
                entity_id=row["entity_id"],
                entity_type=row["entity_type"],
                name=row["name"],
                description=row["description"],
                traits=json.loads(row["traits"]),
                components=json.loads(row["components"]),
                tags=json.loads(row["tags"]),
                data_version=row["data_version"],
                source=row["source"],
                source_url=row["source_url"],
                retrieved_at=row["retrieved_at"],
            )
            for row in rows
        ]

    def _find_comp_by_id_sync(self, canonical_id: str, game_patch: str) -> CompStatsRecord | None:
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
