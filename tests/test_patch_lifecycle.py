import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tft_agent.app.entity_resolution import CanonicalEntityResolver
from tft_agent.core.config import Settings
from tft_agent.core.patches import PatchService, patch_family
from tft_agent.ingestion.sources.datatft import DataTFTSource, RawHTTPResponse
from tft_agent.ingestion.sync_datatft import sync
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.schemas import (
    CompStatsRecord,
    DataSnapshotMetadata,
    DataTFTSourceResponse,
)


def _snapshot(patch: str, snapshot_id: str, content_hash: str) -> DataSnapshotMetadata:
    return DataSnapshotMetadata(
        snapshot_id=snapshot_id,
        server="cn",
        game_patch=patch,
        source_url="https://example.test/comps",
        retrieved_at=datetime.now(UTC),
        content_hash=content_hash,
    )


def _record(metadata: DataSnapshotMetadata, comp_id: str, name: str) -> CompStatsRecord:
    return CompStatsRecord(comp_id=comp_id, name=name, tier="S", snapshot=metadata)


async def _run(repository: SQLiteGameDataRepository, sync_id: str, patch: str) -> None:
    await repository.create_sync_run(
        sync_id, patch, "cn", 18, "Enchanted Wilds", datetime.now(UTC)
    )


@pytest.mark.asyncio
async def test_patch_hash_uniqueness_is_scoped_to_exact_patch(tmp_path: Path) -> None:
    repository = SQLiteGameDataRepository(tmp_path / "tft.sqlite3")
    await repository.initialize()
    first = _snapshot("18.2", "snapshot-18.2", "same-hash")
    hotfix = _snapshot("18.2b", "snapshot-18.2b", "same-hash")
    await _run(repository, "run-18.2", "18.2")
    await _run(repository, "run-18.2b", "18.2b")

    assert await repository.store_candidate_snapshot(
        first, [_record(first, "one", "First")], "{}", "run-18.2", {},
        "CANONICAL_INPUT_ONLY"
    ) == ("snapshot-18.2", True)
    assert await repository.store_candidate_snapshot(
        hotfix, [_record(hotfix, "one", "First")], "{}", "run-18.2b", {},
        "CANONICAL_INPUT_ONLY"
    ) == ("snapshot-18.2b", True)

    with sqlite3.connect(tmp_path / "tft.sqlite3") as connection:
        assert connection.execute(
            "SELECT game_patch,snapshot_id FROM datatft_snapshots ORDER BY game_patch"
        ).fetchall() == [("18.2", "snapshot-18.2"), ("18.2b", "snapshot-18.2b")]


@pytest.mark.asyncio
async def test_identical_resync_preserves_snapshot_origin_lineage(tmp_path: Path) -> None:
    repository = SQLiteGameDataRepository(tmp_path / "tft.sqlite3")
    await repository.initialize()
    origin = _snapshot("18.2", "origin-snapshot", "same-hash")
    repeat = _snapshot("18.2", "repeat-snapshot", "same-hash")
    await _run(repository, "origin-run", "18.2")
    await _run(repository, "repeat-run", "18.2")
    await repository.store_source_responses([
        DataTFTSourceResponse(
            response_id="origin-response", sync_id="origin-run", game_patch="18.2",
            endpoint_type="COMPOSITION_DETAIL", request_url="https://example/origin",
            request_key="one", retrieved_at=datetime.now(UTC), http_status=200,
            content_hash="origin-response-hash", response_body="{}",
        ),
        DataTFTSourceResponse(
            response_id="repeat-response", sync_id="repeat-run", game_patch="18.2",
            endpoint_type="COMPOSITION_DETAIL", request_url="https://example/repeat",
            request_key="one", retrieved_at=datetime.now(UTC), http_status=200,
            content_hash="repeat-response-hash", response_body="{}",
        ),
    ])
    await repository.store_candidate_snapshot(
        origin, [_record(origin, "one", "First")], "{}", "origin-run",
        {"one": "origin-response"}, "CANONICAL_INPUT_ONLY",
    )
    assert await repository.store_candidate_snapshot(
        repeat, [_record(repeat, "one", "First")], "{}", "repeat-run",
        {"one": "repeat-response"}, "CANONICAL_INPUT_ONLY",
    ) == ("origin-snapshot", False)
    await repository.update_sync_run(
        "repeat-run", "validated", detected_patch="18.2", snapshot_id="origin-snapshot"
    )

    with sqlite3.connect(tmp_path / "tft.sqlite3") as connection:
        assert connection.execute(
            "SELECT sync_id FROM datatft_snapshots WHERE snapshot_id='origin-snapshot'"
        ).fetchone() == ("origin-run",)
        assert connection.execute(
            "SELECT source_response_id FROM datatft_comps "
            "WHERE snapshot_id='origin-snapshot' AND comp_id='one'"
        ).fetchone() == ("origin-response",)
        assert connection.execute(
            "SELECT canonical_snapshot_id FROM datatft_sync_runs WHERE sync_id='repeat-run'"
        ).fetchone() == ("origin-snapshot",)
        assert connection.execute(
            "SELECT COUNT(*) FROM datatft_source_responses WHERE sync_id='repeat-run'"
        ).fetchone() == (1,)


@pytest.mark.asyncio
async def test_same_patch_changed_hash_creates_new_snapshot(tmp_path: Path) -> None:
    repository = SQLiteGameDataRepository(tmp_path / "tft.sqlite3")
    await repository.initialize()
    first = _snapshot("18.2", "first-snapshot", "first-hash")
    changed = _snapshot("18.2", "changed-snapshot", "changed-hash")
    await _run(repository, "first-run", "18.2")
    await _run(repository, "changed-run", "18.2")
    assert (await repository.store_candidate_snapshot(
        first, [_record(first, "one", "First")], "{}", "first-run", {},
        "CANONICAL_INPUT_ONLY",
    ))[1]
    assert (await repository.store_candidate_snapshot(
        changed, [_record(changed, "one", "First")], "{\"changed\":true}",
        "changed-run", {}, "CANONICAL_INPUT_ONLY",
    ))[1]
    with sqlite3.connect(tmp_path / "tft.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM datatft_snapshots WHERE game_patch='18.2'"
        ).fetchone() == (2,)


@pytest.mark.asyncio
async def test_candidate_catalog_is_hidden_until_atomic_activation(tmp_path: Path) -> None:
    database = tmp_path / "tft.sqlite3"
    repository = SQLiteGameDataRepository(database)
    await repository.initialize()
    old = _snapshot("18.1d", "old-snapshot", "old-hash")
    await repository.replace_datatft_snapshot(old, [_record(old, "old", "Old Comp")], "{}")
    settings = Settings(
        database_path=database, current_game_patch="18.1d",
        current_set_number=18, current_set_name="Enchanted Wilds",
    )
    active = await PatchService(repository, settings).active_context()
    await repository.rebuild_canonical_catalog("18.1d", 18, "Enchanted Wilds", (), (),
                                               "old-snapshot")

    candidate = _snapshot("18.2", "new-snapshot", "new-hash")
    await _run(repository, "new-run", "18.2")
    await repository.register_candidate_patch(
        "18.2", 18, "Enchanted Wilds", "datatft", datetime.now(UTC)
    )
    await repository.store_candidate_snapshot(
        candidate, [_record(candidate, "new", "New Comp")], "{}", "new-run", {},
        "CANONICAL_INPUT_ONLY"
    )
    await repository.rebuild_canonical_catalog(
        "18.2", 18, "Enchanted Wilds", (), (), "new-snapshot"
    )

    resolver = CanonicalEntityResolver(repository, active, 0.9)
    assert not (await resolver.resolve("New Comp")).entities
    assert [item.name for item in await repository.search_comps(None, "18.1d")] == ["Old Comp"]

    activated = await repository.activate_snapshot(
        "18.2", 18, "Enchanted Wilds", "new-snapshot", "new-run", []
    )
    assert activated.game_patch == "18.2"
    assert (await CanonicalEntityResolver(repository, activated, 0.9).resolve("New Comp")).entities
    assert (await repository.active_patch_context()).game_patch == "18.2"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_failed_run_does_not_replace_active_patch(tmp_path: Path) -> None:
    database = tmp_path / "tft.sqlite3"
    repository = SQLiteGameDataRepository(database)
    await repository.initialize()
    old = _snapshot("18.1d", "old-snapshot", "old-hash")
    await repository.replace_datatft_snapshot(old, [_record(old, "old", "Old Comp")], "{}")
    settings = Settings(database_path=database, current_game_patch="18.1d")
    await PatchService(repository, settings).active_context()
    await _run(repository, "failed-run", "18.2")
    await repository.update_sync_run(
        "failed-run", "failed", detected_patch="18.2",
        failure_stage="fetch", failure_code="HTTPError", failure_message="denied",
    )
    active = await repository.active_patch_context()
    assert active is not None and active.game_patch == "18.1d"


@pytest.mark.asyncio
async def test_activation_failure_rolls_back_previous_active(tmp_path: Path) -> None:
    database = tmp_path / "tft.sqlite3"
    repository = SQLiteGameDataRepository(database)
    await repository.initialize()
    old = _snapshot("18.1d", "old-snapshot", "old-hash")
    await repository.replace_datatft_snapshot(old, [_record(old, "old", "Old Comp")], "{}")
    settings = Settings(database_path=database, current_game_patch="18.1d")
    await PatchService(repository, settings).active_context()
    candidate = _snapshot("18.2", "candidate-snapshot", "candidate-hash")
    await _run(repository, "candidate-run", "18.2")
    await repository.register_candidate_patch(
        "18.2", 18, "Enchanted Wilds", "datatft", datetime.now(UTC)
    )
    await repository.store_candidate_snapshot(
        candidate, [_record(candidate, "new", "New Comp")], "{}", "candidate-run", {},
        "CANONICAL_INPUT_ONLY",
    )
    await repository.rebuild_canonical_catalog(
        "18.2", 18, "Enchanted Wilds", (), (), "candidate-snapshot"
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TRIGGER reject_candidate_activation BEFORE UPDATE OF status ON game_patches
            WHEN OLD.game_patch='18.2' AND NEW.status='active'
            BEGIN SELECT RAISE(ABORT, 'injected activation failure'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected activation failure"):
        await repository.activate_snapshot(
            "18.2", 18, "Enchanted Wilds", "candidate-snapshot", "candidate-run", []
        )
    active = await repository.active_patch_context()
    assert active is not None and active.game_patch == "18.1d"


@pytest.mark.asyncio
async def test_legacy_bootstrap_uses_settings_without_patch_hardcode(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    database = tmp_path / "legacy.sqlite3"
    shutil.copy2(source, database)
    repository = SQLiteGameDataRepository(database)
    await repository.initialize()
    settings = Settings(
        database_path=database, current_game_patch="18.1d",
        current_set_number=18, current_set_name="Enchanted Wilds",
    )
    active = await PatchService(repository, settings).active_context()
    assert active.game_patch == settings.current_game_patch
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT game_patch,status FROM game_patches ORDER BY game_patch"
        ).fetchall() == [("18.1", "historical"), ("18.1d", "active")]


def test_patch_family_preserves_hotfix_identity() -> None:
    assert patch_family("18.2") == "18.2"
    assert patch_family("18.2b") == "18.2"
    with pytest.raises(ValueError):
        patch_family("latest")


@pytest.mark.asyncio
async def test_sync_persists_list_and_detail_raw_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "tft.sqlite3"
    repository = SQLiteGameDataRepository(database)
    await repository.initialize()
    old = _snapshot("18.1d", "old-snapshot", "old-hash")
    await repository.replace_datatft_snapshot(old, [_record(old, "old", "Old Comp")], "{}")
    settings = Settings(database_path=database, current_game_patch="18.1d")
    await PatchService(repository, settings).active_context()
    now = datetime.now(UTC)

    def fake_fetch(
        _self: DataTFTSource, responses: list[RawHTTPResponse],
        expected_game_patch: str | None, request_interval_seconds: float,
    ) -> tuple[str, None]:
        assert expected_game_patch == "18.2"
        responses.extend([
            RawHTTPResponse("VERSION_PAGE", "https://example/version", None, now, 200,
                            "version-etag", "text/html", "Ver: 18.2"),
            RawHTTPResponse("COMPOSITION_LIST", "https://example/list", None, now, 200,
                            "list-etag", "application/json", '{"list":["new"]}'),
            RawHTTPResponse("COMPOSITION_DETAIL", "https://example/detail", "new", now, 200,
                            "detail-etag", "application/json", '{"id":"new"}'),
        ])
        return json.dumps({
            "game_patch": "18.2",
            "comps": [{
                "id": "new", "name": "New Comp", "tier": "S",
                "core_units": [], "recommended_items": [], "augments": [],
                "level_plan": [],
            }],
        }), None

    monkeypatch.setattr(DataTFTSource, "fetch_current_with_raw", fake_fetch)
    snapshot_id, count = await sync(None, database, settings, "18.2", 18, "Enchanted Wilds")
    assert count == 1

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        snapshot = connection.execute(
            "SELECT sync_id,raw_source_status FROM datatft_snapshots WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        assert snapshot is not None and snapshot["raw_source_status"] == "AVAILABLE"
        responses = connection.execute(
            "SELECT endpoint_type,request_key,response_body FROM datatft_source_responses "
            "WHERE sync_id=? ORDER BY response_id", (snapshot["sync_id"],),
        ).fetchall()
        assert {row["endpoint_type"] for row in responses} == {
            "VERSION_PAGE", "COMPOSITION_LIST", "COMPOSITION_DETAIL",
        }
        detail_id = connection.execute(
            "SELECT source_response_id FROM datatft_comps WHERE snapshot_id=? AND comp_id='new'",
            (snapshot_id,),
        ).fetchone()[0]
        assert detail_id.endswith(":response:2")
