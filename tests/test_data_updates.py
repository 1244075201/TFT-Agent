import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tft_agent.app.data_updates import DataUpdateCoordinator
from tft_agent.app.live_knowledge import LiveKnowledge, WebPage
from tft_agent.core.config import Settings
from tft_agent.core.patches import PatchService
from tft_agent.core.update_state import ServerVersion, UpdateState, database_lease
from tft_agent.ingestion.server_version import OfficialServerVersionResolver, parse_announcement
from tft_agent.ingestion.sources.datatft import _extract_current_patch
from tft_agent.ingestion.sync_datatft import sync
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


def confirmed(patch: str = "18.2b") -> ServerVersion:
    return ServerVersion(game_patch=patch, status="confirmed", checked_at=datetime.now(UTC),
                         effective_at=datetime.now(UTC) - timedelta(days=1),
                         source_url="https://lol.qq.com/announcement")


def test_resolver_preserves_unconfirmed_reason(tmp_path, monkeypatch) -> None:
    url = "https://lol.qq.com/announcement"
    page = WebPage(title="云顶公告", body="公告正文", url=url)
    resolver = OfficialServerVersionResolver(
        Settings(database_path=tmp_path / "versions.sqlite3", server_version_urls=[url]),
        search=SimpleNamespace(search=lambda query: []),
        fetcher=SimpleNamespace(fetch=lambda address: page),
    )
    monkeypatch.setattr("tft_agent.ingestion.server_version.parse_announcement",
                        lambda page, now: ServerVersion(
                            checked_at=now, source_url=url, reason="停机更新尚无完成依据",
                        ))
    result = resolver.resolve()
    assert result.status == "unverified"
    assert result.reason == "已自动检查官方公告；停机更新尚无完成依据"
    assert result.source_url == url


@pytest.mark.parametrize("title,body,url,ok", [
    ("云顶之弈18.2b版本不停机更新", "已于2026年9月16日10点进行不停机更新。",
     "https://lol.qq.com/a", True),
    ("云顶之弈18.2b版本不停机更新", "将于2027年9月16日10点进行不停机更新。",
     "https://lol.qq.com/a", False),
    ("云顶之弈18.2b版本预告", "已于2026年9月16日10点更新。",
     "https://lol.qq.com/a", False),
    ("云顶之弈18.2b版本更新", "已于2026年9月16日10点更新。",
     "https://tftacademy.com/a", False),
    ("云顶之弈9月16日不停机更新", "已于2026年9月16日10点进行不停机更新。",
     "https://lol.qq.com/a", False),
    ("金铲铲云顶之弈18.2b版本更新", "已于2026年9月16日10点更新。",
     "https://lol.qq.com/a", False),
])
def test_official_version_requires_region_patch_and_effective_time(
    title: str, body: str, url: str, ok: bool,
) -> None:
    result = parse_announcement(WebPage(url, title, body), datetime(2026, 9, 22, tzinfo=UTC))
    assert (result.status == "confirmed") is ok


def test_scheduled_datatft_version_does_not_activate_future_option() -> None:
    template = ('const d=[{value:"18.2"},{value:"18.1d"}],$="%s",'
                'z=()=>Date.now()<t.getTime(),I=()=>{return[n,t,...e]}')
    assert _extract_current_patch(template % "2099-09-02 17:46:15") == "18.1d"
    assert _extract_current_patch(template % "2020-09-02 17:46:15") == "18.2"


def test_unverified_check_never_downgrades_confirmed_server(tmp_path: Path) -> None:
    state = UpdateState(tmp_path / "db.sqlite3")
    state.record_server(confirmed())
    state.record_server(confirmed("18.2"))
    state.record_server(ServerVersion(checked_at=datetime.now(UTC)))
    assert state.server() and state.server().game_patch == "18.2b"
    assert state.get("server_check")["status"] == "unverified"


@pytest.mark.asyncio
async def test_server_upgrade_excludes_old_snapshot_without_deleting_it(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, database_path=tmp_path / "db.sqlite3")
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"game_patch": "18.2", "comps": [
        {"id": "one", "name": "旧阵容", "tier": "S"},
    ]}), encoding="utf-8")
    await sync(path, settings.database_path, settings)
    UpdateState(settings.database_path).record_server(confirmed())
    repo = SQLiteGameDataRepository(settings.database_path)
    context = await PatchService(repo, settings).active_context()
    assert context.game_patch == "18.2b" and not context.active_snapshot_id
    assert not await repo.search_comps(None, context.game_patch)
    assert await repo.search_comps(None, "18.2")


@pytest.mark.asyncio
async def test_same_patch_publication_failure_preserves_catalog_and_snapshot(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, database_path=tmp_path / "db.sqlite3")
    path = tmp_path / "snapshot.json"
    def write(name: str) -> None:
        path.write_text(json.dumps({"game_patch": "18.2", "comps": [
            {"id": "one", "name": name, "tier": "S"},
        ]}), encoding="utf-8")
    write("original")
    first, _ = await sync(path, settings.database_path, settings)
    with sqlite3.connect(settings.database_path) as db:
        db.execute("CREATE TRIGGER reject_publication BEFORE UPDATE OF active_snapshot_id "
                   "ON game_patches WHEN NEW.active_snapshot_id != OLD.active_snapshot_id "
                   "BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    write("replacement")
    with pytest.raises(sqlite3.IntegrityError):
        await sync(path, settings.database_path, settings)
    with sqlite3.connect(settings.database_path) as db:
        assert db.execute("SELECT active_snapshot_id FROM game_patches WHERE status='active'"
                          ).fetchone()[0] == first
        assert db.execute("SELECT display_name FROM canonical_entities WHERE canonical_id='one'"
                          ).fetchone()[0] == "original"


@pytest.mark.asyncio
async def test_refresh_deduplicates_and_preserves_success_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, database_path=tmp_path / "db.sqlite3")
    knowledge = LiveKnowledge(settings)
    coordinator = DataUpdateCoordinator(settings, knowledge)
    monkeypatch.setattr(coordinator, "resolver", SimpleNamespace(resolve=lambda: confirmed()))
    monkeypatch.setattr(knowledge, "refresh_due", lambda patch: None)
    calls = []
    async def fake_sync(*args, **kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0.02)
        if len(calls) > 1:
            raise ValueError("upstream mismatch")
        return "snapshot", 30
    monkeypatch.setattr("tft_agent.app.data_updates.sync", fake_sync)
    await asyncio.gather(coordinator.refresh(), coordinator.refresh())
    await coordinator.refresh()
    assert len(calls) == 1 and calls[0]["requested_patch"] == "18.2b"
    success = coordinator.state.get("refresh")["last_success"]
    await coordinator.refresh(force=True)
    assert coordinator.state.get("refresh")["last_success"] == success
    assert coordinator.state.get("refresh")["status"] == "failed"
    assert coordinator.state.server().game_patch == "18.2b"


@pytest.mark.asyncio
async def test_cancelled_lease_waiter_does_not_strand_lock(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    async def waiter():
        async with database_lease(path):
            pytest.fail("should still be waiting")
    async with database_lease(path):
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    async with database_lease(path):
        pass


@pytest.mark.asyncio
async def test_tier_request_limits_after_source_rating_sort(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, database_path=tmp_path / "db.sqlite3")
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"game_patch": "18.2", "comps": [
        {"id": "low", "name": "A low", "tier": "C"},
        {"id": "high", "name": "Z high", "tier": "S"},
    ]}), encoding="utf-8")
    snapshot, _ = await sync(path, settings.database_path, settings)
    repo = SQLiteGameDataRepository(settings.database_path)
    result = await repo.discover_compositions(
        snapshot, "18.2", main_unit_cost_min=None, main_unit_cost_max=None,
        cost_quantifier="ANY", strategy_terms=[], strategy_exclusions=[], item_terms=[],
        require_strategy_match=False, requested_fields=["tier"], limit=1,
    )
    assert result["candidate_comps"][0]["comp_id"] == "high"


@pytest.mark.asyncio
async def test_unknown_server_only_refreshes_existing_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, database_path=tmp_path / "db.sqlite3")
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"game_patch": "18.2", "comps": [
        {"id": "one", "name": "current", "tier": "S"},
    ]}), encoding="utf-8")
    await sync(path, settings.database_path, settings)
    knowledge = LiveKnowledge(settings)
    coordinator = DataUpdateCoordinator(settings, knowledge)
    monkeypatch.setattr(coordinator, "resolver", SimpleNamespace(
        resolve=lambda: ServerVersion(checked_at=datetime.now(UTC)),
    ))
    monkeypatch.setattr(knowledge, "refresh_due", lambda patch: None)
    calls = []
    async def fake_sync(*args, **kwargs):
        calls.append(kwargs)
        return "same-patch", 1
    monkeypatch.setattr("tft_agent.app.data_updates.sync", fake_sync)
    await coordinator.refresh()
    assert calls[0] == {"requested_patch": "18.2", "activate": True}
    assert coordinator.state.server() is None
    assert coordinator.state.get("refresh")["status"] == "awaiting_version"
