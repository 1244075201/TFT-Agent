"""One application-owned coordinator for version, snapshots and guide refresh."""
from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta

from tft_agent.app.live_knowledge import LiveKnowledge
from tft_agent.core.config import Settings
from tft_agent.core.update_state import UpdateState, database_lease, patch_order
from tft_agent.ingestion.server_version import OfficialServerVersionResolver
from tft_agent.ingestion.sync_datatft import sync
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


class DataUpdateCoordinator:
    def __init__(self, settings: Settings, knowledge: LiveKnowledge) -> None:
        self.settings = settings
        self.knowledge = knowledge
        self.state = UpdateState(settings.database_path)
        self.resolver = OfficialServerVersionResolver(settings)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    async def refresh(self, force: bool = False) -> None:
        if not self.settings.data_refresh_enabled or not self.lock.acquire(blocking=False):
            return
        try:
            now = datetime.now(UTC)
            status = self.state.get("refresh")
            due = status.get("next_attempt")
            if not force and due and now < datetime.fromisoformat(due):
                return
            status.update(status="running", last_attempt=now.isoformat())
            self.state.put("refresh", status)
            result = await asyncio.to_thread(self.resolver.resolve)
            async with database_lease(self.settings.database_path):
                repo = SQLiteGameDataRepository(self.settings.database_path)
                await repo.initialize()
                active = await repo.active_patch_context()
                if (result.game_patch and active
                        and patch_order(result.game_patch) < patch_order(active.game_patch)):
                    result = result.model_copy(update={
                        "status": "unverified", "reason": "检索到旧公告，当前版本待核验",
                    })
                self.state.record_server(result)
            server = self.state.server()
            # An unverified server check may refresh the already active source
            # version, but can never promote a different source version.
            patch = server.game_patch if server else (active.game_patch if active else None)
            try:
                snapshot, count = await sync(
                    None, self.settings.database_path, self.settings,
                    requested_patch=patch, activate=bool(patch),
                )
                status.update(status="ok" if server else "awaiting_version",
                              last_success=now.isoformat(), snapshot_id=snapshot,
                              composition_count=count, failures=0, error=None)
                delay = timedelta(hours=self.settings.data_refresh_hours)
            except Exception as exc:
                failures = int(status.get("failures", 0)) + 1
                # Do not persist provider text or credentials in the status channel.
                reason = ("来源补丁与目标补丁不一致" if "does not match" in str(exc)
                          else "数据校验未通过" if isinstance(exc, ValueError)
                          else "网络或数据源访问失败")
                status.update(status="failed", failures=failures,
                              error=f"DataTFT 同步失败：{reason}；原资料保留")
                delay = timedelta(minutes=min(60 * 2 ** min(failures - 1, 4), 24 * 60))
            if patch and not self.stop_event.is_set():
                try:
                    await asyncio.to_thread(self.knowledge.refresh_due, patch)
                except (OSError, ValueError):
                    status["guide_error"] = "补充攻略刷新失败，保留已验证资料"
            status["version_status"] = result.status
            status["next_attempt"] = (datetime.now(UTC) + delay).isoformat()
            self.state.put("refresh", status)
        finally:
            self.lock.release()

    def start(self) -> None:
        if self.thread or not self.settings.data_refresh_enabled:
            return
        self.stop_event.clear()

        def run() -> None:
            while not self.stop_event.is_set():
                try:
                    asyncio.run(self.refresh())
                except Exception as exc:
                    self.state.put("refresh", {
                        "status": "failed", "error": type(exc).__name__,
                        "next_attempt": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                    })
                self.stop_event.wait(3600)

        self.thread = threading.Thread(target=run, name="tft-data-refresh", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.knowledge.stop()
