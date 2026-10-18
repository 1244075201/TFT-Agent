"""Persistent server truth and refresh state, separate from source snapshots."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ServerVersion(BaseModel):
    game_patch: str | None = None
    region: str = "cn"
    mode: str = "standard"
    effective_at: datetime | None = None
    source_url: str | None = None
    status: str = "unverified"
    checked_at: datetime
    reason: str = "版本状态待核验"


_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


@asynccontextmanager
async def database_lease(path: Path) -> AsyncIterator[None]:
    """Serialize publication and answers across the desktop's independent loops.

    Nonblocking acquisition is cancellation-safe; no stranded executor waiter.
    """
    with _guard:
        lock = _locks.setdefault(str(path.resolve()).casefold(), threading.Lock())
    while not lock.acquire(blocking=False):
        await asyncio.sleep(0.05)
    try:
        yield
    finally:
        lock.release()


class UpdateState:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS application_updates "
                       "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            with db:
                yield db
        finally:
            db.close()

    def get(self, key: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT value FROM application_updates WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else {}

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO application_updates VALUES (?,?)",
                       (key, json.dumps(value, ensure_ascii=False)))

    def server(self) -> ServerVersion | None:
        value = self.get("server_confirmed")
        return ServerVersion.model_validate(value) if value else None

    def record_server(self, version: ServerVersion) -> None:
        self.put("server_check", version.model_dump(mode="json"))
        if (version.status != "confirmed" or not version.game_patch
                or not version.effective_at or version.effective_at > datetime.now(UTC)):
            return
        previous = self.server()
        if previous and patch_order(version.game_patch) < patch_order(previous.game_patch or ""):
            return
        self.put("server_confirmed", version.model_dump(mode="json"))


def patch_order(patch: str) -> tuple[int, int, str]:
    import re
    match = re.fullmatch(r"(\d+)\.(\d+)([a-z]?)", patch.lower())
    if not match:
        raise ValueError("invalid patch identity")
    return int(match[1]), int(match[2]), match[3]
