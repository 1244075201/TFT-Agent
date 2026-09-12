"""Single active-patch resolution and patch identity helpers."""

from __future__ import annotations

import re

from tft_agent.core.config import Settings
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.schemas import ActivePatchContext


def patch_family(game_patch: str) -> str:
    """Return the numeric patch family while preserving exact hotfix identities elsewhere."""

    value = game_patch.strip()
    match = re.fullmatch(r"(\d+\.\d+)[a-z]*", value, re.IGNORECASE)
    if match is None:
        raise ValueError(f"invalid game patch: {game_patch!r}")
    return match.group(1)


class PatchService:
    """Resolve the database registry as runtime truth, with Settings used only to bootstrap."""

    def __init__(self, repository: SQLiteGameDataRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    async def active_context(self) -> ActivePatchContext:
        context = await self._repository.active_patch_context()
        if context is None:
            context = await self._repository.bootstrap_patch_registry(
                self._settings.current_game_patch,
                self._settings.current_set_number,
                self._settings.current_set_name,
            )
        if context is None:
            context = ActivePatchContext(
                game_patch=self._settings.current_game_patch,
                patch_family=patch_family(self._settings.current_game_patch),
                set_number=self._settings.current_set_number,
                set_name=self._settings.current_set_name,
                active_snapshot_id="",
            )
        return context

    async def context_for(self, game_patch: str) -> ActivePatchContext | None:
        return await self._repository.patch_context(game_patch)
