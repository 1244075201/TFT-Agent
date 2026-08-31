"""Structured game-data access contract."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from tft_agent.tft_mcp_server.schemas import ChampionRecord, ItemInfo, ShopOdds, TraitInfo


class GameDataRepository(Protocol):
    """Read exact, patch-aware TFT records without strategic interpretation."""

    async def replace_champions(
        self,
        records: Sequence[ChampionRecord],
        data_version: str,
        retrieved_at: datetime,
    ) -> None: ...

    async def find_champions(
        self,
        name: str,
        champion_id: str | None = None,
        data_version: str | None = None,
    ) -> list[ChampionRecord]: ...

    async def get_item(self, name: str, patch: str | None) -> ItemInfo | None: ...

    async def get_trait(self, name: str, patch: str | None) -> TraitInfo | None: ...

    async def get_shop_odds(self, level: int, patch: str | None) -> ShopOdds | None: ...
