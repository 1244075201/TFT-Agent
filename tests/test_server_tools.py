from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from tft_agent.tft_mcp_server.schemas import (
    ChampionImage,
    ChampionRecord,
    ItemInfo,
    ShopOdds,
    TraitInfo,
)
from tft_agent.tft_mcp_server.tools.game_data import GameDataTools


def champion(champion_id: str, name: str = "Fixture Champion") -> ChampionRecord:
    return ChampionRecord(
        champion_id=champion_id,
        name=name,
        cost=1,
        image=ChampionImage(full="x.png", sprite="x.png", group="champion", x=0, y=0, w=1, h=1),
        data_version="fixture-version",
        source="test-fixture",
        retrieved_at=datetime.now(UTC),
    )


class FakeGameDataRepository:
    def __init__(self, candidates: list[ChampionRecord]) -> None:
        self.candidates = candidates

    async def replace_champions(
        self,
        records: Sequence[ChampionRecord],
        data_version: str,
        retrieved_at: datetime,
    ) -> None:
        raise AssertionError("not used")

    async def find_champions(
        self,
        name: str,
        champion_id: str | None = None,
        data_version: str | None = None,
    ) -> list[ChampionRecord]:
        return self.candidates

    async def get_item(self, name: str, patch: str | None) -> ItemInfo | None:
        return None

    async def get_trait(self, name: str, patch: str | None) -> TraitInfo | None:
        return None

    async def get_shop_odds(self, level: int, patch: str | None) -> ShopOdds | None:
        return None


@pytest.mark.asyncio
async def test_champion_tool_returns_unique_repository_record() -> None:
    result = await GameDataTools(FakeGameDataRepository([champion("fixture-1")])).lookup_champion(
        "call-1", "Fixture Champion"
    )

    assert result.success is True
    assert isinstance(result.data, dict)
    assert result.data["champion_id"] == "fixture-1"
    assert result.metadata.data_version == "fixture-version"


@pytest.mark.asyncio
async def test_champion_tool_returns_structured_not_found() -> None:
    result = await GameDataTools(FakeGameDataRepository([])).lookup_champion("call-2", "Missing")

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "CHAMPION_NOT_FOUND"


@pytest.mark.asyncio
async def test_champion_tool_preserves_ambiguity() -> None:
    result = await GameDataTools(
        FakeGameDataRepository([champion("fixture-1"), champion("fixture-2")])
    ).lookup_champion("call-3", "Fixture Champion")

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "AMBIGUOUS_CHAMPION"
    assert isinstance(result.data, dict)
    assert len(result.data["candidates"]) == 2
