from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from tft_agent.core.schemas import ToolErrorResolution
from tft_agent.tft_mcp_server.schemas import (
    ChampionImage,
    ChampionRecord,
    CompStatsRecord,
    DataSnapshotMetadata,
    ItemInfo,
    ShopOddsRecord,
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
    def __init__(
        self,
        candidates: list[ChampionRecord],
        shop_odds: ShopOddsRecord | None = None,
    ) -> None:
        self.candidates = candidates
        self.shop_odds = shop_odds

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

    async def replace_shop_odds(self, records: Sequence[ShopOddsRecord]) -> None:
        raise AssertionError("not used")

    async def get_shop_odds(self, level: int, game_patch: str | None) -> ShopOddsRecord | None:
        return self.shop_odds

    async def replace_datatft_snapshot(
        self,
        metadata: DataSnapshotMetadata,
        records: Sequence[CompStatsRecord],
        raw_payload: str,
    ) -> None:
        raise AssertionError("not used")

    async def search_comps(
        self, query: str | None, game_patch: str, limit: int = 5
    ) -> list[CompStatsRecord]:
        return []

    async def latest_datatft_snapshot(self) -> tuple[DataSnapshotMetadata, int] | None:
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
    assert result.error.resolution == ToolErrorResolution.TERMINAL


@pytest.mark.asyncio
async def test_champion_tool_preserves_ambiguity() -> None:
    result = await GameDataTools(
        FakeGameDataRepository([champion("fixture-1"), champion("fixture-2")])
    ).lookup_champion("call-3", "Fixture Champion")

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "AMBIGUOUS_CHAMPION"
    assert result.error.retryable is False
    assert result.error.resolution == ToolErrorResolution.REQUEST_CONTEXT
    assert isinstance(result.data, dict)
    assert len(result.data["candidates"]) == 2


@pytest.mark.asyncio
async def test_shop_odds_tool_returns_structured_probabilities_and_provenance() -> None:
    record = ShopOddsRecord(
        level=7,
        cost_1_probability=0.16,
        cost_2_probability=0.30,
        cost_3_probability=0.43,
        cost_4_probability=0.10,
        cost_5_probability=0.01,
        game_patch="16.17",
        source_version="set18-fixture",
        source="fixture-source",
        source_type="community_reference",
        retrieved_at=datetime.now(UTC),
    )
    result = await GameDataTools(FakeGameDataRepository([], record)).get_shop_odds(
        "shop-1", {"level": 7}
    )

    assert result.success is True
    assert isinstance(result.data, dict)
    assert result.data["odds"]["cost_4"] == 0.10
    assert result.metadata.source_type == "community_reference"


@pytest.mark.asyncio
async def test_shop_odds_tool_returns_structured_not_found() -> None:
    result = await GameDataTools(FakeGameDataRepository([])).get_shop_odds("shop-2", {"level": 7})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "SHOP_ODDS_NOT_FOUND"
