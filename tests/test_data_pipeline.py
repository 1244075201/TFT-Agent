from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.test_integration_vertical_slice import FIXTURE_DOCUMENT
from tft_agent.ingestion.sources.communitydragon import CommunityDragonSource
from tft_agent.ingestion.sources.riot_data_dragon import (
    RiotChampionDocument,
    RiotDataDragonSource,
)
from tft_agent.ingestion.sources.shop_odds import FileShopOddsSource
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.schemas import StaticEntityRecord


def test_official_shape_fixture_normalizes_without_invented_fields() -> None:
    document = RiotChampionDocument.model_validate(FIXTURE_DOCUMENT)
    timestamp = datetime.now(UTC)

    records = RiotDataDragonSource().normalize_champions(document, timestamp)

    assert records[0].champion_id == "TFT_Fixture_Unique"
    assert records[0].cost == 3
    assert records[0].data_version == "fixture-version"
    assert records[0].retrieved_at == timestamp
    assert not hasattr(records[0], "traits")


def test_reviewed_shop_odds_snapshot_normalizes_with_reference_provenance() -> None:
    records = FileShopOddsSource(Path("data_sources/shop_odds_set18_16.17.json")).load()

    assert len(records) == 10
    level_seven = next(record for record in records if record.level == 7)
    assert level_seven.cost_4_probability == 0.10
    assert level_seven.source_type == "community_reference"


@pytest.mark.asyncio
async def test_shop_odds_repository_replaces_and_queries_snapshot(tmp_path: Path) -> None:
    records = FileShopOddsSource(Path("data_sources/shop_odds_set18_16.17.json")).load()
    repository = SQLiteGameDataRepository(tmp_path / "shop.sqlite3")
    await repository.initialize()
    await repository.replace_shop_odds(records)

    result = await repository.get_shop_odds(7, "16.17")
    assert result is not None
    assert result.cost_4_probability == 0.10
    assert await repository.get_shop_odds(11, None) is None


def test_communitydragon_normalizes_chinese_augment_effect() -> None:
    source = CommunityDragonSource("https://example.test/", "16.17")
    records = source.normalize(
        [
            {
                "nameId": "TFT18_Augment_BabyAcademy",
                "name": "宝宝学院",
                "desc": "队伍获得 @Power@% 加成",
                "effects": {"Power": 12},
            }
        ],
        "item",
        "https://example.test/tftitems.json",
        datetime(2026, 9, 9, tzinfo=UTC),
    )

    assert records[0].entity_type == "augment"
    assert records[0].description == "队伍获得 12% 加成"


@pytest.mark.asyncio
async def test_static_entities_and_session_state_round_trip(tmp_path: Path) -> None:
    repository = SQLiteGameDataRepository(tmp_path / "state.sqlite3")
    await repository.initialize()
    record = StaticEntityRecord(
        entity_id="TFT18_Augment_BabyAcademy",
        entity_type="augment",
        name="宝宝学院",
        description="测试效果",
        data_version="16.17",
        source="communitydragon",
        source_url="https://example.test/tft.json",
        retrieved_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    await repository.replace_static_entities([record])
    assert (await repository.search_static_entities(["宝宝学院"]))[0] == record

    await repository.save_session_state("session", '{"level":7}', '{"level":"user"}')
    assert await repository.load_session_state("session") == (
        '{"level":7}',
        '{"level":"user"}',
    )
    await repository.clear_session_state("session")
    assert await repository.load_session_state("session") is None
