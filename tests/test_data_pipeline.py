from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.test_integration_vertical_slice import FIXTURE_DOCUMENT
from tft_agent.ingestion.sources.riot_data_dragon import (
    RiotChampionDocument,
    RiotDataDragonSource,
)
from tft_agent.ingestion.sources.shop_odds import FileShopOddsSource
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


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
