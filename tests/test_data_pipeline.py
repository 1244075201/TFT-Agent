from datetime import UTC, datetime

from tests.test_integration_vertical_slice import FIXTURE_DOCUMENT
from tft_agent.ingestion.sources.riot_data_dragon import (
    RiotChampionDocument,
    RiotDataDragonSource,
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
