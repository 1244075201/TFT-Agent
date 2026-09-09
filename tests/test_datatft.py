import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tft_agent.ingestion.sources.datatft import DataTFTSource, _extract_current_patch
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


def snapshot_payload() -> str:
    return json.dumps(
        {
            "game_patch": "18.1",
            "sample_scope": "CN high elo",
            "comps": [
                {
                    "id": "spirit-mage",
                    "name": "灵魂法师",
                    "tier": "S",
                    "average_place": 3.8,
                    "play_rate": "12%",
                    "top4_rate": 0.57,
                    "win_rate": "15%",
                    "core_units": ["卡尔玛", "维迦"],
                    "recommended_items": ["朔极之矛"],
                    "augments": ["法师之徽"],
                    "level_plan": ["4-1升8"],
                }
            ],
        },
        ensure_ascii=False,
    )


def test_parse_canonical_datatft_snapshot() -> None:
    source = DataTFTSource("https://www.datatft.com/comps/tier/")
    metadata, records = source.parse(
        snapshot_payload(), retrieved_at=datetime(2026, 9, 9, tzinfo=UTC)
    )

    assert metadata.game_patch == "18.1"
    assert records[0].play_rate == 0.12
    assert records[0].snapshot.content_hash == metadata.content_hash


def test_parser_rejects_unknown_page_shape() -> None:
    source = DataTFTSource("https://www.datatft.com/comps/tier/")
    with pytest.raises(ValueError, match="supported public snapshot"):
        source.parse("<html><body>changed</body></html>")


def test_extracts_current_patch_from_frontend_constant() -> None:
    script = 'const Mxe="18.1d",vc=[{value:"18.1d"},{value:"18.1"}]'
    assert _extract_current_patch(script) == "18.1d"


@pytest.mark.asyncio
async def test_snapshot_round_trip_and_exact_patch(tmp_path: Path) -> None:
    source = DataTFTSource("https://www.datatft.com/comps/tier/")
    metadata, records = source.parse(snapshot_payload())
    repository = SQLiteGameDataRepository(tmp_path / "tft.sqlite3")
    await repository.initialize()
    await repository.replace_datatft_snapshot(metadata, records, snapshot_payload())

    found = await repository.search_comps("灵魂", "18.1")
    assert found[0].name == "灵魂法师"
    assert found[0].snapshot.source_url == "https://www.datatft.com/comps/tier/"
    assert await repository.search_comps(None, "18.2") == []
