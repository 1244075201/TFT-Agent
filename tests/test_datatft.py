import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tft_agent.ingestion.sources import datatft
from tft_agent.ingestion.sources.datatft import (
    DataTFTSource,
    RawHTTPResponse,
    _extract_current_patch,
)
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


class _FakeResponse:
    def __init__(self, body: str, content_type: str) -> None:
        self._body = body.encode()
        self.status = 200
        self.headers = {"Content-Type": content_type}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_discovers_patch_from_vite_bootstrap_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = {
        "https://www.datatft.com/explorer": _FakeResponse(
            '<script type="module" src="/assets/index-current.js"></script>', "text/html"
        ),
        "https://www.datatft.com/assets/index-current.js": _FakeResponse(
            'const deps=["assets/bootstrap-current.js","assets/site.css"]',
            "application/javascript",
        ),
        "https://www.datatft.com/assets/bootstrap-current.js": _FakeResponse(
            'const Cxe=crypto,xxe="18.2",gc=[{value:"18.2"},{value:"18.1d"}]',
            "application/javascript",
        ),
    }

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        del timeout
        return pages[request.full_url]

    monkeypatch.setattr(datatft, "urlopen", fake_urlopen)
    responses: list[RawHTTPResponse] = []
    source = DataTFTSource("https://www.datatft.com/comps/tier/")

    assert source.discover_current_patch(responses) == "18.2"
    assert [item.request_url for item in responses] == list(pages)


@pytest.mark.parametrize("label", ["Ver: 18.2b", "数据版本 18.2b"])
def test_legacy_patch_detection_preserves_hotfix(label: str) -> None:
    assert _extract_current_patch(label) == "18.2b"


def test_patch_detection_ignores_unstructured_version_like_number() -> None:
    assert _extract_current_patch('<div data-build="18.2">version 18.2</div>') is None


def test_live_fetch_rejects_expected_patch_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    source = DataTFTSource("https://www.datatft.com/comps/tier/")
    monkeypatch.setattr(source, "_assert_robots_allows", lambda: None)
    monkeypatch.setattr(source, "discover_current_patch", lambda _responses=None: "18.2")

    with pytest.raises(ValueError, match="18.1d.*18.2"):
        source.fetch_current_with_raw([], "18.1d", 0.25)


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
