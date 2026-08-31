"""Official Riot Data Dragon HTTP adapter and payload normalization."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, RootModel

from tft_agent.tft_mcp_server.schemas import ChampionImage, ChampionRecord

VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
CHAMPIONS_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/{locale}/tft-champion.json"
SOURCE_NAME = "riot-data-dragon"


class RiotImagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    full: str
    sprite: str
    group: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(ge=0)
    h: int = Field(ge=0)


class RiotChampionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    tier: int = Field(ge=1)
    cost: int | None = Field(default=None, ge=0)
    image: RiotImagePayload


class RiotChampionDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    format: str | None = None
    version: str
    data: dict[str, RiotChampionPayload]


class VersionsPayload(RootModel[list[str]]):
    pass


class RiotDataDragonSource:
    """Fetch and validate only Riot Data Dragon resources."""

    def __init__(self, timeout_seconds: float = 30.0, locale: str = "en_US") -> None:
        self._timeout_seconds = timeout_seconds
        self._locale = locale

    def get_versions(self) -> list[str]:
        versions = VersionsPayload.model_validate(self._get_json(VERSIONS_URL)).root
        if not versions:
            raise ValueError("Riot Data Dragon returned an empty versions list")
        return versions

    def resolve_data_version(self, requested: str | None = None) -> str:
        versions = self.get_versions()
        if requested is None:
            return versions[0]
        if requested not in versions:
            raise ValueError(f"Data Dragon version is unavailable: {requested}")
        return requested

    def get_champions(self, data_version: str) -> RiotChampionDocument:
        url = CHAMPIONS_URL.format(version=data_version, locale=self._locale)
        document = RiotChampionDocument.model_validate(self._get_json(url))
        if document.version != data_version:
            raise ValueError(
                f"Champion payload version {document.version!r} does not match {data_version!r}"
            )
        return document

    def normalize_champions(
        self, document: RiotChampionDocument, retrieved_at: datetime | None = None
    ) -> list[ChampionRecord]:
        timestamp = retrieved_at or datetime.now(UTC)
        return [
            ChampionRecord(
                champion_id=champion.id,
                name=champion.name,
                cost=champion.tier,
                image=ChampionImage.model_validate(champion.image.model_dump()),
                data_version=document.version,
                source=SOURCE_NAME,
                retrieved_at=timestamp,
            )
            for champion in document.data.values()
        ]

    def _get_json(self, url: str) -> Mapping[str, Any] | list[Any]:
        request = Request(url, headers={"User-Agent": "tft-decision-agent/0.1"})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                return cast(Mapping[str, Any] | list[Any], json.load(response))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to fetch Riot Data Dragon resource {url}: {exc}") from exc
