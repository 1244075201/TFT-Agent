"""Typed records returned by TFT repositories and tools."""

from datetime import datetime
from typing import Any

from pydantic import Field

from tft_agent.core.schemas import StrictModel


class ChampionImage(StrictModel):
    """Normalized Data Dragon image metadata."""

    full: str
    sprite: str
    group: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(alias="w", ge=0)
    height: int = Field(alias="h", ge=0)


class ChampionRecord(StrictModel):
    """Storage-independent normalized champion record."""

    champion_id: str
    name: str
    cost: int = Field(ge=1)
    image: ChampionImage
    data_version: str
    source: str
    retrieved_at: datetime


class ItemInfo(StrictModel):
    name: str
    components: list[str]
    stats: dict[str, Any]
    effect: str
    item_type: str


class TraitBreakpoint(StrictModel):
    count: int = Field(ge=1)
    effect: str


class TraitInfo(StrictModel):
    name: str
    breakpoints: list[TraitBreakpoint]
    associated_champions: list[str]


class ShopOdds(StrictModel):
    level: int = Field(ge=1, le=10)
    probabilities_by_cost: dict[int, float]


class KnowledgeChunk(StrictModel):
    text: str
    source: str
    source_type: str
    game_patch: str | None = None
    relevance_score: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
