"""Typed records returned by TFT repositories and tools."""

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

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


class ShopOddsRecord(StrictModel):
    """Normalized per-slot shop cost-tier probabilities with provenance."""

    level: int = Field(ge=1, le=10)
    cost_1_probability: float = Field(ge=0, le=1)
    cost_2_probability: float = Field(ge=0, le=1)
    cost_3_probability: float = Field(ge=0, le=1)
    cost_4_probability: float = Field(ge=0, le=1)
    cost_5_probability: float = Field(ge=0, le=1)
    game_patch: str
    source_version: str
    source: str
    source_type: str
    retrieved_at: datetime

    @model_validator(mode="after")
    def validate_probability_sum(self) -> "ShopOddsRecord":
        total = sum(
            (
                self.cost_1_probability,
                self.cost_2_probability,
                self.cost_3_probability,
                self.cost_4_probability,
                self.cost_5_probability,
            )
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("shop cost-tier probabilities must sum to 1.0")
        return self


class KnowledgeChunk(StrictModel):
    chunk_id: str
    document_id: str
    title: str
    heading: str | None = None
    text: str
    source: str
    source_type: str
    document_type: str
    game_patch: str | None = None
    set_name: str | None = None
    language: str
    url: str
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime
    embedding_model: str | None = None
    relevance_score: float | None = Field(default=None, ge=-1, le=1)


class KnowledgeDocument(StrictModel):
    """Normalized source snapshot without storage-specific embedding fields."""

    document_id: str
    title: str
    content: str
    source: str
    source_type: str
    document_type: str
    game_patch: str | None = None
    set_name: str | None = None
    language: str
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime
    url: str


class KnowledgeChunkRecord(KnowledgeChunk):
    """Indexable chunk with embedding version metadata."""

    chunk_index: int = Field(ge=0)
    embedding: list[float]
    embedding_model: str
    embedding_model_revision: str | None = None
    embedding_dimension: int = Field(gt=0)
    embedded_at: datetime
