"""Strict contracts for untrusted visual observations and safe state merging."""

from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from tft_agent.core.schemas import GameState, StrictModel


class ObservationCertainty(StrEnum):
    OBSERVED = "observed"
    UNCERTAIN = "uncertain"


class VisionUnit(StrictModel):
    name: str | None = None
    observed_label: str | None = None
    star_level: int | None = Field(default=None, ge=1, le=3)
    count: int = Field(default=1, ge=1, le=9)
    items: list[str] = Field(default_factory=list, max_length=3)
    identity_status: ObservationCertainty = ObservationCertainty.OBSERVED

    @model_validator(mode="after")
    def require_identity(self) -> "VisionUnit":
        if not self.name and not self.observed_label:
            raise ValueError("unit requires a name or unresolved observed_label")
        if self.identity_status == ObservationCertainty.UNCERTAIN and not self.observed_label:
            raise ValueError("uncertain identity requires observed_label")
        return self


class VisionObservation(StrictModel):
    observation_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    image_source: str = Field(min_length=1)
    stage: str | None = None
    level: int | None = Field(default=None, ge=1, le=10)
    hp: int | None = Field(default=None, ge=0, le=100)
    gold: int | None = Field(default=None, ge=0)
    board_units: list[VisionUnit] | None = Field(default=None, max_length=11)
    bench_units: list[VisionUnit] | None = Field(default=None, max_length=9)
    item_components: list[str] | None = None
    completed_items: list[str] | None = None
    augments: list[str] | None = None
    win_streak: int | None = Field(default=None, ge=1, le=20)
    loss_streak: int | None = Field(default=None, ge=1, le=20)
    uncertain_fields: list[str] = Field(default_factory=list)
    not_visible_fields: list[str] = Field(default_factory=list)

    @field_validator("uncertain_fields", "not_visible_fields")
    @classmethod
    def unique_paths(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("field paths cannot be blank")
        return list(dict.fromkeys(item.strip() for item in value))

    @model_validator(mode="after")
    def validate_visibility(self) -> "VisionObservation":
        if set(self.uncertain_fields) & set(self.not_visible_fields):
            raise ValueError("a field cannot be both uncertain and not_visible")
        if self.win_streak is not None and self.loss_streak is not None:
            raise ValueError("win_streak and loss_streak are mutually exclusive")
        for field in self.not_visible_fields:
            root = field.split(".", 1)[0]
            if root in type(self).model_fields and getattr(self, root) is not None:
                raise ValueError(f"not-visible field must be null: {root}")
        return self


class StateConflict(StrictModel):
    field: str
    user_value: Any
    vision_value: Any
    sources: list[str] = Field(min_length=2)


class GameStateMergeResult(StrictModel):
    game_state: GameState
    conflicts: list[StateConflict] = Field(default_factory=list)
    vision_filled_fields: list[str] = Field(default_factory=list)


class VisionTokenUsage(StrictModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class VisionTrace(StrictModel):
    observation_id: str
    image_name: str
    image_sha256: str
    dimensions: tuple[int, int]
    provider: str
    model: str
    detail: str
    extracted_fields: list[str]
    uncertain_fields: list[str]
    not_visible_fields: list[str]
    latency_ms: float = Field(ge=0)
    token_usage: VisionTokenUsage | None = None
    validation_status: str


class VisionExtraction(StrictModel):
    observation: VisionObservation
    trace: VisionTrace


def safe_image_name(path: Path) -> str:
    return path.name
