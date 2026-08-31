"""Replaceable source boundary for versioned shop-odds reference snapshots."""

from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from tft_agent.tft_mcp_server.schemas import ShopOddsRecord


class ShopOddsSource(Protocol):
    """Load normalized shop odds without knowing their persistence target."""

    def load(self) -> list[ShopOddsRecord]: ...


class _Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_patch: str
    source_version: str
    source: str
    source_type: str
    retrieved_at: datetime
    odds_by_level: dict[int, tuple[float, float, float, float, float]]


class FileShopOddsSource:
    """Normalize a reviewed, versioned reference snapshot from disk."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[ShopOddsRecord]:
        snapshot = _Snapshot.model_validate_json(self._path.read_text(encoding="utf-8"))
        return [
            ShopOddsRecord(
                level=level,
                cost_1_probability=odds[0],
                cost_2_probability=odds[1],
                cost_3_probability=odds[2],
                cost_4_probability=odds[3],
                cost_5_probability=odds[4],
                game_patch=snapshot.game_patch,
                source_version=snapshot.source_version,
                source=snapshot.source,
                source_type=snapshot.source_type,
                retrieved_at=snapshot.retrieved_at,
            )
            for level, odds in sorted(snapshot.odds_by_level.items())
        ]
