"""UI-safe, patch-scoped entity catalog and deterministic search."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from pydantic import Field

from tft_agent.core.schemas import StrictModel
from tft_agent.tft_mcp_server.schemas import StaticEntityRecord


class EntityOption(StrictModel):
    canonical_id: str
    display_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    icon_ref: str | None = None


class CompositionCard(StrictModel):
    canonical_id: str
    display_name: str
    tier: str | None = None
    core_units: list[str] = Field(default_factory=list)
    summary: str | None = None


class CanonicalEntityCatalog:
    def __init__(self, options: Iterable[EntityOption]) -> None:
        self.options = list(options)

    @classmethod
    def from_repository(
        cls,
        rows: Iterable[Mapping[str, Any]],
        static_entities: Sequence[StaticEntityRecord] = (),
    ) -> CanonicalEntityCatalog:
        row_list = list(rows)
        details = {item.entity_id: item for item in static_entities}
        names = {str(row["canonical_id"]): str(row["display_name"]) for row in row_list}
        options: list[EntityOption] = []
        for row in row_list:
            canonical_id = str(row["canonical_id"])
            detail = details.get(canonical_id)
            tags = [str(value) for value in getattr(detail, "tags", [])]
            if canonical_id.casefold().startswith("da_component_"):
                tags.append("component")
            options.append(EntityOption(
                canonical_id=canonical_id,
                display_name=str(row["display_name"]),
                entity_type=str(row["entity_type"]),
                aliases=[str(value) for value in row.get("aliases", [])],
                tags=list(dict.fromkeys(tags)),
                components=[
                    names[str(value)]
                    for value in getattr(detail, "components", [])
                    if str(value) in names
                ],
            ))
        return cls(options)

    def for_type(
        self,
        entity_type: str,
        predicate: Callable[[EntityOption], bool] | None = None,
    ) -> list[EntityOption]:
        return [
            option for option in self.options
            if option.entity_type == entity_type and (predicate is None or predicate(option))
        ]

    def search(
        self, query: str, options: Iterable[EntityOption] | None = None, limit: int = 30
    ) -> list[EntityOption]:
        normalized = _normalize(query)
        source = list(options) if options is not None else self.options
        if not normalized:
            return sorted(source, key=lambda item: item.display_name)[:limit]
        ranked: list[tuple[int, int, str, EntityOption]] = []
        for option in source:
            canonical = _normalize(option.display_name)
            aliases = [_normalize(alias) for alias in option.aliases]
            values = [canonical, *aliases]
            if normalized == canonical:
                rank = 0
            elif normalized in aliases:
                rank = 1
            elif any(value.startswith(normalized) for value in values):
                rank = 2
            elif any(normalized in value for value in values):
                rank = 3
            else:
                continue
            ranked.append((rank, len(canonical), option.display_name, option))
        ranked.sort(key=lambda item: item[:3])
        return [item[-1] for item in ranked[:limit]]


def is_component(option: EntityOption) -> bool:
    return "component" in {tag.casefold() for tag in option.tags}


def is_completed_item(option: EntityOption) -> bool:
    return not is_component(option) and bool(option.components)


def search_match_reason(option: EntityOption, query: str) -> str:
    normalized = _normalize(query)
    if normalized and normalized in {_normalize(alias) for alias in option.aliases}:
        alias = next(alias for alias in option.aliases if _normalize(alias) == normalized)
        return f"匹配别名：{alias}"
    labels = {
        "composition": "当前版本 · 阵容",
        "augment": "当前 Set · 强化",
        "item": "当前 Set · 装备",
        "emblem": "当前 Set · 转职",
        "champion": "当前 Set · 棋子",
    }
    return labels.get(option.entity_type, option.entity_type)


def _normalize(value: str) -> str:
    return re.sub(r"[\s，,。！？!?、·•_\-]", "", value).casefold()
