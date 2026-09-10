"""Conservative, patch/set-scoped canonical entity resolution."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import Field

from tft_agent.core.config import Settings
from tft_agent.core.schemas import StrictModel
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


class ResolvedEntity(StrictModel):
    canonical_id: str
    entity_type: str
    display_name: str
    game_patch: str
    set_number: int
    set_name: str
    matched_alias: str
    match_kind: str
    match_score: float = Field(ge=0, le=1)


class EntityResolution(StrictModel):
    entities: list[ResolvedEntity] = Field(default_factory=list)
    unresolved_terms: list[str] = Field(default_factory=list)


class CanonicalEntityResolver:
    def __init__(self, repository: SQLiteGameDataRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings
        self._catalog_ready = False

    async def resolve(self, query: str) -> EntityResolution:
        catalog = await self.repository.canonical_catalog(
            self.settings.current_game_patch, self.settings.current_set_number
        )
        if not self._catalog_ready and not catalog:
            await self.repository.rebuild_canonical_catalog(
                self.settings.current_game_patch,
                self.settings.current_set_number,
                self.settings.current_set_name,
                explicit_aliases(),
            )
            catalog = await self.repository.canonical_catalog(
                self.settings.current_game_patch, self.settings.current_set_number
            )
        self._catalog_ready = True
        normalized_query = _normalize(query)
        matches: dict[tuple[str, str], tuple[dict[str, object], str, str, float]] = {}
        # Exact canonical and aliases are checked before generic span matching.
        for entry in catalog:
            canonical = _normalize(str(entry["display_name"]))
            aliases = [_normalize(str(alias)) for alias in entry["aliases"]]
            if normalized_query == canonical:
                _keep(matches, entry, canonical, "exact_canonical", 1.0)
            elif normalized_query in aliases:
                _keep(matches, entry, normalized_query, "exact_alias", 1.0)
        if not matches:
            spans: list[tuple[int, dict[str, object], str, str]] = []
            for entry in catalog:
                terms = [(str(entry["display_name"]), "canonical"), *(
                    (str(alias), "alias") for alias in entry["aliases"]
                )]
                for term, source_kind in terms:
                    normalized = _normalize(term)
                    if len(normalized) >= 2 and normalized in normalized_query:
                        spans.append((len(normalized), entry, term, source_kind))
            for entry, term in _unambiguous_spans(query, spans):
                _keep(matches, entry, term, "longest_exact_span", 1.0)
        if not matches:
            scored: list[tuple[float, dict[str, object], str]] = []
            for entry in catalog:
                for term in [str(entry["display_name"]), *map(str, entry["aliases"])]:
                    score = SequenceMatcher(None, normalized_query, _normalize(term)).ratio()
                    if score >= self.settings.entity_fuzzy_threshold:
                        scored.append((score, entry, term))
            if scored:
                best = max(score for score, _, _ in scored)
                winners = {(str(entry["canonical_id"]), str(entry["entity_type"]))
                           for score, entry, _ in scored if score == best}
                if len(winners) == 1:
                    score, entry, term = next(item for item in scored if item[0] == best)
                    _keep(matches, entry, term, "unique_fuzzy", score)
        return EntityResolution(entities=[_resolved(*value) for value in matches.values()])

    async def catalog(self) -> list[dict[str, object]]:
        if not self._catalog_ready:
            await self.resolve("初始化当前版本实体目录")
        return await self.repository.canonical_catalog(
            self.settings.current_game_patch, self.settings.current_set_number
        )


def _unambiguous_spans(
    query: str, spans: list[tuple[int, dict[str, object], str, str]]
) -> list[tuple[dict[str, object], str]]:
    """Choose longest exact spans while refusing same-name and cross-type ambiguity."""
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, object], str, str]]] = {}
    for candidate in spans:
        _, entry, term, _ = candidate
        grouped.setdefault((_normalize(term), str(entry["entity_type"])), []).append(candidate)
    unique = [values[0] for values in grouped.values()
              if len({str(value[1]["canonical_id"]) for value in values}) == 1]

    by_term: dict[str, list[tuple[int, dict[str, object], str, str]]] = {}
    for candidate in unique:
        by_term.setdefault(_normalize(candidate[2]), []).append(candidate)
    augment_context = bool(re.search(r"选哪个|三选一|海克斯|强化", query))
    filtered: list[tuple[int, dict[str, object], str, str]] = []
    blocked_terms: list[str] = []
    for term, candidates in by_term.items():
        kinds = {str(value[1]["entity_type"]) for value in candidates}
        if len(kinds) == 1:
            filtered.extend(candidates)
        elif augment_context and "augment" in kinds:
            filtered.extend(value for value in candidates
                            if value[1]["entity_type"] == "augment")
        else:
            # A composition explicitly introduced by “玩” remains unambiguous even
            # when its name collides with another entity type.
            comp = [value for value in candidates
                    if value[1]["entity_type"] == "composition"
                    and f"玩{term}" in _normalize(query)]
            filtered.extend(comp)
            if not comp:
                blocked_terms.append(term)

    output: list[tuple[dict[str, object], str]] = []
    accepted_terms: list[str] = []
    for _, entry, term, _ in sorted(filtered, key=lambda value: value[0], reverse=True):
        normalized = _normalize(term)
        if any(normalized in blocked and normalized != blocked for blocked in blocked_terms):
            continue
        if any(normalized in accepted and normalized != accepted for accepted in accepted_terms):
            continue
        output.append((entry, term))
        accepted_terms.append(normalized)
    return output


def _keep(
    output: dict[tuple[str, str], tuple[dict[str, object], str, str, float]],
    entry: dict[str, object], term: str, kind: str, score: float,
) -> None:
    key = (str(entry["canonical_id"]), str(entry["entity_type"]))
    current = output.get(key)
    if current is None or len(term) > len(current[1]):
        output[key] = (entry, term, kind, score)


def _resolved(
    entry: dict[str, object], term: str, kind: str, score: float
) -> ResolvedEntity:
    return ResolvedEntity(
        canonical_id=str(entry["canonical_id"]), entity_type=str(entry["entity_type"]),
        display_name=str(entry["display_name"]), game_patch=str(entry["game_patch"]),
        set_number=int(str(entry["set_number"])), set_name=str(entry["set_name"]),
        matched_alias=term, match_kind=kind, match_score=score,
    )


def _normalize(value: str) -> str:
    return re.sub(r"[\s，,。！？!?、·•_\-]", "", value).casefold()


def explicit_aliases() -> list[dict[str, str]]:
    path = Path(__file__).parents[3] / "data_sources" / "entity_aliases.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    aliases = value.get("aliases", []) if isinstance(value, dict) else []
    return [dict(item) for item in aliases if isinstance(item, dict)]
