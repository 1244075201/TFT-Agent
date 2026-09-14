"""Audited alias assets for the active canonical entity catalog."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from tft_agent.core.schemas import StrictModel


class AliasEntry(StrictModel):
    aliases: list[str] = Field(min_length=1)
    entity_type: str
    canonical_name: str
    scope: Literal["global", "set", "patch"]
    canonical_id: str | None = None
    set_number: int | None = None
    game_patch: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    review_status: Literal["approved", "rejected", "deferred"] = "approved"
    notes: str = ""


class AliasRejection(StrictModel):
    alias: str
    canonical_name: str
    reason: str


class MaterializedAlias(StrictModel):
    alias: str
    canonical_id: str
    entity_type: str
    alias_kind: str


class EntityTextNormalizer:
    _PUNCTUATION = re.compile(r"[\s，,。！？!?、·•_\-:：;；'\"“”‘’（）()【】\[\]]+")

    @classmethod
    def normalize(cls, value: str) -> str:
        return cls._PUNCTUATION.sub("", unicodedata.normalize("NFKC", value).casefold())


class AliasRegistry:
    def __init__(self, entries: list[AliasEntry]) -> None:
        self.entries = entries
        self.last_rejections: list[AliasRejection] = []
        self._materialized: list[MaterializedAlias] = []

    @classmethod
    def load_default(cls) -> AliasRegistry:
        root = Path(__file__).parents[3] / "data_sources" / "aliases"
        entries: list[AliasEntry] = []
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows = payload.get("aliases", []) if isinstance(payload, dict) else []
            entries.extend(AliasEntry.model_validate(row) for row in rows)
        return cls(entries)

    def materialize(
        self,
        entities: list[tuple[str, str, str, str]],
        set_number: int,
        game_patch: str,
    ) -> list[MaterializedAlias]:
        by_name: dict[tuple[str, str], list[str]] = {}
        available = {(identifier, kind, name) for identifier, kind, name, _ in entities}
        for identifier, kind, name, _ in entities:
            by_name.setdefault((kind, name), []).append(identifier)
        output: list[MaterializedAlias] = []
        rejected: list[AliasRejection] = []
        for entry in self.entries:
            if entry.review_status != "approved":
                rejected.extend(AliasRejection(
                    alias=alias, canonical_name=entry.canonical_name,
                    reason=f"review_{entry.review_status}",
                ) for alias in entry.aliases)
                continue
            if entry.scope == "set" and entry.set_number != set_number:
                continue
            if entry.scope == "patch" and entry.game_patch != game_patch:
                continue
            identifiers = (
                [entry.canonical_id]
                if entry.canonical_id
                and (entry.canonical_id, entry.entity_type, entry.canonical_name) in available
                else by_name.get((entry.entity_type, entry.canonical_name), [])
                if entry.canonical_id is None
                else []
            )
            for alias in entry.aliases:
                if len(identifiers) != 1:
                    reason = "target_missing" if not identifiers else "target_ambiguous"
                    rejected.append(AliasRejection(
                        alias=alias, canonical_name=entry.canonical_name, reason=reason
                    ))
                    continue
                output.append(MaterializedAlias(
                    alias=alias,
                    canonical_id=identifiers[0],
                    entity_type=entry.entity_type,
                    alias_kind=f"registry_{entry.scope}",
                ))
        output.extend(self._materialize_emblem_expressions(entities, output))
        by_alias: dict[str, set[tuple[str, str]]] = {}
        for item in output:
            by_alias.setdefault(EntityTextNormalizer.normalize(item.alias), set()).add(
                (item.canonical_id, item.entity_type)
            )
        conflicts = {key for key, values in by_alias.items() if len(values) > 1}
        accepted: list[MaterializedAlias] = []
        seen: set[tuple[str, str, str]] = set()
        for item in output:
            if EntityTextNormalizer.normalize(item.alias) in conflicts:
                rejected.append(AliasRejection(
                    alias=item.alias, canonical_name="", reason="alias_conflict"
                ))
            elif (
                EntityTextNormalizer.normalize(item.alias),
                item.canonical_id,
                item.entity_type,
            ) not in seen:
                accepted.append(item)
                seen.add((
                    EntityTextNormalizer.normalize(item.alias),
                    item.canonical_id,
                    item.entity_type,
                ))
        self.last_rejections = rejected
        self._materialized = accepted
        return accepted

    @staticmethod
    def _materialize_emblem_expressions(
        entities: list[tuple[str, str, str, str]],
        reviewed: list[MaterializedAlias],
    ) -> list[MaterializedAlias]:
        """Derive product phrases only for a unique, real trait-to-emblem relation."""
        emblems_by_name: dict[str, list[str]] = {}
        traits: dict[str, str] = {}
        for identifier, kind, name, _ in entities:
            if kind == "emblem":
                emblems_by_name.setdefault(name, []).append(identifier)
            elif kind == "trait":
                traits[identifier] = name

        stems_by_trait: dict[str, set[str]] = {}
        for item in reviewed:
            if item.entity_type == "trait" and item.canonical_id in traits:
                stems_by_trait.setdefault(item.canonical_id, set()).add(item.alias)

        generated: list[MaterializedAlias] = []
        for trait_id, trait_name in traits.items():
            emblem_ids = emblems_by_name.get(trait_name + "纹章", [])
            if len(emblem_ids) != 1:
                continue
            emblem_id = emblem_ids[0]
            aliases = {trait_name + "转", trait_name + "转职"}
            for stem in stems_by_trait.get(trait_id, set()):
                aliases.update({stem + "转", stem + "纹章", stem + "转职"})
            generated.extend(
                MaterializedAlias(
                    alias=alias,
                    canonical_id=emblem_id,
                    entity_type="emblem",
                    alias_kind="derived_emblem",
                )
                for alias in sorted(aliases)
            )
        return generated

    def rejections(self) -> list[AliasRejection]:
        return list(self.last_rejections)

    def search(self, text: str) -> list[MaterializedAlias]:
        query = EntityTextNormalizer.normalize(text)
        matches = [item for item in self._materialized
                   if EntityTextNormalizer.normalize(item.alias) in query]
        return sorted(matches, key=lambda item: len(EntityTextNormalizer.normalize(item.alias)),
                      reverse=True)

    def resolve(self, text: str) -> MaterializedAlias | None:
        normalized = EntityTextNormalizer.normalize(text)
        matches = [item for item in self._materialized
                   if EntityTextNormalizer.normalize(item.alias) == normalized]
        targets = {(item.canonical_id, item.entity_type) for item in matches}
        return matches[0] if len(targets) == 1 else None

    def repository_entries(self) -> list[dict[str, Any]]:
        return [entry.model_dump(mode="json") for entry in self.entries]
