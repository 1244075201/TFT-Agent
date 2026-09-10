"""Chinese CommunityDragon static-definition adapter."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, cast
from urllib.request import Request, urlopen

from tft_agent.tft_mcp_server.schemas import StaticEntityRecord


class CommunityDragonSource:
    def __init__(
        self,
        base_url: str,
        data_version: str,
        tft_url: str | None = None,
        current_set_number: int | None = None,
        timeout_seconds: float = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.data_version = data_version
        self.tft_url = tft_url
        self.timeout_seconds = timeout_seconds
        self.current_set_number = current_set_number
        self.active_entity_ids: list[str] = []

    def fetch(self) -> list[StaticEntityRecord]:
        timestamp = datetime.now(UTC)
        records: list[StaticEntityRecord] = []
        for filename, default_type in (
            ("tftitems.json", "item"),
            ("tftchampions.json", "champion"),
            ("tfttraits.json", "trait"),
        ):
            url = self.base_url + filename
            records.extend(self.normalize(self._get_json(url), default_type, url, timestamp))
        if self.tft_url:
            payload = self._get_json(self.tft_url)
            records.extend(self.normalize(payload, "item", self.tft_url, timestamp))
            records.extend(self._active_set_records(payload, timestamp))
            self.active_entity_ids = self._active_ids(payload)
        return list({record.entity_id: record for record in records}.values())

    def _active_set_records(
        self, payload: Any, timestamp: datetime
    ) -> list[StaticEntityRecord]:
        if self.current_set_number is None or not isinstance(payload, dict):
            return []
        set_data = payload.get("setData")
        if not isinstance(set_data, list):
            return []
        selected = next((item for item in set_data if isinstance(item, dict)
                         and item.get("number") == self.current_set_number), None)
        if not isinstance(selected, dict):
            return []
        records: list[StaticEntityRecord] = []
        for key, entity_type in (("champions", "champion"), ("traits", "trait")):
            records.extend(self.normalize(
                selected.get(key, []), entity_type, self.tft_url or self.base_url, timestamp
            ))
        return records

    def _active_ids(self, payload: Any) -> list[str]:
        if self.current_set_number is None or not isinstance(payload, dict):
            return []
        set_data = payload.get("setData")
        if not isinstance(set_data, list):
            return []
        selected = next((item for item in set_data if isinstance(item, dict)
                         and item.get("number") == self.current_set_number), None)
        if not isinstance(selected, dict):
            raise ValueError(f"CommunityDragon has no setData for set {self.current_set_number}")
        identifiers: list[str] = []
        for key in ("augments", "items"):
            identifiers.extend(str(item) for item in selected.get(key, []) if item)
        for key in ("champions", "traits"):
            for item in selected.get(key, []):
                if isinstance(item, dict) and item.get("apiName"):
                    identifiers.append(str(item["apiName"]))
        return list(dict.fromkeys(identifiers))

    def normalize(
        self,
        payload: Any,
        default_type: str,
        source_url: str,
        retrieved_at: datetime | None = None,
    ) -> list[StaticEntityRecord]:
        timestamp = retrieved_at or datetime.now(UTC)
        entries = (
            payload
            if isinstance(payload, list)
            else payload.get("items", payload.get("data", []))
        )
        if isinstance(entries, dict):
            entries = list(entries.values())
        output: list[StaticEntityRecord] = []
        for raw in entries if isinstance(entries, list) else []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            entity_id = str(
                raw.get("apiName") or raw.get("nameId") or raw.get("id") or raw.get("guid") or ""
            ).strip()
            if not name or not entity_id:
                continue
            components = _names(raw.get("composition") or raw.get("components"))
            description = _clean_html(
                str(raw.get("desc") or raw.get("description") or "")
            )
            effects = raw.get("effects")
            if isinstance(effects, dict):
                for key, value in effects.items():
                    description = description.replace(f"@{key}@", str(value))
            output.append(
                StaticEntityRecord(
                    entity_id=entity_id,
                    entity_type=_entity_type(entity_id, default_type, components),
                    name=name,
                    description=description,
                    traits=_names(raw.get("traits") or raw.get("associatedTraits")),
                    components=components,
                    tags=_names(raw.get("tags")),
                    data_version=self.data_version,
                    source="communitydragon",
                    source_url=source_url,
                    retrieved_at=timestamp,
                )
            )
        return output

    def _get_json(self, url: str) -> Any:
        request = Request(url, headers={"User-Agent": "tft-decision-agent/0.1"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return cast(Any, json.load(response))


def _entity_type(entity_id: str, default: str, components: list[str]) -> str:
    folded = entity_id.casefold()
    if "augment" in folded:
        return "augment"
    if "emblem" in folded or "spatula" in folded:
        return "emblem"
    if components:
        return "item"
    return default


def _names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item.get("name") if isinstance(item, dict) else item).strip()
        for item in value
        if item
    ]


def _clean_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()
