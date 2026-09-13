"""Deterministic DataTFT runtime-catalog and team-builder normalization."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

MAP_CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!<>."
DEFAULT_CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
EXCLUDED_EQUIP_IDS = {612, 545, 573, 31004, 31005, 31006, 31007, 2252}
APPENDED_EQUIP_IDS = {31004, 31005, 31006, 31007}
EXTRA_TRAIT_KEYS = (
    "TFT15_Destroyer",
    "TFT15_Juggernaut",
    "TFT15_Duelist",
    "TFT17_ManaTrait",
    "TFT17_ASTrait",
    "TFT17_APTrait",
    "DA_18_Blossom",
    "DA_18_Inferno",
    "DA_18_Elderwood",
    "DA_18_Blackthorn",
    "DA_18_Coven",
    "DA_18_Fae",
    "DA_Primal18",
    "DA_18_Solar",
    "DA_18_Lunar",
    "DA_18_Eclipse",
    "DA_18_Slayer",
    "DA_18_Rapidfire",
    "DA_18_Spellweaver",
    "DA_18_Executioner",
)
POWERUP_POSITIONS = {"L": "left", "R": "right", "T": "top", "B": "bottom"}
# The official ``s4`` hero selector appends these set-specific pseudo-units before
# assigning fallback zA/zB/... URL codes. They are not champion catalog guesses.
SPECIAL_HERO_IDS_BY_SET = {
    18: (9977, 9978, 999001, 999002, 999003, 999004),
}


@dataclass(frozen=True)
class RuntimeEntity:
    external_id: str
    key: str | None
    name: str | None
    code: str | None


@dataclass(frozen=True)
class RuntimeCatalog:
    heroes_by_id: dict[str, RuntimeEntity]
    heroes_by_code: dict[str, RuntimeEntity]
    items_by_id: dict[str, RuntimeEntity]
    items_by_code: dict[str, RuntimeEntity]
    hexes_by_id: dict[str, RuntimeEntity]
    hexes_by_key: dict[str, RuntimeEntity]
    hexes_by_code: dict[str, RuntimeEntity]
    powerups_by_code: dict[str, RuntimeEntity]


def decrypt_runtime_catalog(module_source: str, encoded_key: str) -> dict[str, Any]:
    """Port the frontend's Base64 -> AES/ECB/PKCS7 -> JSON operation."""

    match = re.search(r'const\s+\w+="([A-Za-z0-9+/=]+)";export\{', module_source)
    if match is None:
        raise ValueError("DataTFT runtime catalog module has no encrypted data export")
    key = base64.b64decode(encoded_key, validate=True)
    ciphertext = base64.b64decode(match.group(1), validate=True)
    decryptor = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = PKCS7(algorithms.AES.block_size).unpadder()
    decoded = unpadder.update(padded) + unpadder.finalize()
    value = json.loads(decoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DataTFT runtime catalog root is not an object")
    return dict(value)


def build_runtime_catalog(raw: dict[str, Any], set_number: int) -> RuntimeCatalog:
    heroes = _entities(raw.get(f"heros{set_number}"), "chessId", "displayName")
    special_heroes = [
        RuntimeEntity(str(identifier), str(identifier), None, None)
        for identifier in SPECIAL_HERO_IDS_BY_SET.get(set_number, ())
    ]
    code_heroes = [*heroes, *special_heroes]
    raw_items = _entities(raw.get(f"equips{set_number}"), "equipId", "name")
    ordered_items = [
        item for item in raw_items if int(item.external_id) not in EXCLUDED_EQUIP_IDS
    ] + [item for item in raw_items if int(item.external_id) in APPENDED_EQUIP_IDS]
    hexes = [
        RuntimeEntity(item.external_id, item.key or item.external_id, item.name, item.code)
        for item in _entities(_flatten(raw.get(f"hexs{set_number}")), "hexId", "name")
    ]
    powerups = _entities(raw.get(f"anomalies{set_number}"), "key", "name")

    hero_code: dict[str, RuntimeEntity] = {}
    fallback_index = 0
    for hero in code_heroes:
        code = hero.code
        if not code:
            code = "z" + DEFAULT_CODES[fallback_index]
            fallback_index += 1
        hero_code[code] = hero
    item_code: dict[str, RuntimeEntity] = {}
    for index, item in enumerate(ordered_items):
        generated = DEFAULT_CODES[index // 52] + DEFAULT_CODES[index % 52]
        item_code[item.code or generated] = item
    return RuntimeCatalog(
        heroes_by_id={item.external_id: item for item in code_heroes},
        heroes_by_code=hero_code,
        items_by_id={item.external_id: item for item in raw_items},
        items_by_code=item_code,
        hexes_by_id={item.external_id: item for item in hexes},
        hexes_by_key={item.key: item for item in hexes if item.key},
        hexes_by_code={item.code: item for item in hexes if item.code},
        powerups_by_code={item.code: item for item in powerups if item.code},
    )


def decode_team_url(url: str, catalog: RuntimeCatalog) -> dict[str, Any]:
    """Port DataTFT's fixed-width team-builder decoder without executing JavaScript."""

    encoded = unquote(urlparse(url).path.rsplit("/", 1)[-1]).replace(" ", "")
    encoded = re.sub(r"s\d{1,2}$", "", encoded).rstrip(".")
    cursor = 0
    units: list[dict[str, Any]] = []
    hexes: list[dict[str, Any]] = []
    issues: list[str] = []
    map_lookup = {code: index for index, code in enumerate(MAP_CODES)}
    while cursor < len(encoded):
        if encoded[cursor] == "|":
            cursor += 1
            while cursor < len(encoded):
                token = encoded[cursor : cursor + 2]
                entity = catalog.hexes_by_code.get(token)
                hexes.append(_entity_ref(entity, token))
                if entity is None:
                    issues.append(f"unknown hex token {token!r}")
                cursor += 2
            break
        map_token = encoded[cursor : cursor + 1]
        cursor += 1
        board_index = map_lookup.get(map_token)
        hero_token = encoded[cursor : cursor + 2]
        cursor += 2
        hero = catalog.heroes_by_code.get(hero_token)
        if board_index is None or board_index > 27:
            issues.append(f"invalid board token {map_token!r}")
        if hero is None:
            issues.append(f"unknown hero token {hero_token!r}")
        unit: dict[str, Any] = {
            "board_index": board_index,
            # The frontend renders mapHeroItems[0:28] in a seven-column row-major grid.
            "x": board_index % 7 if board_index is not None and board_index <= 27 else None,
            "y": board_index // 7 if board_index is not None and board_index <= 27 else None,
            **_entity_ref(hero, hero_token, "hero"),
            "stars": None,
            "is_main": None,
            "replace_id": None,
            "replace_key": None,
            "extra_traits": [],
            "items": [],
            "powerups": {},
        }
        if cursor < len(encoded) and encoded[cursor].isdigit():
            count = int(encoded[cursor])
            cursor += 1
            for _ in range(count):
                token = encoded[cursor : cursor + 2]
                cursor += 2
                entity = catalog.items_by_code.get(token)
                unit["items"].append(_entity_ref(entity, token, "item"))
                if entity is None:
                    issues.append(f"unknown item token {token!r}")
        if cursor < len(encoded) and encoded[cursor] == "~":
            cursor += 1
            count = (
                int(encoded[cursor]) if cursor < len(encoded) and encoded[cursor].isdigit() else 0
            )
            cursor += 1
            for _ in range(count):
                position_token = encoded[cursor : cursor + 1]
                token = encoded[cursor + 1 : cursor + 3]
                cursor += 3
                entity = catalog.powerups_by_code.get(token)
                position = POWERUP_POSITIONS.get(position_token)
                if position is None:
                    issues.append(f"unknown powerup position {position_token!r}")
                else:
                    unit["powerups"].setdefault(position, []).append(
                        _entity_ref(entity, token, "powerup")
                    )
                if entity is None:
                    issues.append(f"unknown powerup token {token!r}")
        if cursor < len(encoded) and encoded[cursor] == "*":
            cursor += 1
            if cursor < len(encoded) and encoded[cursor].isdigit():
                unit["stars"] = int(encoded[cursor])
                cursor += 1
        if cursor < len(encoded) and encoded[cursor] == "_":
            cursor += 1
            token = encoded[cursor : cursor + 2]
            cursor += 2
            replacement = catalog.heroes_by_code.get(token)
            unit["replace_id"] = replacement.external_id if replacement else None
            unit["replace_key"] = replacement.key if replacement else None
            if cursor < len(encoded):
                cursor += 1  # upstream replacement price marker
        while cursor < len(encoded) and encoded[cursor] == "^":
            cursor += 1
            if cursor < len(encoded):
                marker = encoded[cursor]
                index = int(marker, 36) if marker.isalnum() else -1
                if 0 <= index < len(EXTRA_TRAIT_KEYS):
                    unit["extra_traits"].append(EXTRA_TRAIT_KEYS[index])
                else:
                    unit["extra_traits"].append({"unresolved_raw_token": marker})
                    issues.append(f"unknown extra-trait token {marker!r}")
                cursor += 1
        if cursor < len(encoded) and encoded[cursor] == "-":
            cursor += min(2, len(encoded) - cursor)
            unit["is_main"] = True
        unit["resolution_status"] = (
            "EXACT" if hero is not None and board_index is not None else "UNRESOLVED"
        )
        unit["unresolved_raw_token"] = (
            None if unit["resolution_status"] == "EXACT" else map_token + hero_token
        )
        units.append(unit)
    return {"units": units, "hexes": hexes, "issues": issues}


def canonical_knowledge(
    summary: dict[str, Any], detail: dict[str, Any], catalog: RuntimeCatalog
) -> dict[str, Any]:
    """Build the semantic V2 knowledge included in hashing and relational storage."""

    final = decode_team_url(str(detail.get("teamUrl") or ""), catalog)
    early = decode_team_url(str(detail.get("earlyTeamUrl") or ""), catalog)
    plans: list[dict[str, Any]] = []
    for holder in detail.get("equips") or []:
        if not isinstance(holder, dict):
            continue
        hero_id = str(holder.get("heroId") or "")
        hero = catalog.heroes_by_id.get(hero_id)
        for plan_index, plan in enumerate(holder.get("equips") or [], start=1):
            if not isinstance(plan, list):
                continue
            for slot_index, item_id in enumerate(plan, start=1):
                item = catalog.items_by_id.get(str(item_id))
                plans.append(
                    {
                        "holder": _entity_ref(hero, hero_id, "hero"),
                        "plan_index": plan_index,
                        "slot_index": slot_index,
                        "item": _entity_ref(item, str(item_id), "item"),
                    }
                )
    conditions: list[dict[str, Any]] = []
    raw_conditions = detail.get("conditions")
    if isinstance(raw_conditions, str):
        try:
            raw_conditions = json.loads(raw_conditions)
        except json.JSONDecodeError:
            raw_conditions = {}
    if isinstance(raw_conditions, dict):
        for condition_kind, key in (
            ("INCLUSION", "filterOptionResults"),
            ("EXCLUSION", "exclusionFilterOptionResults"),
        ):
            for entry in raw_conditions.get(key) or []:
                if isinstance(entry, dict):
                    conditions.append(
                        {
                            "condition_kind": condition_kind,
                            "condition_type": entry.get("type"),
                            "id": _text(entry.get("id")),
                            "key": _text(entry.get("key")),
                            "name": _text(entry.get("name")),
                            "level_value": _text(entry.get("levelValue")),
                            "trait_level_text": _text(entry.get("traitLevelText")),
                            "trait_level_idx": _text(entry.get("traitLevelIdx")),
                            "carry_hero_key": _text(entry.get("carryHeroKey")),
                            "carry_hero_value": _text(entry.get("carryHeroValue")),
                            "raw_entry_json": json.dumps(
                                entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                            ),
                        }
                    )
    hexes = [{**item, "source_kind": "TEAM_URL", "rank": None} for item in final["hexes"]]
    sections: list[dict[str, Any]] = []
    for section_type, key in (
        ("starting_conditions", "open"),
        ("core", "core"),
        ("early_game", "earlyDesc"),
        ("mid_game", "midDesc"),
        ("late_game", "finalDesc"),
    ):
        if _text(detail.get(key)):
            sections.append(
                {
                    "section_type": section_type,
                    "text": _text(detail[key]),
                    "ordinal": 0,
                    "labels": [],
                    "author": None,
                    "upstream_updated_at": _text(detail.get("updateTime")),
                    "raw_entry_json": None,
                }
            )
    for tip_index, tip in enumerate(detail.get("tips") or []):
        if not isinstance(tip, dict) or not _text(tip.get("description")):
            continue
        labels = [item for item in tip.get("labels") or [] if isinstance(item, dict)]
        sections.append(
            {
                "section_type": "tip",
                "text": _text(tip["description"]),
                "ordinal": tip_index,
                "labels": labels,
                "author": _text(tip.get("author")),
                "upstream_updated_at": _text(tip.get("updateTime")),
                "raw_entry_json": json.dumps(
                    tip, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        for label in labels:
            if label.get("type") == "hex":
                entity = catalog.hexes_by_id.get(str(label.get("id") or ""))
                entity = entity or catalog.hexes_by_key.get(str(label.get("key") or ""))
                ref = _entity_ref(entity, str(label.get("id") or label.get("key") or ""), "hex")
                hexes.append({**ref, "source_kind": "TIP_LABEL", "rank": None})
    representative_id = str(summary.get("heroId") or detail.get("titleApi") or "")
    representative = catalog.heroes_by_id.get(representative_id)
    return {
        "metadata": {
            "difficulty": detail.get("difficulty"),
            "tags": detail.get("tags") or [],
            "upstream_updated_at": _text(detail.get("updateTime")),
            "representative_hero": _entity_ref(representative, representative_id, "hero"),
        },
        "boards": {"FINAL": final, "EARLY": early},
        "item_plans": plans,
        "conditions": conditions,
        "hexes": hexes,
        "strategy_sections": sections,
    }


def _entities(value: Any, id_key: str, name_key: str) -> list[RuntimeEntity]:
    if not isinstance(value, list):
        return []
    return [
        RuntimeEntity(
            str(item[id_key]),
            _text(item.get("key")),
            _text(item.get(name_key)),
            _text(item.get("code")),
        )
        for item in value
        if isinstance(item, dict) and item.get(id_key) is not None
    ]


def _flatten(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [child for item in value for child in (item if isinstance(item, list) else [item])]


def _entity_ref(
    entity: RuntimeEntity | None, raw_token: str, kind: str = "entity"
) -> dict[str, Any]:
    return {
        f"external_{kind}_id": entity.external_id if entity else None,
        f"external_{kind}_key": entity.key if entity else None,
        "display_name": entity.name if entity else None,
        "resolution_status": "EXACT" if entity else "UNRESOLVED",
        "unresolved_raw_token": None if entity else raw_token,
    }


def _text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None
