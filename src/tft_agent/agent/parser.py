"""Input parsing boundary for factual game state."""

import re
from collections.abc import Mapping
from typing import Any

from tft_agent.core.schemas import GameState, StreakState, StreakType, UnitState


class GameStateParser:
    """Parse already-structured input without adding inferred facts."""

    def parse(self, payload: Mapping[str, Any]) -> GameState:
        return GameState.model_validate(payload)

    def parse_text(self, text: str) -> GameState:
        """Extract only explicitly stated match and strategy-context facts."""

        values: dict[str, Any] = {"user_question": text}
        patterns = {
            "level": r"(?:现在)?\s*(\d{1,2})\s*级",
            "hp": r"(\d{1,3})\s*血",
            "gold": r"(\d{1,3})\s*(?:金币|金)(?!币卡)",
            "target_cost": r"(?:目标是|找|抽|要)\s*(?:一张|一个)?\s*(\d)\s*费",
            "stage": r"(?<!\d)(\d+-\d+)(?!\d)",
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values[field] = int(match.group(1)) if field != "stage" else match.group(1)
        champion = re.search(r"(?:想找|找|目标是)\s*([A-Za-z][A-Za-z0-9' -]{1,30})", text)
        if champion:
            values["target_champion"] = champion.group(1).strip()
        comp = re.search(
            r"(?:正在玩|玩的是|阵容(?:是|叫))\s*[【「]?([^】」,，。；;]{2,40}?)(?:阵容)?"
            r"(?=[,，。；;]|$)",
            text,
            re.IGNORECASE,
        )
        if comp:
            comp_name = comp.group(1).strip()
            if comp_name.casefold() not in {"这套", "这个", "当前", "这套阵容", "这个阵容"}:
                values["comp_name"] = comp_name
        elif named_strategy := re.search(
            r"\b([A-Z][A-Za-z]+(?:\s+(?:[A-Z][A-Za-z0-9]+|\d+))+)"
            r"(?=[^\n,，。；;]{0,40}(?:怎么玩|怎么运营|如何运营|攻略))",
            text,
        ):
            values["comp_name"] = named_strategy.group(1).strip()
        core = re.search(
            r"(?:核心(?:牌|单位)\s*(?:是|有|：|:)?|核心\s*(?:是|有|：|:))"
            r"\s*([^,，。；;]+)",
            text,
            re.IGNORECASE,
        )
        if core:
            values["core_units"] = [
                unit.strip()
                for unit in re.split(r"[、/]|\s+(?:and|和)\s+", core.group(1))
                if unit.strip()
            ]
        unit_labels: dict[str, tuple[str, ...]] = {
            "board": ("棋盘", "场上", "board"),
            "bench": ("备战席", "替补席", "bench"),
        }
        for field, labels in unit_labels.items():
            segment = _labeled_segment(text, labels)
            if segment:
                values[field] = _parse_units(segment)
        fact_labels: dict[str, tuple[str, ...]] = {
            "item_components": ("散件", "装备散件", "components", "item components"),
            "completed_items": ("成装", "完整装备", "completed items"),
            "augments": ("强化符文", "强化", "augments"),
        }
        for field, labels in fact_labels.items():
            segment = _labeled_segment(text, labels)
            if segment:
                values[field] = _parse_names(segment)
        offered = re.search(
            r"(?:现在有|可选|三选一[:：]?)\s*(.{2,80}?)\s*"
            r"(?:这三个|三个)?(?:海克斯|强化)(?:怎么选|选哪个|如何选)",
            text,
        )
        if offered:
            values["augment_choices"] = _parse_choice_names(offered.group(1))[:3]
        special = [name for name in ("铲子", "金锅锅", "锅") if name in text]
        if special:
            values["special_components"] = list(dict.fromkeys(special))
        emblems = []
        for raw in re.findall(r"([\u4e00-\u9fffA-Za-z]{1,12}(?:转|纹章))", text):
            name = re.split(r"和|及|有|拿到|一个", raw)[-1]
            if name:
                emblems.append(name)
        if emblems:
            values["emblems"] = list(dict.fromkeys(emblems))
        streak = re.search(r"(?:当前)?\s*(\d{1,2})\s*(连胜|连败)", text, re.IGNORECASE)
        if streak:
            values["streak"] = StreakState(
                type=StreakType.WIN if streak.group(2) == "连胜" else StreakType.LOSS,
                length=int(streak.group(1)),
            )
        else:
            streak = re.search(r"\b(\d{1,2})[- ]game\s+(win|loss)\s+streak\b", text, re.I)
            if streak:
                values["streak"] = StreakState(
                    type=StreakType(streak.group(2).lower()), length=int(streak.group(1))
                )
        return GameState.model_validate(values)


def _labeled_segment(text: str, labels: tuple[str, ...]) -> str | None:
    label = "|".join(re.escape(item) for item in labels)
    match = re.search(
        rf"(?:^|[；;。\n])\s*(?:{label})\s*(?:是|有|为|：|:)?\s*([^；;。\n]+)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _parse_names(segment: str) -> list[str]:
    return [item.strip() for item in re.split(r"[、,，/]", segment) if item.strip()]


def _parse_choice_names(segment: str) -> list[str]:
    cleaned = re.sub(r"^(?:我)?", "", segment.strip())
    return [
        item.strip()
        for item in re.split(r"[、,，/]|和|\s+", cleaned)
        if item.strip()
    ]


def _parse_units(segment: str) -> list[UnitState]:
    units: list[UnitState] = []
    for raw in _split_outside_brackets(segment):
        item_match = re.search(r"[（(]([^()（）]+)[）)]", raw)
        items = _parse_names(item_match.group(1)) if item_match else []
        without_items = re.sub(r"[（(][^()（）]+[）)]", "", raw).strip()
        star_match = re.search(r"(?:\s|^)([1-4])\s*(?:星|stars?\b)", without_items, re.I)
        star_level = int(star_match.group(1)) if star_match else 1
        name = re.sub(r"(?:\s|^)[1-4]\s*(?:星|stars?\b)", "", without_items, flags=re.I)
        name = name.strip()
        if name:
            units.append(UnitState(name=name, star_level=star_level, items=items))
    return units


def _split_outside_brackets(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "(（":
            depth += 1
        elif char in ")）":
            depth = max(0, depth - 1)
        elif char in ",，、/" and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]
