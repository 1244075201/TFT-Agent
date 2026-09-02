"""Input parsing boundary for factual game state."""

import re
from collections.abc import Mapping
from typing import Any

from tft_agent.core.schemas import GameState


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
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z0-9]+)+)\s*(?:现在)?"
            r"(?:怎么玩|怎么运营|如何运营|攻略)",
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
        return GameState.model_validate(values)
