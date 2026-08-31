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
        """Extract only the small set of explicit V0.2 facts from Chinese input."""

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
        return GameState.model_validate(values)
