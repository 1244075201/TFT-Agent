"""Input parsing boundary for factual game state."""

from collections.abc import Mapping
from typing import Any

from tft_agent.core.schemas import GameState


class GameStateParser:
    """Parse already-structured input without adding inferred facts."""

    def parse(self, payload: Mapping[str, Any]) -> GameState:
        return GameState.model_validate(payload)
