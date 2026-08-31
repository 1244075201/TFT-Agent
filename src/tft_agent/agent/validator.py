"""Application-level checks that are intentionally separate from field validation."""

from tft_agent.core.schemas import GameState


class GameStateValidator:
    """Report missing context useful for decision making without inventing defaults."""

    def missing_basic_context(self, game_state: GameState) -> list[str]:
        fields = []
        if game_state.patch is None:
            fields.append("patch")
        if game_state.stage is None:
            fields.append("stage")
        if game_state.user_question is None:
            fields.append("user_question")
        return fields
