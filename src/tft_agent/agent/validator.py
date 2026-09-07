"""Application-level checks that are intentionally separate from field validation."""

import re

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

    def missing_strategy_context(self, game_state: GameState) -> list[str]:
        """Require identity facts when a strategy question only points at an unnamed comp."""

        question = game_state.user_question or ""
        unnamed_reference = re.search(
            r"(?:这套|这个|当前)(?:阵容|体系)|\bthis (?:comp|composition)\b",
            question,
            re.IGNORECASE,
        )
        if unnamed_reference and game_state.comp_name is None and not game_state.core_units:
            return ["comp_name or core_units"]
        return []

    def rich_state_issues(self, game_state: GameState) -> list[str]:
        """Report only deterministic contradictions in an otherwise valid state."""

        issues: list[str] = []
        if game_state.level is not None and len(game_state.board) > game_state.level:
            issues.append("board unit count exceeds level")
        return issues
