"""GUI-independent request contracts for the two desktop workspaces."""

from __future__ import annotations

from pathlib import Path

from tft_agent.core.schemas import ChatRequest, GameState

GUIDE_SESSION_ID = "desktop-guide"
MATCH_SESSION_ID = "desktop-match"


def guide_request(question: str) -> ChatRequest:
    return ChatRequest(
        text=question,
        session_id=GUIDE_SESSION_ID,
        product_mode="general_guide",
    )


def match_request(
    question: str, state: GameState, screenshot_path: Path | None = None
) -> ChatRequest:
    return ChatRequest(
        text=question,
        session_id=MATCH_SESSION_ID,
        product_mode="match_guidance",
        confirmed_game_state=state,
        screenshot_path=screenshot_path,
    )


def game_state_summary(state: GameState) -> str:
    values = [
        state.stage,
        f"{state.level}级" if state.level is not None else None,
        f"{state.gold}金" if state.gold is not None else None,
        f"{state.hp}血" if state.hp is not None else None,
        state.comp_name,
    ]
    return " · ".join(value for value in values if value) or "尚未记录"
