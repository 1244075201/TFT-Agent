"""Product-mode rules for isolating conversational and match state."""

from __future__ import annotations

from dataclasses import dataclass

from tft_agent.app.pipeline import ProductMode


@dataclass(frozen=True, slots=True)
class SessionScope:
    session_id: str
    uses_game_state: bool
    accepts_screenshot: bool


def session_scope(session_id: str, product_mode: ProductMode) -> SessionScope:
    """Return the storage namespace and capabilities for a product surface."""
    if product_mode == ProductMode.GENERAL_GUIDE:
        return SessionScope(
            session_id=f"{session_id}:general-guide",
            uses_game_state=False,
            accepts_screenshot=False,
        )
    return SessionScope(session_id=session_id, uses_game_state=True, accepts_screenshot=True)
