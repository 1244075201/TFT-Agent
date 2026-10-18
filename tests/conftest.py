"""Unit tests never call the application planner's paid provider implicitly."""

from __future__ import annotations

import pytest

from tft_agent.app.semantic_retrieval import PlannerOutcome, safe_fallback_plan
from tft_agent.core.config import Settings
from tft_agent.core.schemas import GameState


class OfflineGuidePlanner:
    def __init__(self, settings: Settings) -> None:
        pass

    async def plan(
        self,
        query: str,
        game_state: GameState | None = None,
        *,
        state_aware: bool = False,
    ) -> PlannerOutcome:
        return PlannerOutcome(
            plan=safe_fallback_plan(),
            model="offline-unit-planner",
            latency_ms=0,
            used_fallback=True,
        )

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def offline_application_planner(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.get_closest_marker("real_llm"):
        return
    monkeypatch.setenv("TFT_AGENT_KNOWLEDGE_SEARCH_API_KEY", "")
    # Tests for semantic contracts explicitly replace this with their own planner.
    # Provider adapter tests import the real class from semantic_retrieval directly.
    monkeypatch.setattr("tft_agent.app.service.RetrievalPlannerClient", OfflineGuidePlanner)
