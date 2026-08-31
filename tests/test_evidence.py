from datetime import UTC, datetime

from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import GameState, ToolCall, ToolCallRecord, ToolResult
from tft_agent.llm.evidence import build_evidence_context, evidence_field_paths


def test_evidence_context_separates_user_and_tool_facts() -> None:
    now = datetime.now(UTC)
    state = AgentState(
        game_state=GameState(level=7, hp=35, gold=40, target_cost=4),
        goal="decide",
        tool_history=[
            ToolCallRecord(
                tool_call=ToolCall(id="1", name="get_shop_odds", arguments={"level": 7}),
                result=ToolResult(
                    tool_call_id="1",
                    success=True,
                    data={"level": 7, "odds": {"cost_4": 0.10}},
                ),
                started_at=now,
                finished_at=now,
                latency_ms=0,
            )
        ],
    )

    context = build_evidence_context(state)
    paths = evidence_field_paths(context)

    assert context["user_facts"]["gold"] == 40
    assert context["tool_evidence"][0]["arguments"] == {"level": 7}
    assert "user_facts.hp" in paths
    assert "tool_evidence[0].get_shop_odds.data.odds.cost_4" in paths
