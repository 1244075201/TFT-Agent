from datetime import UTC, datetime

from tft_agent.agent.state import AgentState
from tft_agent.core.evidence import build_evidence_alias_registry, build_evidence_registry
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


def test_evidence_context_separates_community_strategy_from_official_knowledge() -> None:
    now = datetime.now(UTC)
    chunks = [
        {"chunk_id": "official", "title": "Official", "source_type": "riot_official"},
        {"chunk_id": "guide", "title": "Guide", "source_type": "community_strategy"},
    ]
    state = AgentState(
        game_state=GameState(comp_name="Blossom Ahri", core_units=["Ahri", "Sett"]),
        goal="decide",
        tool_history=[
            ToolCallRecord(
                tool_call=ToolCall(id="rag", name="search_game_knowledge", arguments={}),
                result=ToolResult(tool_call_id="rag", success=True, data={"results": chunks}),
                started_at=now,
                finished_at=now,
                latency_ms=0,
            )
        ],
    )

    context = build_evidence_context(state)

    assert context["user_facts"]["comp_name"] == "Blossom Ahri"
    assert context["official_knowledge"] == [chunks[0]]
    assert context["community_strategy_evidence"] == [chunks[1]]
    assert context["structured_evidence"] == []
    aliases = build_evidence_alias_registry(state)
    assert "knowledge.rag.official" in aliases.values()
    assert "knowledge.rag.guide" in aliases.values()
    assert not any(ref.startswith("tool.rag.data.results") for ref in aliases.values())


def test_nested_rich_state_is_registered_and_aliased() -> None:
    state = AgentState(
        game_state=GameState.model_validate(
            {
                "board": [{"name": "Ahri", "star_level": 2, "items": ["Blue Buff"]}],
                "bench": [{"name": "Sett"}],
                "item_components": ["Tear"],
                "completed_items": ["Warmog's Armor"],
                "augments": ["Jeweled Lotus"],
                "streak": {"type": "loss", "length": 3},
            }
        ),
        goal="decide",
    )

    registry = build_evidence_registry(state)
    aliases = build_evidence_alias_registry(state)
    context = build_evidence_context(state)

    assert registry["game_state.board.0.name"] == "Ahri"
    assert registry["game_state.board.0.items.0"] == "Blue Buff"
    assert registry["game_state.streak.length"] == 3
    assert "game_state.completed_items.0" in aliases.values()
    assert "user_facts.board.0.name" in evidence_field_paths(context)
