from datetime import UTC, datetime

from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    GameState,
    QualitativeClaimStatus,
    StrategyRecommendation,
    ToolCall,
    ToolCallRecord,
    ToolResult,
)
from tft_agent.llm.qualitative_claims import QualitativeRecommendationValidator


def strategy_state(
    *,
    source_type: str = "community_strategy",
    document_type: str = "strategy_guide",
    patch: str = "18.1",
    comp_names: list[str] | None = None,
) -> AgentState:
    call = ToolCall(
        id="search-1",
        name="search_game_knowledge",
        arguments={"strategy_comp_name": "Blossom Ahri", "strategy_topic": "core_quality"},
    )
    result = ToolResult(
        tool_call_id="search-1",
        success=True,
        data={
            "results": [
                {
                    "chunk_id": "blossom:2",
                    "text": "Improve Ahri and Sett before leveling.",
                    "source_type": source_type,
                    "document_type": document_type,
                    "game_patch": patch,
                    "strategy_comp_names": comp_names or ["Blossom Ahri"],
                    "strategy_topics": ["core_quality"],
                }
            ]
        },
    )
    state = AgentState(
        game_state=GameState(comp_name="Blossom Ahri"),
        goal="Should I roll?",
        current_patch="18.1",
    )
    now = datetime.now(UTC)
    state.tool_history.append(
        ToolCallRecord(
            tool_call=call,
            result=result,
            started_at=now,
            finished_at=now,
            latency_ms=0,
        )
    )
    return state


def recommendation(ref: str = "knowledge.search-1.blossom:2") -> StrategyRecommendation:
    return StrategyRecommendation(
        text="Improve the core pair before leveling.", evidence_refs=[ref]
    )


def test_fake_strategy_reference_is_invalid() -> None:
    result = QualitativeRecommendationValidator().validate(
        recommendation("knowledge.search-1.fake_chunk"), strategy_state()
    )
    assert result.status == QualitativeClaimStatus.INVALID_REFERENCE


def test_existing_chunk_field_reference_is_supported() -> None:
    result = QualitativeRecommendationValidator().validate(
        recommendation("knowledge.search-1.blossom:2.text"), strategy_state()
    )
    assert result.status == QualitativeClaimStatus.SUPPORTED


def test_wrong_strategy_entity_is_irrelevant() -> None:
    result = QualitativeRecommendationValidator().validate(
        recommendation(), strategy_state(comp_names=["Solar Kayle"])
    )
    assert result.status == QualitativeClaimStatus.IRRELEVANT_EVIDENCE


def test_official_mechanic_cannot_support_strategy_recommendation() -> None:
    result = QualitativeRecommendationValidator().validate(
        recommendation(),
        strategy_state(source_type="riot_official", document_type="set_overview"),
    )
    assert result.status == QualitativeClaimStatus.WRONG_SOURCE_TYPE
