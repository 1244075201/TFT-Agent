"""Deterministic understanding, retrieval planning, and context construction."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import Field

from tft_agent.agent.state import AgentMessage
from tft_agent.core.schemas import GameState, StrictModel
from tft_agent.tft_mcp_server.schemas import CompStatsRecord, StaticEntityRecord


class QueryKind(StrEnum):
    DETERMINISTIC_LOOKUP = "deterministic_lookup"
    STRATEGY_DECISION = "strategy_decision"


class ProductMode(StrEnum):
    COMP_BROWSER = "comp_browser"
    MATCH_GUIDANCE = "match_guidance"


class QueryStateUnderstanding(StrictModel):
    kind: QueryKind
    product_mode: ProductMode
    decision_terms: list[str] = Field(default_factory=list)
    named_comp: str | None = None


class RetrievalPlan(StrictModel):
    game_patch: str
    comp_query: str | None = None
    retrieve_all_comps: bool = False
    retrieve_strategy_text: bool = False
    static_entity_names: list[str] = Field(default_factory=list)
    shop_odds_level: int | None = None
    reasons: list[str] = Field(default_factory=list)


class RetrievedKnowledge(StrictModel):
    comps: list[dict[str, Any]] = Field(default_factory=list)
    strategy_chunks: list[dict[str, Any]] = Field(default_factory=list)
    static_entities: list[dict[str, Any]] = Field(default_factory=list)
    shop_odds: dict[str, Any] | None = None


class AnswerContext(StrictModel):
    question: str
    understanding: QueryStateUnderstanding
    game_state: GameState
    history: list[AgentMessage] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan
    knowledge: RetrievedKnowledge
    context_gaps: list[str] = Field(default_factory=list)


_DECISION_TERMS = (
    "怎么玩", "如何", "为什么", "推荐", "比较", "哪个好", "选择", "怎么选",
    "判断", "该不该", "要不要", "怎么办", "转阵", "变阵", "运营", "搜牌",
    "给谁", "适合", "建议", "should", "recommend", "why", "how", "choose",
    "compare", "versus", " vs ",
)

_FACT_PATTERNS = (
    r"^(?:当前版本|这个版本)?(?:有哪些|有哪几套|列出|显示)(?:阵容)?$",
    r"^(?:阵容列表|阵容梯度|版本阵容|数据状态)$",
    r"^(?:查询|查看|显示).+(?:资料|数据|定义|详情)$",
)


def understand_query(text: str, state: GameState, mode: ProductMode) -> QueryStateUnderstanding:
    lowered = text.casefold().strip()
    decision_terms = [term for term in _DECISION_TERMS if term in lowered]
    explicit_fact = mode == ProductMode.COMP_BROWSER or any(
        re.search(pattern, text, re.IGNORECASE) for pattern in _FACT_PATTERNS
    )
    kind = (
        QueryKind.DETERMINISTIC_LOOKUP
        if explicit_fact and not decision_terms
        else QueryKind.STRATEGY_DECISION
    )
    return QueryStateUnderstanding(
        kind=kind,
        product_mode=mode,
        decision_terms=decision_terms,
        named_comp=state.comp_name,
    )


def build_retrieval_plan(
    understanding: QueryStateUnderstanding, state: GameState, patch: str
) -> RetrievalPlan:
    names = list(dict.fromkeys([
        *state.augments,
        *state.augment_choices,
        *state.item_components,
        *state.completed_items,
        *state.emblems,
        *state.special_components,
        *state.core_units,
    ]))
    decision = understanding.kind == QueryKind.STRATEGY_DECISION
    return RetrievalPlan(
        game_patch=patch,
        comp_query=state.comp_name,
        retrieve_all_comps=state.comp_name is None,
        retrieve_strategy_text=decision,
        static_entity_names=names,
        shop_odds_level=state.level if decision and state.target_cost is not None else None,
        reasons=[
            "current-patch compositions",
            *( ["decision requires strategy evidence"] if decision else [] ),
            *( ["state entities require definitions"] if names else [] ),
        ],
    )


def context_gaps(state: GameState) -> list[str]:
    fields = {
        "target_comp": state.comp_name,
        "stage": state.stage,
        "level": state.level,
        "gold": state.gold,
        "hp": state.hp,
        "augments": state.augments,
        "augment_choices": state.augment_choices,
        "components": state.item_components,
        "completed_items": state.completed_items,
        "emblems": state.emblems,
        "key_units": state.core_units,
        "notes": state.notes,
    }
    return [name for name, value in fields.items() if value is None or value == [] or value == ""]


def comp_payload(record: CompStatsRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def entity_payload(record: StaticEntityRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")
