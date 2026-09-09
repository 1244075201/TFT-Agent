"""Typed understanding, retrieval planning, clean evidence, and development traces."""
from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import Field

from tft_agent.agent.state import AgentMessage
from tft_agent.app.entity_resolution import EntityResolution, ResolvedEntity
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
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    target_comp: ResolvedEntity | None = None
    browse_intent: bool = False


class RetrievalPlan(StrictModel):
    game_patch: str
    set_number: int
    set_name: str
    target_comp: ResolvedEntity | None = None
    entities: list[ResolvedEntity] = Field(default_factory=list)
    knowledge_needs: list[str] = Field(default_factory=list)
    candidate_selection_inputs: list[ResolvedEntity] = Field(default_factory=list)
    retrieve_all_comps: bool = False
    retrieve_strategy_text: bool = False


class EvidenceRecord(StrictModel):
    evidence_id: str
    source: str
    requested_patch: str
    source_patch: str | None = None
    patch_match: str
    entity_type: str
    canonical_entity_id: str
    display_name: str
    section: str
    semantic_content: str | list[str] | dict[str, Any]
    relevance_score: float = Field(ge=0, le=1)
    url: str


class RetrievedKnowledge(StrictModel):
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)


class RepositoryCall(StrictModel):
    operation: str
    filters: dict[str, Any] = Field(default_factory=dict)
    result_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class RetrievalTrace(StrictModel):
    original_query: str
    resolved_entities: list[ResolvedEntity]
    understanding: QueryStateUnderstanding
    retrieval_plan: RetrievalPlan
    repository_calls: list[RepositoryCall] = Field(default_factory=list)
    retrieved_knowledge: RetrievedKnowledge
    knowledge_gaps: list[str] = Field(default_factory=list)
    clean_context_summary: str = ""
    llm_call_count: int = Field(default=0, ge=0)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)


class AnswerContext(StrictModel):
    question: str
    understanding: QueryStateUnderstanding
    game_state: GameState
    history: list[AgentMessage] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    context_gaps: list[str] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)

    def readable_summary(self) -> str:
        sections: dict[str, list[str]] = {}
        for item in self.evidence:
            sections.setdefault(item.section, []).append(
                f"{item.display_name}: {item.semantic_content} "
                f"[来源：{item.source}｜版本：{item.source_patch or '未标注'}｜"
                f"区块：{item.section}]"
            )
        blocks = [
            "CURRENT GAME STATE\n" + self.game_state.model_dump_json(
                exclude={"user_question", "active_comp_id", "board", "bench", "shop"},
                exclude_none=True,
            ),
            *[f"{name.upper()}\n" + "\n".join(values)
              for name, values in sections.items()],
        ]
        if self.context_gaps:
            blocks.append("CONTEXT GAPS\n- " + "\n- ".join(self.context_gaps))
        if self.knowledge_gaps:
            blocks.append("KNOWLEDGE GAPS\n- " + "\n- ".join(self.knowledge_gaps))
        return "\n\n".join(blocks)


_DECISION_TERMS = (
    "怎么玩", "如何", "为什么", "推荐", "比较", "哪个好", "选择", "怎么选", "选哪个",
    "判断", "该不该", "要不要", "怎么办", "转阵", "变阵", "运营", "过渡", "搜牌",
    "给谁", "适合", "建议", "玩什么", "调整", "should", "recommend", "why", "how",
    "choose", "compare", "versus", " vs ",
)
_BROWSE_TERMS = ("强势阵容", "推荐几套", "s级阵容", "阵容榜", "meta阵容", "版本阵容")
_FACT_TERMS = ("核心棋子", "有哪些棋子", "阵容成员", "装备是什么", "定义", "效果是什么")


def understand_query(
    text: str,
    state: GameState,
    mode: ProductMode,
    resolution: EntityResolution | None = None,
) -> QueryStateUnderstanding:
    entities = resolution.entities if resolution else []
    target = next((item for item in entities if item.entity_type == "composition"), None)
    lowered = text.casefold()
    decision_terms = [term for term in _DECISION_TERMS if term in lowered]
    browse = mode == ProductMode.COMP_BROWSER or any(term in lowered for term in _BROWSE_TERMS)
    factual = any(term in lowered for term in _FACT_TERMS) and not decision_terms
    return QueryStateUnderstanding(
        kind=QueryKind.DETERMINISTIC_LOOKUP if factual or (browse and not decision_terms)
        else QueryKind.STRATEGY_DECISION,
        product_mode=mode,
        decision_terms=decision_terms,
        resolved_entities=entities,
        target_comp=target,
        browse_intent=browse,
    )


def build_retrieval_plan(
    understanding: QueryStateUnderstanding,
    state: GameState,
    patch: str,
    set_number: int,
    set_name: str,
) -> RetrievalPlan:
    target = understanding.target_comp
    other = [item for item in understanding.resolved_entities
             if item.entity_type != "composition"]
    decision = understanding.kind == QueryKind.STRATEGY_DECISION
    needs: list[str] = []
    if target:
        needs.extend(["composition", "core_units", "items"])
        if decision:
            needs.extend(["early_game", "mid_game", "late_game", "leveling"])
    if any(item.entity_type == "emblem" for item in other):
        needs.extend(["emblem_definition", "key_traits", "composition_emblem_advice",
                      "transition", "flex", "conditional_advice"])
    if any(item.entity_type == "augment" for item in other):
        needs.extend(["augment_definition", "conditional_advice"])
    if target is None and decision:
        needs.extend(["general_early_game", "item_flexibility", "early_slam", "transition"])
    return RetrievalPlan(
        game_patch=patch, set_number=set_number, set_name=set_name, target_comp=target,
        entities=other, knowledge_needs=list(dict.fromkeys(needs)),
        candidate_selection_inputs=[item for item in other
                                    if item.entity_type in {"item", "augment", "champion"}],
        retrieve_all_comps=understanding.browse_intent,
        retrieve_strategy_text=decision,
    )


def apply_resolution(state: GameState, resolution: EntityResolution, question: str) -> GameState:
    update: dict[str, Any] = {"user_question": question}
    comp = next((item for item in resolution.entities if item.entity_type == "composition"), None)
    if comp:
        update.update(comp_name=comp.display_name, active_comp_id=comp.canonical_id)
    augments = [item.display_name for item in resolution.entities if item.entity_type == "augment"]
    if augments and re.search(r"选哪个|三选一|海克斯|强化", question):
        update["augment_choices"] = augments[:3]
    emblems = [item.display_name for item in resolution.entities if item.entity_type == "emblem"]
    if emblems:
        update["emblems"] = emblems
    items = [item.display_name for item in resolution.entities if item.entity_type == "item"]
    if items:
        update["item_components"] = items
    return state.model_copy(update=update)


def comp_evidence(record: CompStatsRecord, requested_patch: str) -> list[EvidenceRecord]:
    base = dict(source="DataTFT", requested_patch=requested_patch,
                source_patch=record.snapshot.game_patch, patch_match="exact",
                entity_type="composition", canonical_entity_id=record.comp_id,
                display_name=record.name, relevance_score=1.0, url=record.snapshot.source_url)
    sections: list[tuple[str, str | list[str]]] = []
    if record.core_units:
        sections.append(("core_units", record.core_units))
    clean_items = [_clean_item(value) for value in record.recommended_items]
    clean_items = [value for value in clean_items if value]
    if clean_items:
        sections.append(("items", clean_items))
    for value in record.level_plan:
        label, _, content = value.partition(":")
        section = {"前期": "early_game", "中期": "mid_game", "成型": "late_game",
                   "核心": "core", "开局条件": "starting_conditions"}.get(label, "operation")
        if content:
            sections.append((section, content))
    return [EvidenceRecord(
        evidence_id=_evidence_id("DataTFT", record.snapshot.game_patch, "composition",
                                 record.comp_id, section),
        section=section, semantic_content=content, **base,
    ) for section, content in sections]


def static_evidence(
    record: StaticEntityRecord, resolved: ResolvedEntity, requested_patch: str
) -> EvidenceRecord | None:
    description = _clean_template(record.description)
    if not description:
        return None
    section = "augment_definition" if resolved.entity_type == "augment" else (
        "emblem_definition" if resolved.entity_type == "emblem" else "item_definition"
    )
    return EvidenceRecord(
        evidence_id=_evidence_id("CommunityDragon", requested_patch, resolved.entity_type,
                                 resolved.canonical_id, section),
        source="CommunityDragon", requested_patch=requested_patch, source_patch=requested_patch,
        patch_match="exact", entity_type=resolved.entity_type,
        canonical_entity_id=resolved.canonical_id, display_name=resolved.display_name,
        section=section, semantic_content=description, relevance_score=1.0,
        url=record.source_url,
    )


def context_gaps(state: GameState) -> list[str]:
    values = {"target_comp": state.comp_name, "stage": state.stage, "level": state.level,
              "gold": state.gold, "hp": state.hp, "augments": state.augments,
              "augment_choices": state.augment_choices, "components": state.item_components,
              "completed_items": state.completed_items, "emblems": state.emblems,
              "key_units": state.core_units, "notes": state.notes}
    return [key for key, value in values.items() if value is None or value == [] or value == ""]


def _clean_item(value: str) -> str:
    owner, separator, identifiers = value.partition(":")
    if separator and re.fullmatch(r"[\d/]*", identifiers):
        return ""
    if not separator:
        return owner.strip()
    return value


def _clean_template(value: str) -> str:
    return re.sub(r"@[A-Za-z0-9_.*:]+@", "未公开数值", value).strip()


def _evidence_id(source: str, patch: str, kind: str, identifier: str, section: str) -> str:
    return ":".join((source, patch, kind, identifier, section))
