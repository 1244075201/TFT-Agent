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
    GENERAL_GUIDE = "general_guide"
    COMP_BROWSER = "comp_browser"
    MATCH_GUIDANCE = "match_guidance"


class StateMutationIntent(StrEnum):
    RETAIN = "retain"
    REPLACE = "replace"
    CLEAR = "clear"
    CANDIDATES = "candidates"


class DecisionIntent(StrEnum):
    FULL_GUIDE = "full_guide"
    CURRENT_DECISION = "current_decision"
    COMPOSITION_COMPARISON = "composition_comparison"
    GENERAL_STRATEGY = "general_strategy"


class QueryStateUnderstanding(StrictModel):
    kind: QueryKind
    product_mode: ProductMode
    decision_terms: list[str] = Field(default_factory=list)
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    target_comp: ResolvedEntity | None = None
    candidate_comps: list[ResolvedEntity] = Field(default_factory=list)
    state_mutation: StateMutationIntent = StateMutationIntent.RETAIN
    decision_intent: DecisionIntent = DecisionIntent.GENERAL_STRATEGY
    browse_intent: bool = False


class RetrievalPlan(StrictModel):
    game_patch: str
    set_number: int
    set_name: str
    target_comp: ResolvedEntity | None = None
    candidate_comps: list[ResolvedEntity] = Field(default_factory=list)
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
    persisted_state_before: GameState = Field(default_factory=GameState)
    parsed_state: GameState = Field(default_factory=GameState)
    game_state_after_merge: GameState = Field(default_factory=GameState)
    injected_history: list[AgentMessage] = Field(default_factory=list)


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
        facts = _user_facts(self.game_state)
        conditions = [item for item in self.evidence if item.section == "starting_conditions"]
        recommendations = [item for item in self.evidence
                           if item.section != "starting_conditions"]
        blocks = [
            "USER QUESTION\n" + self.question.strip(),
            "USER FACTS\n" + ("\n".join(f"- {key}: {value}" for key, value in facts.items())
                                if facts else "- No structured facts provided"),
        ]
        if self.history:
            lines = [f"{'User' if item.role == 'user' else 'Assistant'}: {item.content}"
                     for item in self.history]
            blocks.append(
                "RECENT CONVERSATION (auxiliary only; never overrides USER FACTS)\n"
                + "\n".join(lines)
            )
        if self.understanding.candidate_comps:
            blocks.append("CANDIDATE COMPOSITIONS\n- " + "\n- ".join(
                item.display_name for item in self.understanding.candidate_comps
            ))
        if conditions:
            blocks.append(
                "RETRIEVED STRATEGY CONDITIONS (applicability only; not user facts)\n"
                + "\n".join(_format_evidence(item) for item in conditions)
            )
        if recommendations:
            blocks.append(
                "RETRIEVED STRATEGY RECOMMENDATIONS\n"
                + _grouped_evidence(recommendations, self.understanding.candidate_comps)
            )
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
    compositions = _ordered_compositions(text, entities)
    lowered = text.casefold()
    decision_terms = [term for term in _DECISION_TERMS if term in lowered]
    browse = mode == ProductMode.COMP_BROWSER or any(term in lowered for term in _BROWSE_TERMS)
    factual = any(term in lowered for term in _FACT_TERMS) and not decision_terms
    mutation, target, candidates = _composition_semantics(text, compositions)
    if len(candidates) > 1:
        intent = DecisionIntent.COMPOSITION_COMPARISON
    elif re.search(r"d牌|搜牌|上\s*9|升\s*9|硬扛|要不要", lowered, re.I):
        intent = DecisionIntent.CURRENT_DECISION
    elif any(term in lowered for term in ("怎么玩", "完整攻略", "完整运营", "攻略")):
        intent = DecisionIntent.FULL_GUIDE
    else:
        intent = DecisionIntent.GENERAL_STRATEGY
    return QueryStateUnderstanding(
        kind=QueryKind.DETERMINISTIC_LOOKUP if factual or (browse and not decision_terms)
        else QueryKind.STRATEGY_DECISION,
        product_mode=mode,
        decision_terms=decision_terms,
        resolved_entities=entities,
        target_comp=target,
        candidate_comps=candidates,
        state_mutation=mutation,
        decision_intent=intent,
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
    candidates = understanding.candidate_comps
    other = [item for item in understanding.resolved_entities
             if item.entity_type != "composition"]
    decision = understanding.kind == QueryKind.STRATEGY_DECISION
    needs: list[str] = []
    if target or candidates:
        needs.extend(["composition", "core_units", "items"])
        if decision:
            needs.extend(_strategy_sections(understanding.decision_intent, state))
    if any(item.entity_type == "emblem" for item in other):
        needs.extend(["emblem_definition", "key_traits", "composition_emblem_advice",
                      "transition", "flex", "conditional_advice"])
    if any(item.entity_type == "augment" for item in other):
        needs.extend(["augment_definition", "conditional_advice"])
    if target is None and not candidates and decision:
        needs.extend(["general_early_game", "item_flexibility", "early_slam", "transition"])
    return RetrievalPlan(
        game_patch=patch, set_number=set_number, set_name=set_name, target_comp=target,
        candidate_comps=candidates,
        entities=other, knowledge_needs=list(dict.fromkeys(needs)),
        candidate_selection_inputs=[item for item in other
                                    if item.entity_type in {"item", "augment", "champion"}],
        retrieve_all_comps=understanding.browse_intent,
        retrieve_strategy_text=decision,
    )


def apply_state_mutation(
    state: GameState, understanding: QueryStateUnderstanding, question: str
) -> GameState:
    update: dict[str, Any] = {"user_question": question}
    comp = understanding.target_comp
    if understanding.state_mutation == StateMutationIntent.REPLACE and comp:
        update.update(comp_name=comp.display_name, active_comp_id=comp.canonical_id)
    elif understanding.state_mutation in {
        StateMutationIntent.CLEAR, StateMutationIntent.CANDIDATES,
    }:
        update.update(comp_name=None, active_comp_id=None)
    augments = [item.display_name for item in understanding.resolved_entities
                if item.entity_type == "augment"]
    if augments and re.search(r"选哪个|三选一|海克斯|强化", question):
        update["augment_choices"] = augments[:3]
    emblems = [item.display_name for item in understanding.resolved_entities
               if item.entity_type == "emblem"]
    if emblems:
        update["emblems"] = emblems
    items = [item.display_name for item in understanding.resolved_entities
             if item.entity_type == "item"]
    if items:
        update["item_components"] = items
    return state.model_copy(update=update)


def _ordered_compositions(text: str, entities: list[ResolvedEntity]) -> list[ResolvedEntity]:
    comps = [item for item in entities if item.entity_type == "composition"]
    return sorted(comps, key=lambda item: _mention_position(text, item))


def _mention_position(text: str, item: ResolvedEntity) -> int:
    positions = [text.casefold().find(value.casefold()) for value in (
        item.matched_alias, item.display_name,
    )]
    valid = [position for position in positions if position >= 0]
    return min(valid) if valid else len(text)


def _composition_semantics(
    text: str, compositions: list[ResolvedEntity]
) -> tuple[StateMutationIntent, ResolvedEntity | None, list[ResolvedEntity]]:
    clear = bool(re.search(r"还?没定(?:阵容)?|未定阵容|暂时不玩|不玩.+了", text))
    switch = re.search(r"换(?:成|到)?", text)
    if switch:
        target = next((item for item in compositions
                       if _mention_position(text, item) >= switch.end()), None)
        return ((StateMutationIntent.REPLACE, target, []) if target else
                (StateMutationIntent.CLEAR, None, []))
    available = [item for item in compositions if not re.search(
        rf"(?:不玩|不考虑|放弃)[^，,。；;]*{re.escape(item.matched_alias)}", text,
    )]
    comparison = len(available) > 1 and bool(re.search(
        r"和|与|还是|比较|判断|哪套|候选|考虑", text
    ))
    if comparison:
        return StateMutationIntent.CANDIDATES, None, available
    if clear:
        return StateMutationIntent.CLEAR, None, []
    if available:
        return StateMutationIntent.REPLACE, available[0], []
    return StateMutationIntent.RETAIN, None, []


def _strategy_sections(intent: DecisionIntent, state: GameState) -> list[str]:
    if intent == DecisionIntent.FULL_GUIDE:
        return ["starting_conditions", "early_game", "mid_game", "late_game", "core",
                "leveling"]
    if intent == DecisionIntent.COMPOSITION_COMPARISON:
        return ["starting_conditions", "mid_game", "late_game", "core", "leveling",
                "economy", "reroll_pattern"]
    bucket = _stage_bucket(state.stage, state.level)
    if intent == DecisionIntent.CURRENT_DECISION:
        if bucket == "early":
            return ["starting_conditions", "early_game", "mid_game", "core", "economy",
                    "reroll_pattern"]
        return ["mid_game", "late_game", "core", "economy", "reroll_pattern",
                "level_8_stabilization", "level_9_condition"]
    if bucket == "early":
        return ["early_game", "mid_game", "core", "transition", "conditional_advice"]
    return ["mid_game", "late_game", "core", "transition", "conditional_advice"]


def _stage_bucket(stage: str | None, level: int | None) -> str:
    major = int(stage.split("-", 1)[0]) if stage and stage.split("-", 1)[0].isdigit() else None
    if (major is not None and major <= 2) or (major is None and level is not None and level <= 5):
        return "early"
    if (major is not None and major >= 5) or (major is None and level is not None and level >= 9):
        return "late"
    return "mid"


def _user_facts(state: GameState) -> dict[str, object]:
    raw = state.model_dump(
        exclude={"user_question", "active_comp_id", "board", "bench", "shop"},
        exclude_none=True,
    )
    return {key: value for key, value in raw.items() if value not in ([], "")}


def _format_evidence(item: EvidenceRecord) -> str:
    return (
        f"- {item.display_name} / {item.section}: {item.semantic_content} "
        f"[来源：{item.source}｜版本：{item.source_patch or '未标注'}]"
    )


def _grouped_evidence(
    evidence: list[EvidenceRecord], candidates: list[ResolvedEntity]
) -> str:
    if not candidates:
        return "\n".join(_format_evidence(item) for item in evidence)
    blocks: list[str] = []
    candidate_ids = {item.canonical_id for item in candidates}
    for index, candidate in enumerate(candidates, start=1):
        records = [item for item in evidence
                   if item.canonical_entity_id == candidate.canonical_id]
        blocks.append(
            f"CANDIDATE {index}: {candidate.display_name}\n"
            + ("\n".join(_format_evidence(item) for item in records)
               if records else "- No retrieved evidence")
        )
    remaining = [item for item in evidence
                 if item.canonical_entity_id not in candidate_ids]
    if remaining:
        blocks.append("SHARED EVIDENCE\n" + "\n".join(
            _format_evidence(item) for item in remaining
        ))
    return "\n\n".join(blocks)


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
