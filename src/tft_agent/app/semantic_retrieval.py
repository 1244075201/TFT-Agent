"""Bounded LLM planning contract for semantic composition discovery."""
from __future__ import annotations

import json
import re
from enum import StrEnum
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI
from pydantic import Field, ValidationError

from tft_agent.core.config import Settings
from tft_agent.core.schemas import GameState, StrictModel


class RetrievalCapability(StrEnum):
    LIST_COMPOSITIONS = "LIST_COMPOSITIONS"
    FILTER_BY_MAIN_UNIT_COST = "FILTER_BY_MAIN_UNIT_COST"
    SEARCH_STRATEGY_TEXT = "SEARCH_STRATEGY_TEXT"
    GET_COMPOSITION_STRATEGY = "GET_COMPOSITION_STRATEGY"
    GET_COMPOSITION_ROSTER = "GET_COMPOSITION_ROSTER"
    GET_COMPOSITION_ITEMS = "GET_COMPOSITION_ITEMS"
    FIND_BY_ITEM_OR_ARTIFACT = "FIND_BY_ITEM_OR_ARTIFACT"


class CostQuantifier(StrEnum):
    ANY = "ANY"
    ALL = "ALL"
    REPRESENTATIVE = "REPRESENTATIVE"


class DiscoveryQueryType(StrEnum):
    COMPOSITION_DISCOVERY = "composition_discovery"
    STATE_AWARE_STRATEGY = "state_aware_strategy"


class CompositionFilters(StrictModel):
    main_unit_cost_min: int | None = Field(default=None, ge=1, le=5)
    main_unit_cost_max: int | None = Field(default=None, ge=1, le=5)
    cost_quantifier: CostQuantifier = CostQuantifier.ANY
    require_strategy_match: bool = False


class SemanticRetrievalPlan(StrictModel):
    """Provider-produced intent only; patch and repository details are deliberately absent."""

    user_goal: str = Field(min_length=1, max_length=240)
    query_type: DiscoveryQueryType = DiscoveryQueryType.COMPOSITION_DISCOVERY
    observed_state: list[str] = Field(default_factory=list, max_length=8)
    decision_questions: list[str] = Field(default_factory=list, max_length=8)
    capabilities: list[RetrievalCapability] = Field(min_length=1, max_length=9)
    composition_filters: CompositionFilters = Field(default_factory=CompositionFilters)
    strategy_searches: list[str] = Field(default_factory=list, max_length=12)
    strategy_exclusions: list[str] = Field(default_factory=list, max_length=12)
    item_searches: list[str] = Field(default_factory=list, max_length=12)
    requested_fields: list[str] = Field(default_factory=list, max_length=10)
    limit: int = Field(default=8, ge=1, le=12)


class PlannerOutcome(StrictModel):
    plan: SemanticRetrievalPlan
    model: str
    latency_ms: float = Field(ge=0)
    used_fallback: bool = False
    validation_error: str | None = None


_ALLOWED_FIELDS = {
    "tier", "main_units", "starting_conditions", "core", "early_game",
    "mid_game", "late_game", "tips", "roster", "items",
}


def is_semantic_discovery_query(text: str, resolved_entity_types: set[str]) -> bool:
    """Route structurally candidate-seeking requests without enumerating preferences."""
    if resolved_entity_types & {"composition", "champion", "trait", "emblem", "augment"}:
        return False
    normalized = re.sub(r"\s+", "", text.casefold())
    asks_for_candidates = bool(re.search(r"玩什么|有(?:哪|啥|什么)|哪些|推荐", normalized))
    describes_preference = bool(re.search(
        r"^(?:我)?(?:想|要|希望|有没有|有无).+的(?:阵容)?$|^.+的(?:阵容)?(?:有.*)?$",
        normalized,
    ))
    return asks_for_candidates or describes_preference


def is_state_aware_strategy_query(text: str, resolved_entity_types: set[str]) -> bool:
    """A decision about a described state, not a request to discover compositions."""
    if resolved_entity_types:
        return False
    normalized = re.sub(r"\s+", "", text.casefold())
    if is_semantic_discovery_query(text, resolved_entity_types):
        return False
    open_decision = re.search(r"怎么(?!玩|构筑|出装|运营)|如何应对", normalized)
    if open_decision:
        return open_decision.start() >= 3
    binary_decision = re.search(r"该不该|要不要", normalized)
    return bool(binary_decision and re.search(r"如果|但|又|没|不够|不足", normalized))


def validate_plan_payload(payload: dict[str, Any]) -> SemanticRetrievalPlan:
    """Strictly validate the capability vocabulary and bounded filter values."""
    plan = SemanticRetrievalPlan.model_validate(payload)
    unknown_fields = set(plan.requested_fields) - _ALLOWED_FIELDS
    if unknown_fields:
        raise ValueError(f"unsupported requested fields: {sorted(unknown_fields)}")
    low = plan.composition_filters.main_unit_cost_min
    high = plan.composition_filters.main_unit_cost_max
    if low is not None and high is not None and low > high:
        raise ValueError("main unit cost minimum exceeds maximum")
    if plan.query_type == DiscoveryQueryType.STATE_AWARE_STRATEGY and not (
        plan.observed_state and plan.decision_questions and plan.strategy_searches
    ):
        raise ValueError("state-aware strategy requires state, decisions and search concepts")
    if plan.query_type == DiscoveryQueryType.STATE_AWARE_STRATEGY and plan.strategy_exclusions:
        # State negations are decision constraints, not composition exclusion filters.
        plan = plan.model_copy(update={
            "strategy_searches": list(dict.fromkeys([
                *plan.strategy_searches, *plan.strategy_exclusions,
            ])),
            "strategy_exclusions": [],
        })
    declared = set(plan.capabilities)
    strategy_fields = {
        "starting_conditions", "core", "early_game", "mid_game", "late_game", "tips",
    }
    implied = {
        RetrievalCapability.FILTER_BY_MAIN_UNIT_COST: low is not None or high is not None,
        RetrievalCapability.SEARCH_STRATEGY_TEXT: bool(
            plan.strategy_searches or plan.strategy_exclusions
        ),
        RetrievalCapability.GET_COMPOSITION_STRATEGY: bool(
            set(plan.requested_fields) & strategy_fields
        ),
        RetrievalCapability.GET_COMPOSITION_ROSTER: "roster" in plan.requested_fields,
        RetrievalCapability.GET_COMPOSITION_ITEMS: "items" in plan.requested_fields,
        RetrievalCapability.FIND_BY_ITEM_OR_ARTIFACT: bool(plan.item_searches),
    }
    # Parameters are the executable intent. Canonicalize an omitted declaration so
    # the validated plan and executor trace cannot silently disagree.
    declared.update(capability for capability, present in implied.items() if present)
    plan = plan.model_copy(update={
        "capabilities": [
            capability for capability in RetrievalCapability if capability in declared
        ]
    })
    requirements = {
        RetrievalCapability.FILTER_BY_MAIN_UNIT_COST: low is not None or high is not None,
        RetrievalCapability.SEARCH_STRATEGY_TEXT: bool(
            plan.strategy_searches or plan.strategy_exclusions
        ),
        RetrievalCapability.GET_COMPOSITION_STRATEGY: bool(
            set(plan.requested_fields) & strategy_fields
        ),
        RetrievalCapability.GET_COMPOSITION_ROSTER: "roster" in plan.requested_fields,
        RetrievalCapability.GET_COMPOSITION_ITEMS: "items" in plan.requested_fields,
        RetrievalCapability.FIND_BY_ITEM_OR_ARTIFACT: bool(plan.item_searches),
    }
    missing = [capability.value for capability, valid in requirements.items()
               if capability in declared and not valid]
    if missing:
        raise ValueError(f"capabilities missing required parameters: {missing}")
    return plan


def safe_fallback_plan() -> SemanticRetrievalPlan:
    return SemanticRetrievalPlan(
        user_goal="find compositions matching the user's stated strategic conditions",
        capabilities=[
            RetrievalCapability.LIST_COMPOSITIONS,
            RetrievalCapability.GET_COMPOSITION_STRATEGY,
        ],
        requested_fields=["tier", "main_units", "starting_conditions", "core", "early_game"],
        limit=8,
    )


def safe_state_aware_fallback_plan(query: str) -> SemanticRetrievalPlan:
    return SemanticRetrievalPlan(
        user_goal="find active-patch strategy evidence for the user's stated decision",
        query_type=DiscoveryQueryType.STATE_AWARE_STRATEGY,
        observed_state=[query], decision_questions=[query],
        capabilities=[RetrievalCapability.SEARCH_STRATEGY_TEXT,
                      RetrievalCapability.GET_COMPOSITION_STRATEGY],
        strategy_searches=[query],
        requested_fields=["starting_conditions", "core", "mid_game", "late_game", "tips"],
    )


class RetrievalPlannerClient:
    """One provider call producing a validated, non-executable retrieval intent."""

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(
            api_key=settings.require_deepseek_api_key().get_secret_value(),
            base_url=str(settings.deepseek_base_url).rstrip("/"), max_retries=0,
            timeout=settings.decision_timeout_seconds,
        )
        self._owns_client = client is None
        self._model = settings.model

    async def plan(self, query: str, game_state: GameState | None = None,
                   *, state_aware: bool = False) -> PlannerOutcome:
        started = perf_counter()
        state = (game_state.model_dump(exclude_none=True, exclude_defaults=True)
                 if game_state is not None else {})
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": (
                    "你是检索规划器，不回答用户问题。只输出JSON检索意图。只能使用以下capabilities："
                    + ", ".join(item.value for item in RetrievalCapability)
                    + "。不得输出patch、SQL、工具名、repository方法、文件或网络操作。"
                    "如果用户描述现有局面并问下一步，query_type=state_aware_strategy；"
                    "observed_state只记用户明确说出的状态，decision_questions列出要判断的分支。"
                    "strategy_searches使用短的可检索运营概念，覆盖经济、升级、搜牌、"
                    "保血、阶段和条件式攻略；requested_fields优先mid_game、late_game、"
                    "starting_conditions、core、tips。不要把此类问题改写成阵容推荐。"
                    "低费赌狗需要同时规划低费主核筛选与追三星/慢D/D牌/低等级搜牌等攻略语义，"
                    "并令require_strategy_match=true；低费阵容只筛低费主核，不得自动要求赌牌。"
                    "否定某种玩法时不要把被否定概念设为require_strategy_match；应把该概念的"
                    "语义扩展放入strategy_exclusions，并声明SEARCH_STRATEGY_TEXT。"
                    "费用主C默认cost_quantifier=ANY；只有用户明确要求所有核心都满足时才用ALL，"
                    "明确要求代表英雄时才用REPRESENTATIVE。"
                    "对‘不依赖X费单位’这类费用依赖约束，应转换成主核费用范围，而不是只列阵容。"
                    "物理装备开局应检索装备方案和起手条件；前期强势应检索starting_conditions与"
                    "early_game，并选择SEARCH_STRATEGY_TEXT，不要选择无具体羁绊条件的"
                    "FIND_BY_TRAIT，也不要声称有阶段胜率。requested_fields只能取tier, main_units, "
                    "starting_conditions, core, early_game, mid_game, late_game, tips, roster, "
                    "items。声明capability时必须提供对应参数：FILTER需要费用范围，SEARCH需要"
                    "strategy_searches，GET_STRATEGY需要攻略requested_fields，GET_ROSTER需要"
                    "roster，GET_ITEMS需要items，FIND_BY_ITEM需要item_searches。"
                    "反向也成立：任何非空过滤器、search或roster/items字段都必须声明对应"
                    "capability，不得输出不会执行的孤立参数。"
                    "limit为1到12。输出字段必须且只能符合schema。"
                    "精确JSON形状如下：{\"user_goal\":\"一句检索目标\","
                    "\"query_type\":\"composition_discovery\",\"capabilities\":["
                    "\"LIST_COMPOSITIONS\"],\"observed_state\":[],"
                    "\"decision_questions\":[],\"composition_filters\":{"
                    "\"main_unit_cost_min\":null,\"main_unit_cost_max\":null,"
                    "\"cost_quantifier\":\"ANY\","
                    "\"require_strategy_match\":false},\"strategy_searches\":[],"
                    "\"strategy_exclusions\":[],"
                    "\"item_searches\":[],\"requested_fields\":[\"tier\"],\"limit\":8}。"
                    "不要增加解释或任何其他键。物理装备可以扩展为现有装备名/散件名搜索词，"
                    "例如暴风之剑、反曲之弓、拳套及相关物理成装；这些词只是检索概念。"
                )},
                {"role": "user", "content": json.dumps(
                    {"query": query, "game_state": state}, ensure_ascii=False
                )},
            ],
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content or ""
        try:
            decoded = json.loads(content)
            if not isinstance(decoded, dict):
                raise ValueError("planner output is not an object")
            plan = validate_plan_payload(decoded)
            if state_aware and plan.query_type != DiscoveryQueryType.STATE_AWARE_STRATEGY:
                raise ValueError("planner did not represent the state-aware decision")
            return PlannerOutcome(
                plan=plan, model=self._model, latency_ms=(perf_counter() - started) * 1000,
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return PlannerOutcome(
                plan=(safe_state_aware_fallback_plan(query) if state_aware else
                      safe_fallback_plan()), model=self._model,
                latency_ms=(perf_counter() - started) * 1000,
                used_fallback=True, validation_error=str(exc)[:500],
            )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()
