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
    FIND_BY_TRAIT = "FIND_BY_TRAIT"
    FIND_BY_ITEM_OR_ARTIFACT = "FIND_BY_ITEM_OR_ARTIFACT"
    FIND_BY_AUGMENT = "FIND_BY_AUGMENT"


class DiscoveryQueryType(StrEnum):
    COMPOSITION_DISCOVERY = "composition_discovery"


class CompositionFilters(StrictModel):
    main_unit_cost_min: int | None = Field(default=None, ge=1, le=5)
    main_unit_cost_max: int | None = Field(default=None, ge=1, le=5)
    require_strategy_match: bool = False


class SemanticRetrievalPlan(StrictModel):
    """Provider-produced intent only; patch and repository details are deliberately absent."""

    user_goal: str = Field(min_length=1, max_length=240)
    query_type: DiscoveryQueryType = DiscoveryQueryType.COMPOSITION_DISCOVERY
    capabilities: list[RetrievalCapability] = Field(min_length=1, max_length=9)
    composition_filters: CompositionFilters = Field(default_factory=CompositionFilters)
    strategy_searches: list[str] = Field(default_factory=list, max_length=12)
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
    """Conservative gate for abstract composition filters, not named strategy subjects."""
    if resolved_entity_types & {"composition", "champion", "trait", "emblem", "augment"}:
        return False
    normalized = re.sub(r"\s+", "", text.casefold())
    asks_for_compositions = bool(re.search(r"玩什么|阵容|有哪些|有什么|适合玩", normalized))
    abstract_condition = bool(re.search(
        r"低费|高费|赌狗|赌牌|追三|三星|物理(?:装|装备|散件)|法系(?:装|装备|散件)|"
        r"前期|后期|容易成型|好成型|容易追|开局|起手|过渡强", normalized,
    ))
    return asks_for_compositions and abstract_condition


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
    return plan


def safe_fallback_plan() -> SemanticRetrievalPlan:
    return SemanticRetrievalPlan(
        user_goal="find compositions matching the user's stated strategic conditions",
        capabilities=[
            RetrievalCapability.LIST_COMPOSITIONS,
            RetrievalCapability.SEARCH_STRATEGY_TEXT,
            RetrievalCapability.GET_COMPOSITION_STRATEGY,
        ],
        requested_fields=["tier", "main_units", "starting_conditions", "core", "early_game"],
        limit=8,
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

    async def plan(self, query: str, game_state: GameState | None = None) -> PlannerOutcome:
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
                    "低费赌狗需要同时规划低费主核筛选与追三星/慢D/D牌/低等级搜牌等攻略语义，"
                    "并令require_strategy_match=true；低费阵容只筛低费主核，不得自动要求赌牌。"
                    "物理装备开局应检索装备方案和起手条件；前期强势应检索starting_conditions与"
                    "early_game，并选择SEARCH_STRATEGY_TEXT，不要选择无具体羁绊条件的"
                    "FIND_BY_TRAIT，也不要声称有阶段胜率。requested_fields只能取tier, main_units, "
                    "starting_conditions, core, early_game, mid_game, late_game, tips, roster, "
                    "items。"
                    "limit为1到12。输出字段必须且只能符合schema。"
                    "精确JSON形状如下：{\"user_goal\":\"一句检索目标\","
                    "\"query_type\":\"composition_discovery\",\"capabilities\":["
                    "\"LIST_COMPOSITIONS\"],\"composition_filters\":{"
                    "\"main_unit_cost_min\":null,\"main_unit_cost_max\":null,"
                    "\"require_strategy_match\":false},\"strategy_searches\":[],"
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
            return PlannerOutcome(
                plan=plan, model=self._model, latency_ms=(perf_counter() - started) * 1000,
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return PlannerOutcome(
                plan=safe_fallback_plan(), model=self._model,
                latency_ms=(perf_counter() - started) * 1000,
                used_fallback=True, validation_error=str(exc)[:500],
            )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()
