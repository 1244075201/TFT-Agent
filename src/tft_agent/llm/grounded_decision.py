"""Single-call DeepSeek decision client with local-only output normalization."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from tft_agent.app.pipeline import AnswerContext
from tft_agent.core.config import Settings
from tft_agent.core.schemas import AdviceOption, FinalResponse


@dataclass(frozen=True)
class DecisionOutcome:
    response: FinalResponse
    options: list[AdviceOption]


@dataclass(frozen=True)
class GroundingValidation:
    outcome: DecisionOutcome
    violations: list[str]
    used_fallback: bool


class GroundedDecisionClient:
    """Make exactly one provider call for a strategy answer."""

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(
            api_key=settings.require_deepseek_api_key().get_secret_value(),
            base_url=str(settings.deepseek_base_url).rstrip("/"), max_retries=0,
            timeout=settings.decision_timeout_seconds,
        )
        self._owns_client = client is None
        self._model = settings.model

    async def decide_context(self, context: AnswerContext) -> DecisionOutcome:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": (
                    "你是云顶之弈中文决策助手。直接回答USER QUESTION，只使用输入中的"
                    "CURRENT GAMESTATE/USER FACTS与版本化检索证据。先理解用户原始问题"
                    "真正要解决的决策，再综合最相关的证据给出具体直接的回答。"
                    "CURRENT GAMESTATE/USER FACTS优先于RECENT CONVERSATION；"
                    "历史仅用于理解省略表达，不能覆盖当前事实。RETRIEVED STRATEGY "
                    "CONDITIONS是攻略适用条件，不代表玩家已经满足，除非同一事实明确出现于"
                    "USER FACTS，否则不得写成‘你现在已经’。不要把互斥策略无条件并列；"
                    "若证据只支持一条路线，不得凭常识添加相反路线；若有条件分支，必须明确"
                    "写成‘如果A，则X；如果B，则Y’。缺失字段位于CONTEXT/KNOWLEDGE GAPS；"
                    "若仍可回答，给出条件式建议，不要拒答。"
                    "允许基于已给事实做有条件的适配推理和权衡解释，但不得把关联顺序"
                    "解释成排名，也不得把没有统计证据的选择说成绝对最优。"
                    "CANDIDATE COMPOSITION与STRUCTURED RELATION只是由阵容名单、棋子羁绊、"
                    "装备持有者或强化association反查出的候选；其出现顺序和relation数量只"
                    "表示检索相关性，不代表阵容强度、胜率或最佳选择。"
                    "若存在PREVIOUS TOPIC和CURRENT CANDIDATES，当前问句中的‘第二个’、"
                    "‘其他路线’、‘这几个’已由程序确定性解析为这些阵容；必须按该阵容"
                    "范围回答，不得重新解释成装备槽位、棋子顺序或其他序号。"
                    "事实边界是封闭的：只有USER FACTS和RETRIEVED证据中明确出现的实体、"
                    "路线、装备、强化及其relation才能写成已知事实。不得发明候选路线或"
                    "候选数量，不得把历史对话、知识缺口或一般常识改写成当前用户状态。"
                    "若KNOWLEDGE GAPS指出已识别实体之间缺少relation，必须直接陈述缺口；"
                    "任何超出证据的建议都必须以‘如果’开头并明确标成假设。"
                    "GENERAL_GUIDE是版本攻略咨询：不要把没有个人对局信息、没有结构化"
                    "GameState描述成问题；只有MATCH_GUIDANCE才讨论对局状态是否缺失。"
                    "返回JSON：recommendation、reasoning、next_steps，以及可选的alternatives和"
                    "missing_information。只有确有不同选择时才给alternatives。不得调用工具，"
                    "不得编造上下阵单位、数值或知识库未提供的事实。"
                )},
                {"role": "user", "content": context.readable_summary()},
            ],
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("模型未返回可用内容")
        return normalize_decision_output(content, context)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def normalize_decision_output(content: str, context: AnswerContext) -> DecisionOutcome:
    """Normalize JSON, fenced JSON, partial JSON, or useful plain text locally."""
    data = _extract_object(content)
    if data is None:
        return DecisionOutcome(FinalResponse(recommendation=content.strip()), [])
    summary = _first_text(data, "recommendation", "summary", "answer", "建议")
    if not summary:
        raise ValueError("模型输出没有可用答案")
    sources = [f"{item.display_name} · {item.section} — {item.url}"
               for item in context.evidence]
    return DecisionOutcome(FinalResponse(
        recommendation=summary,
        reasoning=_string_list(data.get("reasoning")),
        next_steps=_string_list(data.get("next_steps")),
        alternatives=_string_list(data.get("alternatives")),
        used_tools=["structured_sqlite_retrieval", "text_retrieval"], sources=sources,
    ), [])


def enforce_closed_world_grounding(
    outcome: DecisionOutcome,
    context: AnswerContext,
    catalog: list[dict[str, object]],
) -> GroundingValidation:
    """Reject canonical entities and statistics unsupported by current-task evidence."""
    response = outcome.response
    answer_text = "\n".join([
        response.recommendation, *response.reasoning, *response.next_steps,
        *response.alternatives,
    ])
    evidence_text = json.dumps(
        [item.model_dump(mode="json") for item in context.evidence], ensure_ascii=False
    )
    allowed_text = "\n".join((
        context.question,
        evidence_text,
        json.dumps(context.game_state.model_dump(mode="json"), ensure_ascii=False),
        " ".join(item.display_name for item in context.current_candidates),
    ))
    violations: list[str] = []
    seen_entities: set[str] = set()
    for row in catalog:
        names = [str(row.get("display_name") or "")]
        aliases = row.get("aliases")
        if isinstance(aliases, list):
            names.extend(str(alias) for alias in aliases)
        for name in names:
            normalized = name.strip()
            if (
                len(normalized) < 2
                or normalized in seen_entities
                or re.fullmatch(r"(?:\d+星)?\d+费(?:弈子|卡|单位)?", normalized)
            ):
                continue
            if normalized in answer_text and normalized not in allowed_text:
                violations.append(f"unsupported canonical entity: {normalized}")
                seen_entities.add(normalized)
                break
    statistic_patterns = (
        r"(?:胜率|前四率|登顶率|第一率|平均排名|top\s*4|win\s*rate|first\s*rate|"
        r"average\s*placement)\s*(?:为|是|[:：])?\s*\d+(?:\.\d+)?%?",
        r"\d+(?:\.\d+)?%?\s*(?:的)?\s*(?:胜率|前四率|登顶率|第一率|平均排名|"
        r"top\s*4|win\s*rate|first\s*rate|average\s*placement)",
    )
    for pattern in statistic_patterns:
        for match in re.finditer(pattern, answer_text, re.I):
            claim = match.group(0)
            if not _evidence_supports_statistic(claim, context):
                violations.append(f"unsupported statistic: {claim[:80]}")
    qualitative_statistic_patterns = (
        r"(?:胜率|前四率|登顶率|第一率|平均排名)\s*(?:更|较|很|非常)?(?:高|低|好|差|领先)",
        r"(?:更|较|很|非常)?(?:高|低|好|差|领先)\s*的?(?:胜率|前四率|登顶率|第一率|平均排名)",
    )
    if not _has_statistic_evidence(context):
        for pattern in qualitative_statistic_patterns:
            for match in re.finditer(pattern, answer_text):
                violations.append(f"unsupported statistic: {match.group(0)[:80]}")
    if not violations:
        if context.retrieval_interpretation and not _has_statistic_evidence(context):
            caveat = (
                "当前证据不含胜率、前四率、登顶率或平均排名；以上只能视为攻略证据上的"
                "定性筛选，不是统计排名。"
            )
            response = response.model_copy(update={
                "recommendation": response.recommendation.rstrip() + "\n\n" + caveat
            })
            outcome = DecisionOutcome(response=response, options=outcome.options)
        return GroundingValidation(outcome=outcome, violations=[], used_fallback=False)
    candidates = list(dict.fromkeys(
        item.display_name for item in context.evidence if item.entity_type == "composition"
    ))
    if candidates:
        recommendation = (
            f"当前 {context.retrieval_plan.game_patch} 检索证据只支持定性考虑："
            + "、".join(candidates[:context.retrieval_plan.composition_limit])
            + "。可以根据已检索到的攻略条件和运营阶段做选择，但现有证据不足以支持"
              "回答中出现的额外游戏实体或统计排名。"
        )
    else:
        recommendation = (
            f"当前 {context.retrieval_plan.game_patch} 知识库没有足够检索证据支持这个判断；"
            "我不能提供未经当前证据支持的游戏实体或统计比较。"
        )
    fallback = DecisionOutcome(
        response=FinalResponse(
            recommendation=recommendation,
            used_tools=["closed_world_grounding_fallback"],
            sources=[f"{item.display_name} · {item.section} — {item.url}"
                     for item in context.evidence],
        ),
        options=[],
    )
    return GroundingValidation(
        outcome=fallback, violations=list(dict.fromkeys(violations)), used_fallback=True
    )


def _evidence_supports_statistic(claim: str, context: AnswerContext) -> bool:
    keys = {
        "胜率": "win_rate", "win rate": "win_rate",
        "前四率": "top4_rate", "top4": "top4_rate", "top 4": "top4_rate",
        "登顶率": "first_rate", "第一率": "first_rate", "first rate": "first_rate",
        "平均排名": "average_place", "average placement": "average_place",
    }
    lowered = claim.casefold()
    expected = next((key for label, key in keys.items() if label in lowered), None)
    numbers = re.findall(r"\d+(?:\.\d+)?%?", claim)
    if expected is None or not numbers:
        return False
    for evidence in context.evidence:
        if not isinstance(evidence.semantic_content, dict):
            continue
        value = evidence.semantic_content.get(expected)
        if value is not None and any(number in str(value) for number in numbers):
            return True
    return False


def _has_statistic_evidence(context: AnswerContext) -> bool:
    keys = {"win_rate", "top4_rate", "first_rate", "average_place"}
    return any(
        isinstance(item.semantic_content, dict)
        and any(item.semantic_content.get(key) is not None for key in keys)
        for item in context.evidence
    )


def _extract_object(content: str) -> dict[str, Any] | None:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    candidates = [stripped]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _first_text(value: dict[str, Any], *keys: str) -> str:
    return next((str(value[key]).strip() for key in keys
                 if isinstance(value.get(key), str) and str(value[key]).strip()), "")


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item).strip()] if isinstance(value, list) else []
