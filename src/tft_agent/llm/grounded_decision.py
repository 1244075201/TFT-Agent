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
                    "USER FACTS与版本化检索证据。USER FACTS优先于RECENT CONVERSATION；"
                    "历史仅用于理解省略表达，不能覆盖当前事实。RETRIEVED STRATEGY "
                    "CONDITIONS是攻略适用条件，不代表玩家已经满足，除非同一事实明确出现于"
                    "USER FACTS，否则不得写成‘你现在已经’。不要把互斥策略无条件并列；"
                    "若证据只支持一条路线，不得凭常识添加相反路线；若有条件分支，必须明确"
                    "写成‘如果A，则X；如果B，则Y’。缺失字段位于CONTEXT/KNOWLEDGE GAPS；"
                    "若仍可回答，给出条件式建议，不要拒答。"
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
