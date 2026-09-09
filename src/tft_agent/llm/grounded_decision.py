"""Single-call DeepSeek decision client with local-only output normalization."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from tft_agent.app.pipeline import AnswerContext
from tft_agent.core.config import Settings
from tft_agent.core.schemas import AdviceEvidence, AdviceOption, FinalResponse


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
                    "你是云顶之弈中文决策助手。只使用输入中的版本化证据与玩家事实。"
                    "缺失字段位于context_gaps；若仍可回答，给出条件式建议，不要拒答。"
                    "只有关键缺失导致结论无法判断时才在summary中明确询问。"
                    "返回JSON，字段为summary和options；options最多3项，每项包含title、action、"
                    "conditions、benefit、risk、confidence。不得调用工具，不得编造数值。"
                )},
                {"role": "user", "content": context.model_dump_json()},
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
    summary = _first_text(data, "summary", "recommendation", "answer", "建议")
    raw_options = data.get("options") or data.get("choices") or data.get("建议选项") or []
    if not isinstance(raw_options, list):
        raw_options = []
    evidence = _evidence(context)
    options: list[AdviceOption] = []
    for index, raw in enumerate(raw_options[:3], start=1):
        if isinstance(raw, str):
            raw = {"action": raw}
        if not isinstance(raw, dict):
            continue
        action = _first_text(raw, "action", "recommendation", "text", "建议")
        if not action:
            continue
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        options.append(AdviceOption(
            title=_first_text(raw, "title", "name", "标题") or f"选择 {index}", action=action,
            conditions=_string_list(raw.get("conditions") or raw.get("条件")),
            expected_benefit=_first_text(raw, "benefit", "expected_benefit", "收益")
            or "在满足条件时提高当前方案的稳定性。",
            risk=_first_text(raw, "risk", "风险") or "缺失局面信息会降低判断精度。",
            confidence=confidence, evidence=evidence,
        ))
    if not summary and options:
        summary = options[0].action
    if not summary:
        raise ValueError("模型输出没有可用答案")
    sources = list(dict.fromkeys(f"{item.label} — {item.url}" for item in evidence))
    return DecisionOutcome(FinalResponse(
        recommendation=summary, alternatives=[item.action for item in options[1:]],
        confidence=options[0].confidence if options else None,
        used_tools=["structured_sqlite_retrieval", "text_retrieval"], sources=sources,
    ), options)


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


def _evidence(context: AnswerContext) -> list[AdviceEvidence]:
    output: list[AdviceEvidence] = []
    seen: set[str] = set()
    records = [*context.knowledge.comps, *context.knowledge.strategy_chunks,
               *context.knowledge.static_entities]
    for item in records:
        raw_snapshot = item.get("snapshot")
        snapshot: dict[str, Any] = raw_snapshot if isinstance(raw_snapshot, dict) else {}
        url = item.get("url") or item.get("source_url") or snapshot.get("source_url")
        if not isinstance(url, str) or not url or url in seen:
            continue
        seen.add(url)
        output.append(AdviceEvidence(
            label=str(item.get("title") or item.get("name") or item.get("source") or "来源"),
            url=url, source=str(item.get("source") or snapshot.get("source") or "local knowledge"),
            game_patch=str(item.get("game_patch") or snapshot.get("game_patch") or "") or None,
        ))
    return output
