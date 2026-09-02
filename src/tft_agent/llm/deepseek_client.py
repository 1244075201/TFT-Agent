"""DeepSeek Chat Completions adapter implementing the project LLM boundary."""

import asyncio
import json
import re
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)

from tft_agent.agent.state import AgentState
from tft_agent.agent.trace import (
    LLMCallTrace,
    NumericClaimTrace,
    NumericClaimValidationTrace,
    QualitativeRecommendationTrace,
    TokenUsage,
)
from tft_agent.core.config import Settings
from tft_agent.core.exceptions import LLMProviderError
from tft_agent.core.schemas import (
    AgentAction,
    AgentDecision,
    FinalResponse,
    NumericClaim,
    NumericClaimStatus,
    QualitativeClaimStatus,
    StrategyRecommendation,
)
from tft_agent.llm.deepseek_messages import (
    build_provider_messages,
    build_provider_tools,
    normalize_provider_response,
)
from tft_agent.llm.evidence import build_evidence_alias_registry, build_evidence_registry
from tft_agent.llm.numeric_claims import NumericClaimValidator, NumericValidationReport
from tft_agent.llm.qualitative_claims import QualitativeRecommendationValidator
from tft_agent.mcp_client.client import MCPToolDefinition


class DeepSeekLLMClient:
    """Call DeepSeek with native tools and return provider-neutral decisions."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        thinking_mode: str = "disabled",
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMProviderError(
                code="MISSING_CONFIGURATION",
                message="DEEPSEEK_API_KEY is not configured",
                retryable=False,
            )
        self._model = model
        self._thinking_mode = thinking_mode
        self._max_retries = max_retries
        # SDK retries are disabled so this adapter owns one bounded retry policy.
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "DeepSeekLLMClient":
        """Construct from centralized secret-safe settings."""

        return cls(
            api_key=settings.require_deepseek_api_key().get_secret_value(),
            model=settings.model,
            base_url=str(settings.deepseek_base_url).rstrip("/"),
            thinking_mode=settings.thinking_mode,
            max_retries=settings.llm_max_retries,
        )

    async def generate(
        self,
        state: AgentState,
        tools: Sequence[MCPToolDefinition],
    ) -> AgentDecision:
        """Make one bounded provider call and normalize its decision."""

        tool_list = list(tools)
        request: dict[str, Any] = {
            "model": self._model,
            "messages": build_provider_messages(state, tool_list),
            "tools": build_provider_tools(tool_list),
            "tool_choice": "auto",
            "extra_body": {"thinking": {"type": self._thinking_mode}},
        }
        started = perf_counter()
        try:
            response = await self._complete_with_retry(request)
        except LLMProviderError as error:
            state.trace.llm_calls.append(
                LLMCallTrace(
                    call_index=len(state.trace.llm_calls) + 1,
                    model=self._model,
                    latency_ms=(perf_counter() - started) * 1000,
                    outcome=error.code,
                )
            )
            raise
        latency_ms = (perf_counter() - started) * 1000
        repaired_invalid_envelope = False
        try:
            decision = normalize_provider_response(response, tool_list, state)
        except LLMProviderError as error:
            if error.code != "MALFORMED_PROVIDER_RESPONSE":
                raise
            state.trace.finalization.invalid_envelope_count += 1
            state.trace.llm_calls.append(
                LLMCallTrace(
                    call_index=len(state.trace.llm_calls) + 1,
                    model=self._model,
                    latency_ms=latency_ms,
                    outcome="provider_envelope_invalid",
                    token_usage=_extract_usage(response),
                )
            )
            decision = await self._repair_finalization(
                state, request, tool_list, "provider_envelope_invalid"
            )
            repaired_invalid_envelope = True
        if not repaired_invalid_envelope:
            state.trace.llm_calls.append(
                LLMCallTrace(
                    call_index=len(state.trace.llm_calls) + 1,
                    model=self._model,
                    latency_ms=latency_ms,
                    outcome=decision.action.value,
                    token_usage=_extract_usage(response),
                )
            )
        if decision.action == AgentAction.REQUEST_CONTEXT and _request_context_is_premature(
            decision, state
        ):
            state.trace.finalization.premature_request_context_count += 1
            decision = await self._repair_finalization(
                state, request, tool_list, "premature_request_context"
            )
        if decision.action != AgentAction.FINAL:
            return decision
        return await self._enforce_numeric_claims(state, tool_list, request, decision)

    async def _repair_finalization(
        self,
        state: AgentState,
        initial_request: dict[str, Any],
        tools: list[MCPToolDefinition],
        reason: str,
    ) -> AgentDecision:
        """Spend one bounded retry on an invalid orchestration action."""

        trace = state.trace.finalization
        if trace.repair_count >= 2:
            raise LLMProviderError(
                "FINALIZATION_REPAIR_EXHAUSTED",
                "Finalization repair budget exhausted",
                False,
            )
        trace.repair_count += 1
        trace.repair_reasons.append(reason)
        aliases = build_evidence_alias_registry(state)
        feedback = (
            f"Your previous action failed finalization validation: {reason}. "
            "All successful observations shown in ACCUMULATED CURRENT-TASK EVIDENCE remain "
            "available. Do not request a field already present. Continue with the next materially "
            "useful tool call if evidence is still required; otherwise return exactly one valid "
            "FINAL envelope that directly answers the user's core question. Do not combine FINAL "
            "with REQUEST_CONTEXT. Use only these evidence aliases: "
            f"{json.dumps(list(aliases), ensure_ascii=False)}."
        )
        repair_request = {
            **initial_request,
            "messages": [*initial_request["messages"], {"role": "user", "content": feedback}],
            "tools": build_provider_tools(tools),
            "tool_choice": "auto",
        }
        started = perf_counter()
        response = await self._complete_with_retry(repair_request)
        decision = normalize_provider_response(response, tools, state)
        state.trace.llm_calls.append(
            LLMCallTrace(
                call_index=len(state.trace.llm_calls) + 1,
                model=self._model,
                latency_ms=(perf_counter() - started) * 1000,
                outcome="finalization_repair",
                token_usage=_extract_usage(response),
            )
        )
        if decision.action != AgentAction.REQUEST_CONTEXT:
            trace.repair_success += 1
        return decision

    async def _enforce_numeric_claims(
        self,
        state: AgentState,
        tools: list[MCPToolDefinition],
        initial_request: dict[str, Any],
        decision: AgentDecision,
    ) -> AgentDecision:
        """Validate once, repair once without tools, then use a deterministic fallback."""

        registry = build_evidence_registry(state)
        report, validation_ms = _validate_decision(decision, registry)
        assert decision.final_response is not None
        qualitative = QualitativeRecommendationValidator().validate(
            decision.final_response.strategy_recommendation, state
        )
        recommendation_required = _strategy_recommendation_required(state)
        qualitative_complete = (
            not recommendation_required
            or decision.final_response.strategy_recommendation is not None
        )
        state.trace.qualitative_recommendation_validation = QualitativeRecommendationTrace(
            status=(
                qualitative.status
                if qualitative_complete
                else QualitativeClaimStatus.INVALID_REFERENCE
            ),
            evidence_refs=(
                decision.final_response.strategy_recommendation.evidence_refs
                if decision.final_response.strategy_recommendation
                else []
            ),
        )
        if qualitative.status == QualitativeClaimStatus.INVALID_REFERENCE:
            state.trace.finalization.invalid_evidence_ref_count += 1
        trace = _validation_trace(
            report,
            validation_ms,
            decision.final_response.recommendation,
        )
        state.trace.numeric_claim_validation = trace
        if (
            report.valid
            and _declaration_complete(decision)
            and qualitative.valid
            and qualitative_complete
        ):
            return decision

        trace.correction_attempted = True
        finalization_trace = state.trace.finalization
        if finalization_trace.repair_count >= 2:
            trace.fallback_used = True
            finalization_trace.deterministic_fallback_count += 1
            return _safe_fallback(
                report, decision.final_response.strategy_recommendation, qualitative.valid
            )
        finalization_trace.repair_count += 1
        finalization_trace.repair_reasons.append("numeric_or_evidence_validation")
        correction_request = {
            "model": self._model,
            "messages": [
                *initial_request["messages"],
                {"role": "assistant", "content": _decision_envelope(decision)},
                {"role": "user", "content": _correction_feedback(report, decision, state)},
            ],
            "extra_body": {"thinking": {"type": self._thinking_mode}},
        }
        started = perf_counter()
        try:
            response = await self._complete_with_retry(correction_request)
            correction = normalize_provider_response(response, [], state)
        except LLMProviderError as error:
            if error.code == "MALFORMED_PROVIDER_RESPONSE":
                finalization_trace.invalid_envelope_count += 1
            trace.correction_latency_ms = (perf_counter() - started) * 1000
            trace.fallback_used = True
            finalization_trace.deterministic_fallback_count += 1
            fallback = _safe_fallback(
                report, decision.final_response.strategy_recommendation, qualitative.valid
            )
            qualitative_trace = state.trace.qualitative_recommendation_validation
            assert qualitative_trace is not None
            qualitative_trace.preserved_after_repair = (
                fallback.final_response is not None
                and fallback.final_response.strategy_recommendation is not None
            )
            return fallback
        correction_ms = (perf_counter() - started) * 1000
        trace.correction_latency_ms = correction_ms
        state.trace.llm_calls.append(
            LLMCallTrace(
                call_index=len(state.trace.llm_calls) + 1,
                model=self._model,
                latency_ms=correction_ms,
                outcome="numeric_correction",
                token_usage=_extract_usage(response),
            )
        )
        if correction.action == AgentAction.FINAL:
            corrected_report, corrected_validation_ms = _validate_decision(correction, registry)
            assert correction.final_response is not None
            corrected_qualitative = QualitativeRecommendationValidator().validate(
                correction.final_response.strategy_recommendation, state
            )
            trace.validation_latency_ms += corrected_validation_ms
            if (
                corrected_report.valid
                and _declaration_complete(correction)
                and corrected_qualitative.valid
                and (
                    not recommendation_required
                    or correction.final_response.strategy_recommendation is not None
                )
            ):
                trace.correction_succeeded = True
                finalization_trace.repair_success += 1
                qualitative_trace = state.trace.qualitative_recommendation_validation
                assert qualitative_trace is not None
                qualitative_trace.status = QualitativeClaimStatus.SUPPORTED
                qualitative_trace.preserved_after_repair = (
                    correction.final_response.strategy_recommendation is not None
                )
                return correction
        if correction.action == AgentAction.REQUEST_CONTEXT and not state.tool_history:
            trace.correction_succeeded = True
            return correction
        trace.fallback_used = True
        finalization_trace.deterministic_fallback_count += 1
        fallback = _safe_fallback(
            report, decision.final_response.strategy_recommendation, qualitative.valid
        )
        qualitative_trace = state.trace.qualitative_recommendation_validation
        assert qualitative_trace is not None
        qualitative_trace.preserved_after_repair = (
            fallback.final_response is not None
            and fallback.final_response.strategy_recommendation is not None
        )
        return fallback

    async def close(self) -> None:
        """Close provider transports owned by this adapter."""

        await self._client.close()

    async def _complete_with_retry(self, request: dict[str, Any]) -> Any:
        for attempt in range(self._max_retries + 1):
            try:
                return await self._client.chat.completions.create(**request)
            except Exception as exc:
                error = _map_provider_error(exc)
                if not error.retryable or attempt >= self._max_retries:
                    raise error from exc
                await asyncio.sleep(0.25 * (2**attempt))
        raise AssertionError("bounded provider retry loop exhausted unexpectedly")


def _extract_usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def _validate_decision(
    decision: AgentDecision,
    registry: dict[str, int | float | str],
) -> tuple[NumericValidationReport, float]:
    started = perf_counter()
    assert decision.final_response is not None
    report = NumericClaimValidator().validate_answer(
        decision.final_response.recommendation,
        decision.numeric_claims,
        registry,
    )
    return report, (perf_counter() - started) * 1000


def _declaration_complete(decision: AgentDecision) -> bool:
    assert decision.final_response is not None
    return (
        decision.numeric_claims_declared
        or re.search(r"\d", decision.final_response.recommendation) is None
    )


def _validation_trace(
    report: NumericValidationReport,
    latency_ms: float,
    initial_answer: str,
) -> NumericClaimValidationTrace:
    trace = NumericClaimValidationTrace(
        validation_latency_ms=latency_ms,
        initial_answer=initial_answer,
    )
    _replace_trace_counts(trace, report)
    return trace


def _replace_trace_counts(
    trace: NumericClaimValidationTrace,
    report: NumericValidationReport,
) -> None:
    trace.total_claims = len(report.results)
    trace.supported_claims = sum(
        result.status == NumericClaimStatus.SUPPORTED for result in report.results
    )
    trace.unsupported_claims = sum(
        result.status == NumericClaimStatus.UNSUPPORTED for result in report.results
    )
    trace.mismatched_claims = sum(
        result.status == NumericClaimStatus.MISMATCHED for result in report.results
    )
    trace.invalid_reference_claims = sum(
        result.status == NumericClaimStatus.INVALID_REFERENCE for result in report.results
    )
    trace.claims = [
        NumericClaimTrace(
            value=result.claim.value,
            meaning=result.claim.meaning,
            evidence_ref=result.claim.evidence_ref,
            status=result.status,
        )
        for result in report.results
    ]


def _decision_envelope(decision: AgentDecision) -> str:
    assert decision.final_response is not None
    return json.dumps(
        {
            "answer": decision.final_response.recommendation,
            "numeric_claims": [claim.model_dump(mode="json") for claim in decision.numeric_claims],
            "strategy_recommendation": (
                decision.final_response.strategy_recommendation.model_dump(mode="json")
                if decision.final_response.strategy_recommendation
                else None
            ),
        },
        ensure_ascii=False,
    )


def _correction_feedback(
    report: NumericValidationReport, decision: AgentDecision, state: AgentState
) -> str:
    failures = [
        {
            "value": result.claim.value,
            "meaning": result.claim.meaning,
            "evidence_ref": result.claim.evidence_ref,
            "status": result.status.value,
        }
        for result in report.failures
    ]
    if not decision.numeric_claims_declared:
        failures.append({"status": "missing_numeric_claim_envelope"})
    aliases = build_evidence_alias_registry(state)
    strategy_refs = set(_available_strategy_refs(state))
    refs = [alias for alias, ref in aliases.items() if ref in strategy_refs]
    return (
        "Repair only the final output. Unsupported or invalid numeric claims are: "
        f"{json.dumps(failures, ensure_ascii=False)}. Rewrite the answer while preserving "
        "supported "
        "facts and qualitative advice. Remove unsupported precision; do not introduce replacement "
        "numbers. Preserve the original strategy recommendation only with its verified "
        "strategy_recommendation evidence_refs. Return only the segmented supported_facts + "
        "qualitative_recommendation object envelope. "
        f"Verified current-task strategy aliases are: {json.dumps(refs)}. "
        "If a responsible answer requires missing user facts, return the request_context envelope "
        "instead. Do not call tools."
    )


def _available_strategy_refs(state: AgentState) -> list[str]:
    refs: list[str] = []
    for record in state.tool_history:
        if record.tool_call.name != "search_game_knowledge" or not isinstance(
            record.result.data, dict
        ):
            continue
        chunks = record.result.data.get("results")
        if not isinstance(chunks, list):
            continue
        refs.extend(
            f"knowledge.{record.tool_call.id}.{chunk['chunk_id']}"
            for chunk in chunks
            if isinstance(chunk, dict)
            and isinstance(chunk.get("chunk_id"), str)
            and chunk.get("source_type") == "community_strategy"
            and chunk.get("document_type") == "strategy_guide"
            and chunk.get("game_patch") in {None, state.current_patch}
        )
    return refs


def _strategy_recommendation_required(state: AgentState) -> bool:
    return bool(_available_strategy_refs(state))


def _request_context_is_premature(decision: AgentDecision, state: AgentState) -> bool:
    """Reject context requests after successful evidence or for an already present field."""

    if any(record.result.success for record in state.tool_history):
        return True
    registry = build_evidence_registry(state)
    available_fields = {path.rsplit(".", 1)[-1].casefold() for path in registry}
    for missing in decision.missing_context:
        normalized = re.sub(r"[^a-z0-9_]+", " ", missing.casefold())
        if any(field in normalized.split() for field in available_fields if len(field) > 1):
            return True
    # REQUEST_CONTEXT is only for facts the user can supply. Missing corpus coverage,
    # confidence, or explanation is an honest-final condition rather than a waiting state.
    user_context_terms = (
        "level",
        "health",
        "hp",
        "gold",
        "stage",
        "patch",
        "comp",
        "composition",
        "core unit",
        "champion",
        "target",
        "board",
        "bench",
        "item",
        "augment",
        "shop",
        "等级",
        "血量",
        "金币",
        "阶段",
        "版本",
        "阵容",
        "核心",
        "英雄",
        "目标",
        "棋盘",
        "备战席",
        "装备",
        "强化符文",
        "商店",
    )
    return not any(
        term in missing.casefold()
        for missing in decision.missing_context
        for term in user_context_terms
    )


def _safe_fallback(
    report: NumericValidationReport,
    recommendation: StrategyRecommendation | None,
    recommendation_valid: bool,
) -> AgentDecision:
    supported = [
        result.claim for result in report.results if result.status == NumericClaimStatus.SUPPORTED
    ]
    facts = "; ".join(_render_claim(claim) for claim in supported)
    prefix = f"Verified facts: {facts}. " if facts else ""
    preserved = (
        recommendation.model_copy(update={"text": _remove_numeric_precision(recommendation.text)})
        if recommendation is not None and recommendation_valid
        else None
    )
    qualitative = (
        preserved.text
        if preserved is not None
        else (
            "I can provide a qualitative recommendation, but the available evidence does not "
            "support the other precise quantities in the draft."
        )
    )
    return AgentDecision(
        action=AgentAction.FINAL,
        final_response=FinalResponse(
            recommendation=(f"{prefix}{qualitative}"),
            strategy_recommendation=preserved,
        ),
        numeric_claims=supported,
        numeric_claims_declared=True,
    )


def _render_claim(claim: NumericClaim) -> str:
    value: str
    if claim.unit == "probability" and isinstance(claim.value, (int, float)):
        value = f"{float(claim.value) * 100:g}%"
    elif claim.unit == "percent":
        value = f"{claim.value}%"
    else:
        value = str(claim.value)
        if claim.unit:
            value = f"{value} {claim.unit}"
    return f"{claim.meaning} = {value}"


def _remove_numeric_precision(text: str) -> str:
    """Drop numeric precision from an already provenance-valid qualitative conclusion."""

    text = re.sub(r"\bStage\s+\d+-\d+\b", "the current stage", text, flags=re.IGNORECASE)
    text = re.sub(r"\blevel\s+\d+\b", "a higher level", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+-\d+", "当前阶段", text)
    text = re.sub(r"(?<!\w)\d+-\d+\s*阶段", "当前阶段", text)
    text = re.sub(r"升\s*\d+", "立即升级", text)
    text = re.sub(r"\d+(?:\.\d+)*(?:%)?", "", text)
    text = re.sub(r"(?:一|二|三|四|五)星", "完成升级", text)
    text = re.sub(r"[〇零一二三四五六七八九十两]+", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _map_provider_error(exc: Exception) -> LLMProviderError:
    if isinstance(exc, AuthenticationError):
        return LLMProviderError("AUTHENTICATION_FAILED", "DeepSeek authentication failed", False)
    if isinstance(exc, APITimeoutError):
        return LLMProviderError("PROVIDER_TIMEOUT", "DeepSeek request timed out", True)
    if isinstance(exc, RateLimitError):
        return LLMProviderError("RATE_LIMITED", "DeepSeek rate limit exceeded", True)
    if isinstance(exc, BadRequestError):
        return LLMProviderError("INVALID_REQUEST", "DeepSeek rejected the request", False)
    if isinstance(exc, (InternalServerError, APIConnectionError)):
        return LLMProviderError("TRANSIENT_PROVIDER_ERROR", "DeepSeek is unavailable", True)
    if isinstance(exc, APIStatusError):
        retryable = exc.status_code >= 500
        code = "TRANSIENT_PROVIDER_ERROR" if retryable else "PROVIDER_ERROR"
        return LLMProviderError(code, "DeepSeek returned an API error", retryable)
    if isinstance(exc, OpenAIError):
        return LLMProviderError("PROVIDER_ERROR", "DeepSeek request failed", False)
    return LLMProviderError("CLIENT_ERROR", "DeepSeek client failed", False)
