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
    TokenUsage,
)
from tft_agent.core.config import Settings
from tft_agent.core.exceptions import LLMProviderError
from tft_agent.core.schemas import (
    AgentAction,
    AgentDecision,
    FinalResponse,
    NumericClaimStatus,
)
from tft_agent.llm.deepseek_messages import (
    build_provider_messages,
    build_provider_tools,
    normalize_provider_response,
)
from tft_agent.llm.evidence import build_evidence_registry
from tft_agent.llm.numeric_claims import NumericClaimValidator, NumericValidationReport
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
        decision = normalize_provider_response(response, tool_list, state)
        state.trace.llm_calls.append(
            LLMCallTrace(
                call_index=len(state.trace.llm_calls) + 1,
                model=self._model,
                latency_ms=latency_ms,
                outcome=decision.action.value,
                token_usage=_extract_usage(response),
            )
        )
        if decision.action != AgentAction.FINAL:
            return decision
        return await self._enforce_numeric_claims(state, tool_list, request, decision)

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
        trace = _validation_trace(
            report,
            validation_ms,
            decision.final_response.recommendation,
        )
        state.trace.numeric_claim_validation = trace
        if report.valid and _declaration_complete(decision):
            return decision

        trace.correction_attempted = True
        correction_request = {
            "model": self._model,
            "messages": [
                *initial_request["messages"],
                {"role": "assistant", "content": _decision_envelope(decision)},
                {"role": "user", "content": _correction_feedback(report, decision)},
            ],
            "extra_body": {"thinking": {"type": self._thinking_mode}},
        }
        started = perf_counter()
        try:
            response = await self._complete_with_retry(correction_request)
            correction = normalize_provider_response(response, [], state)
        except LLMProviderError:
            trace.correction_latency_ms = (perf_counter() - started) * 1000
            trace.fallback_used = True
            return _safe_fallback()
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
            trace.validation_latency_ms += corrected_validation_ms
            if corrected_report.valid and _declaration_complete(correction):
                trace.correction_succeeded = True
                return correction
        trace.fallback_used = True
        return _safe_fallback()

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
        },
        ensure_ascii=False,
    )


def _correction_feedback(report: NumericValidationReport, decision: AgentDecision) -> str:
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
    return (
        "Repair only the final output. Unsupported or invalid numeric claims are: "
        f"{json.dumps(failures, ensure_ascii=False)}. Rewrite the answer while preserving "
        "supported "
        "facts and qualitative advice. Remove unsupported precision; do not introduce replacement "
        "numbers. Return only the small answer + numeric_claims envelope. Do not call tools."
    )


def _safe_fallback() -> AgentDecision:
    return AgentDecision(
        action=AgentAction.FINAL,
        final_response=FinalResponse(
            recommendation=(
                "I can provide a qualitative recommendation, but the available evidence does not "
                "support the precise numerical claims in the draft."
            )
        ),
        numeric_claims_declared=True,
    )


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
