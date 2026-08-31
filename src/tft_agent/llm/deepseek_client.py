"""DeepSeek Chat Completions adapter implementing the project LLM boundary."""

import asyncio
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
from tft_agent.agent.trace import LLMCallTrace, TokenUsage
from tft_agent.core.config import Settings
from tft_agent.core.exceptions import LLMProviderError
from tft_agent.core.schemas import AgentDecision
from tft_agent.llm.deepseek_messages import (
    build_provider_messages,
    build_provider_tools,
    normalize_provider_response,
)
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
        return decision

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
