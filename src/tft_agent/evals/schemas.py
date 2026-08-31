"""Typed models for tool-selection evaluation."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from tft_agent.core.schemas import StrictModel

ToolBehavior = Literal["call", "no_call"]
Classification = Literal["TP", "TN", "FP", "FN"]


class ToolSelectionCase(StrictModel):
    id: str = Field(min_length=1)
    slice: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_tool_behavior: ToolBehavior
    expected_tool_name: str | None
    notes: str = ""

    @model_validator(mode="after")
    def validate_expected_tool(self) -> "ToolSelectionCase":
        if self.expected_tool_behavior == "call" and not self.expected_tool_name:
            raise ValueError("call cases require expected_tool_name")
        if self.expected_tool_behavior == "no_call" and self.expected_tool_name is not None:
            raise ValueError("no_call cases cannot specify expected_tool_name")
        return self


class GoldenSet(StrictModel):
    dataset_version: str
    policy: str
    cases: list[ToolSelectionCase]


class CaseResult(StrictModel):
    id: str
    slice: str
    prompt: str
    expected_behavior: ToolBehavior
    expected_tool_name: str | None
    actual_behavior: ToolBehavior
    actual_tool_names: list[str]
    classification: Classification
    tool_name_match: bool | None
    final_status: str
    tool_error_codes: list[str]
    final_text: str | None
    agent_steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    total_latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class MetricSummary(StrictModel):
    case_count: int
    accuracy: float
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    avg_tool_calls: float
    avg_agent_steps: float
    avg_llm_calls: float
    avg_total_latency_ms: float | None
    p50_total_latency_ms: float | None
    p95_total_latency_ms: float | None
    avg_input_tokens: float | None
    avg_output_tokens: float | None
    avg_total_tokens: float | None


class BenchmarkResult(StrictModel):
    benchmark_name: str
    model: str
    thinking_mode: str
    timestamp: datetime
    system_prompt_version: str
    system_prompt_hash: str
    tool_description_version: str
    tool_descriptions: dict[str, str]
    dataset_version: str
    dataset_policy: str
    overall_metrics: MetricSummary
    per_slice_metrics: dict[str, MetricSummary]
    cases: list[CaseResult]
