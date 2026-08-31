"""Pure metric calculations for tool-selection results."""

from math import ceil
from statistics import fmean, median

from tft_agent.evals.schemas import CaseResult, MetricSummary, ToolSelectionCase


def classify(case: ToolSelectionCase, actual_called: bool) -> str:
    expected_called = case.expected_tool_behavior == "call"
    if expected_called and actual_called:
        return "TP"
    if not expected_called and not actual_called:
        return "TN"
    if not expected_called and actual_called:
        return "FP"
    return "FN"


def summarize(results: list[CaseResult]) -> MetricSummary:
    if not results:
        raise ValueError("cannot summarize an empty result set")
    labels = ("TP", "TN", "FP", "FN")
    counts = {label: sum(result.classification == label for result in results) for label in labels}
    tp, tn, fp, fn = (counts[label] for label in labels)
    latencies = [v for result in results if (v := result.total_latency_ms) is not None]
    inputs = [v for result in results if (v := result.input_tokens) is not None]
    outputs = [v for result in results if (v := result.output_tokens) is not None]
    totals = [v for result in results if (v := result.total_tokens) is not None]
    return MetricSummary(
        case_count=len(results),
        accuracy=(tp + tn) / len(results),
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        false_positive_rate=_ratio(fp, fp + tn),
        false_negative_rate=_ratio(fn, fn + tp),
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        avg_tool_calls=fmean(r.tool_calls for r in results),
        avg_agent_steps=fmean(r.agent_steps for r in results),
        avg_llm_calls=fmean(r.llm_calls for r in results),
        avg_total_latency_ms=fmean(latencies) if latencies else None,
        p50_total_latency_ms=median(latencies) if latencies else None,
        p95_total_latency_ms=_p95(latencies),
        avg_input_tokens=fmean(inputs) if inputs else None,
        avg_output_tokens=fmean(outputs) if outputs else None,
        avg_total_tokens=fmean(totals) if totals else None,
    )


def summarize_by_slice(results: list[CaseResult]) -> dict[str, MetricSummary]:
    return {
        name: summarize([r for r in results if r.slice == name])
        for name in sorted({r.slice for r in results})
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _p95(values: list[float]) -> float | None:
    return sorted(values)[ceil(0.95 * len(values)) - 1] if values else None
