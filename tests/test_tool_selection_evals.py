from datetime import UTC, datetime
from pathlib import Path

from tft_agent.evals.metrics import classify, summarize, summarize_by_slice
from tft_agent.evals.schemas import BenchmarkResult, CaseResult, ToolSelectionCase
from tft_agent.evals.tool_selection_runner import load_golden_set, save_result, select_cases


def case(case_id: str, behavior: str = "call", slice_name: str = "direct") -> ToolSelectionCase:
    return ToolSelectionCase.model_validate(
        {
            "id": case_id,
            "slice": slice_name,
            "prompt": case_id,
            "expected_tool_behavior": behavior,
            "expected_tool_name": "get_champion_info" if behavior == "call" else None,
        }
    )


def result(case_id: str, classification: str, latency: float, tokens: int) -> CaseResult:
    expected = "call" if classification in ("TP", "FN") else "no_call"
    actual = "call" if classification in ("TP", "FP") else "no_call"
    return CaseResult.model_validate(
        {
            "id": case_id,
            "slice": "direct",
            "prompt": case_id,
            "expected_behavior": expected,
            "expected_tool_name": "get_champion_info" if expected == "call" else None,
            "actual_behavior": actual,
            "actual_tool_names": ["get_champion_info"] if actual == "call" else [],
            "classification": classification,
            "tool_name_match": True if expected == "call" else None,
            "final_status": "completed",
            "tool_error_codes": [],
            "final_text": "visible",
            "agent_steps": 2 if actual == "call" else 1,
            "tool_calls": int(actual == "call"),
            "llm_calls": 2 if actual == "call" else 1,
            "total_latency_ms": latency,
            "input_tokens": tokens,
            "output_tokens": 10,
            "total_tokens": tokens + 10,
        }
    )


def test_confusion_matrix_and_metrics() -> None:
    results = [
        result("tp", "TP", 10, 100),
        result("tn", "TN", 20, 200),
        result("fp", "FP", 30, 300),
        result("fn", "FN", 40, 400),
    ]
    metrics = summarize(results)
    assert metrics.accuracy == metrics.precision == metrics.recall == 0.5
    assert metrics.false_positive_rate == metrics.false_negative_rate == 0.5
    assert metrics.p50_total_latency_ms == 25
    assert metrics.p95_total_latency_ms == 40
    assert metrics.avg_total_tokens == 260


def test_classification_and_slice_grouping() -> None:
    assert classify(case("a"), True) == "TP"
    assert classify(case("b", "no_call"), True) == "FP"
    grouped = summarize_by_slice([result("one", "TP", 10, 10)])
    assert grouped["direct"].case_count == 1


def test_dataset_filters_and_serialization(tmp_path: Path) -> None:
    golden = load_golden_set(Path("evals/tool_selection_cases.json"))
    assert len(golden.cases) == 36
    assert len(select_cases(golden.cases, "direct_tool_need", None, 2)) == 2
    assert [item.id for item in select_cases(golden.cases, None, "direct_003", None)] == [
        "direct_003"
    ]
    metrics = summarize([result("one", "TP", 10, 10)])
    benchmark = BenchmarkResult(
        benchmark_name="unit",
        model="fake",
        thinking_mode="disabled",
        timestamp=datetime.now(UTC),
        system_prompt_version="test",
        system_prompt_hash="abc",
        tool_description_version="test",
        tool_descriptions={"tool": "description"},
        dataset_version="test",
        dataset_policy="test",
        overall_metrics=metrics,
        per_slice_metrics={"direct": metrics},
        cases=[result("one", "TP", 10, 10)],
    )
    output = save_result(benchmark, tmp_path)
    assert BenchmarkResult.model_validate_json(output.read_text(encoding="utf-8")) == benchmark
