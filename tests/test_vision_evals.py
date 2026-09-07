from pathlib import Path

from tft_agent.evals.vision_extraction_v05_runner import (
    VisionDataset,
    aggregate,
    score_observation,
)
from tft_agent.vision.schemas import VisionObservation


def _observation(payload: dict[str, object], suffix: str) -> VisionObservation:
    return VisionObservation.model_validate(
        {**payload, "observation_id": suffix, "image_source": f"{suffix}.png"}
    )


def test_controlled_vision_dataset_has_required_slices_and_files() -> None:
    dataset = VisionDataset.model_validate_json(
        Path("evals/vision_extraction_v0_5_cases.json").read_text(encoding="utf-8")
    )

    assert len(dataset.cases) == 8
    assert {case.slice for case in dataset.cases} >= {
        "full_board",
        "hud_only",
        "board_without_bench",
        "blurred_low_resolution",
        "tool_explosion",
    }
    assert all(case.image.is_file() for case in dataset.cases)


def test_vision_metrics_score_exact_and_hallucinated_fields() -> None:
    truth = _observation(
        {
            "level": 7,
            "board_units": [{"name": "Ahri", "star_level": 1}],
            "bench_units": None,
            "not_visible_fields": ["bench_units", "gold"],
        },
        "truth",
    )
    predicted = _observation(
        {
            "level": 7,
            "gold": 40,
            "board_units": [{"name": "Ahri", "star_level": 1}],
            "bench_units": None,
            "not_visible_fields": ["bench_units"],
        },
        "predicted",
    )

    score = score_observation(predicted, truth)
    metrics = aggregate([score])

    assert metrics["level_exact_accuracy"] == 1
    assert metrics["board"]["f1"] == 1
    assert metrics["hallucinated_field_rate"] > 0
    assert metrics["not_visible_false_fill_rate"] == 0.5
