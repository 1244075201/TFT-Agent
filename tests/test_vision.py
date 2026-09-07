import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tft_agent.agent.state import AgentState
from tft_agent.agent.tool_arguments import ToolArgumentValidator
from tft_agent.core.evidence import build_evidence_alias_registry, build_evidence_registry
from tft_agent.core.schemas import GameState, NumericClaimStatus, ToolArgumentClaim, ToolCall
from tft_agent.vision.client import VisionDetail
from tft_agent.vision.deepseek_client import _normalize_visibility_once, _parse_json_once
from tft_agent.vision.exceptions import VisionExtractionError
from tft_agent.vision.merge import merge_game_state
from tft_agent.vision.schemas import (
    VisionExtraction,
    VisionObservation,
    VisionTokenUsage,
    VisionTrace,
)
from tft_agent.vision.workflow import observe_and_merge


def observation(**updates: object) -> VisionObservation:
    payload = {
        "observation_id": "obs-1",
        "image_source": "fixture.png",
        "level": 7,
        "gold": 40,
        "board_units": [{"name": "Ahri", "star_level": 1}],
        "not_visible_fields": ["augments"],
    }
    payload.update(updates)
    return VisionObservation.model_validate(payload)


def test_vision_observation_rejects_invalid_star_and_false_fill() -> None:
    with pytest.raises(ValidationError):
        observation(board_units=[{"name": "Ahri", "star_level": 4}])
    with pytest.raises(ValidationError):
        observation(augments=["Jeweled Lotus"], not_visible_fields=["augments"])


def test_provider_json_is_normalized_at_most_once_without_inventing_facts() -> None:
    payload = _parse_json_once('```json\n{"bench_units":[]}\n```')
    normalized = _normalize_visibility_once({**payload, "not_visible_fields": ["bench_units"]})

    assert normalized["bench_units"] is None
    overlap = _normalize_visibility_once(
        {
            "stage": None,
            "uncertain_fields": ["stage"],
            "not_visible_fields": ["stage"],
        }
    )
    assert overlap["uncertain_fields"] == []
    with pytest.raises(VisionExtractionError):
        _parse_json_once("not json")


def test_uncertain_identity_is_not_autocorrected() -> None:
    observed = observation(
        board_units=[
            {
                "name": None,
                "observed_label": "Ahrl",
                "star_level": 1,
                "identity_status": "uncertain",
            }
        ],
        uncertain_fields=["board_units.0.name"],
    )

    assert observed.board_units is not None
    assert observed.board_units[0].name is None
    assert observed.board_units[0].observed_label == "Ahrl"


def test_merge_fills_missing_state_from_vision() -> None:
    result = merge_game_state(GameState(user_question="decide"), observation())

    assert result.game_state.level == 7
    assert result.game_state.gold == 40
    assert result.game_state.board[0].name == "Ahri"
    assert set(result.vision_filled_fields) >= {"level", "gold", "board"}


def test_merge_records_conflict_and_preserves_explicit_user_fact() -> None:
    result = merge_game_state(GameState(level=8), observation(level=7))

    assert result.game_state.level == 8
    assert result.conflicts[0].field == "level"
    assert result.conflicts[0].user_value == 8
    assert result.conflicts[0].vision_value == 7


def test_not_visible_does_not_fill_missing_game_state() -> None:
    result = merge_game_state(
        GameState(), observation(gold=None, not_visible_fields=["gold", "augments"])
    )

    assert result.game_state.gold is None


def test_uncertain_scalar_does_not_fill_missing_game_state() -> None:
    result = merge_game_state(GameState(), observation(level=7, uncertain_fields=["level"]))

    assert result.game_state.level is None


def test_vision_evidence_is_distinct_and_can_ground_tool_argument() -> None:
    state = AgentState(
        game_state=GameState(level=7),
        goal="roll?",
        vision_observations=[observation()],
    )
    registry = build_evidence_registry(state)
    aliases = build_evidence_alias_registry(state)
    vision_ref = "vision.obs-1.level"
    call = ToolCall(
        id="odds",
        name="get_shop_odds",
        arguments={"level": 7},
        argument_claims=[
            ToolArgumentClaim(argument_name="level", value=7, evidence_ref=vision_ref)
        ],
    )

    assert registry[vision_ref] == 7
    assert vision_ref in aliases.values()
    result = ToolArgumentValidator().validate(call, registry)[0]
    assert result.status == NumericClaimStatus.SUPPORTED


class FakeVisionClient:
    async def extract_game_state(
        self, image_path: Path, detail: VisionDetail = "high"
    ) -> VisionExtraction:
        observed = observation(image_source=image_path.name)
        return VisionExtraction(
            observation=observed,
            trace=VisionTrace(
                observation_id=observed.observation_id,
                image_name=image_path.name,
                image_sha256="a" * 64,
                dimensions=(1280, 720),
                provider="fake",
                model="fake-vision",
                detail=detail,
                extracted_fields=["level", "gold", "board_units"],
                uncertain_fields=[],
                not_visible_fields=["augments"],
                latency_ms=1,
                token_usage=VisionTokenUsage(),
                validation_status="valid",
            ),
        )


@pytest.mark.asyncio
async def test_fake_vision_to_agent_state_workflow() -> None:
    state = AgentState(game_state=GameState(user_question="roll?"), goal="roll?")

    await observe_and_merge(state, Path("fixture.png"), FakeVisionClient())

    assert state.game_state.level == 7
    assert state.game_state.board[0].name == "Ahri"
    assert state.vision_observations[0].observation_id == "obs-1"
    assert state.vision_traces[0].image_sha256 == "a" * 64
    assert json.dumps(state.model_dump(mode="json"))
