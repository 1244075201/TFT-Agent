"""Deterministic user-fact and vision-observation merge policy."""

from typing import Any

from tft_agent.core.schemas import GameState, StreakState, StreakType, UnitState
from tft_agent.vision.schemas import GameStateMergeResult, StateConflict, VisionObservation


def merge_game_state(user: GameState, vision: VisionObservation) -> GameStateMergeResult:
    values = user.model_dump(mode="python")
    conflicts: list[StateConflict] = []
    filled: list[str] = []
    mapped: dict[str, Any] = {
        "stage": vision.stage,
        "level": vision.level,
        "hp": vision.hp,
        "gold": vision.gold,
        "board": _units(vision.board_units),
        "bench": _units(vision.bench_units, expand_copies=True),
        "item_components": vision.item_components,
        "completed_items": vision.completed_items,
        "augments": vision.augments,
        "streak": _streak(vision),
    }
    for field, vision_value in mapped.items():
        if vision_value is None:
            continue
        if field in {path.split(".", 1)[0] for path in vision.uncertain_fields}:
            continue
        user_value = values[field]
        user_missing = user_value is None or (
            user_value == [] and field not in user.model_fields_set
        )
        if user_missing:
            values[field] = vision_value
            filled.append(field)
        elif user_value != vision_value:
            conflicts.append(
                StateConflict(
                    field=field,
                    user_value=user_value,
                    vision_value=vision_value,
                    sources=["user", f"vision:{vision.observation_id}"],
                )
            )
    return GameStateMergeResult(
        game_state=GameState.model_validate(values),
        conflicts=conflicts,
        vision_filled_fields=filled,
    )


def _units(units: list[Any] | None, expand_copies: bool = False) -> list[UnitState] | None:
    if units is None:
        return None
    result: list[UnitState] = []
    for unit in units:
        name = unit.name or unit.observed_label
        assert name is not None
        copies = unit.count if expand_copies else 1
        result.extend(
            UnitState(name=name, star_level=unit.star_level or 1, items=unit.items)
            for _ in range(copies)
        )
    return result


def _streak(vision: VisionObservation) -> StreakState | None:
    if vision.win_streak is not None:
        return StreakState(type=StreakType.WIN, length=vision.win_streak)
    if vision.loss_streak is not None:
        return StreakState(type=StreakType.LOSS, length=vision.loss_streak)
    return None
