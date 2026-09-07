"""Composition helpers that keep observation separate from agent reasoning."""

from pathlib import Path

from tft_agent.agent.state import AgentState
from tft_agent.vision.client import VisionClient, VisionDetail
from tft_agent.vision.merge import merge_game_state


async def observe_and_merge(
    state: AgentState,
    image_path: Path,
    client: VisionClient,
    detail: VisionDetail = "high",
) -> AgentState:
    extraction = await client.extract_game_state(image_path, detail)
    merged = merge_game_state(state.game_state, extraction.observation)
    state.game_state = merged.game_state
    state.vision_observations.append(extraction.observation)
    state.vision_traces.append(extraction.trace)
    state.state_conflicts.extend(merged.conflicts)
    return state
