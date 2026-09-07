"""Provider-neutral vision client contract."""

from pathlib import Path
from typing import Literal, Protocol

from tft_agent.vision.schemas import VisionExtraction

VisionDetail = Literal["low", "high", "original", "auto"]


class VisionClient(Protocol):
    async def extract_game_state(
        self, image_path: Path, detail: VisionDetail = "high"
    ) -> VisionExtraction: ...
