"""Window-independent screen capture and stable visual-change detection."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from PIL import Image, ImageChops, ImageStat
from pydantic import Field

from tft_agent.core.schemas import StrictModel


class ScreenCalibration(StrictModel):
    """Normalized own-board crop, portable across resolutions and display scaling."""

    monitor_index: int = Field(default=1, ge=1)
    left: float = Field(default=0, ge=0, le=1)
    top: float = Field(default=0, ge=0, le=1)
    width: float = Field(default=1, gt=0, le=1)
    height: float = Field(default=1, gt=0, le=1)


@dataclass(frozen=True)
class ScreenFrame:
    png_bytes: bytes
    sha256: str
    dimensions: tuple[int, int]
    captured_at_monotonic: float


class ScreenMonitor(Protocol):
    def capture(self) -> ScreenFrame: ...

    def is_stable_change(self, frame: ScreenFrame) -> bool: ...


class MSSScreenMonitor:
    """Capture pixels only; it never reads memory, hooks processes, or sends input."""

    def __init__(
        self,
        calibration: ScreenCalibration | None = None,
        change_threshold: float = 8.0,
        stable_frames: int = 2,
    ) -> None:
        self.calibration = calibration or ScreenCalibration()
        self.change_threshold = change_threshold
        self.stable_frames = stable_frames
        self._baseline: Image.Image | None = None
        self._candidate_hash: str | None = None
        self._candidate_count = 0

    def capture(self) -> ScreenFrame:
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError("屏幕监控需要安装 mss：uv sync --all-extras") from exc
        with mss.mss() as capture:
            monitors = capture.monitors
            if self.calibration.monitor_index >= len(monitors):
                raise ValueError("校准的显示器当前不可用")
            source = monitors[self.calibration.monitor_index]
            region = {
                "left": source["left"] + round(source["width"] * self.calibration.left),
                "top": source["top"] + round(source["height"] * self.calibration.top),
                "width": round(source["width"] * self.calibration.width),
                "height": round(source["height"] * self.calibration.height),
            }
            shot = capture.grab(region)
            image = Image.frombytes("RGB", shot.size, shot.rgb)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        content = output.getvalue()
        return ScreenFrame(content, hashlib.sha256(content).hexdigest(), image.size, monotonic())

    def is_stable_change(self, frame: ScreenFrame) -> bool:
        image = Image.open(io.BytesIO(frame.png_bytes)).convert("L").resize((160, 90))
        if self._baseline is None:
            self._baseline = image
            return False
        score = ImageStat.Stat(ImageChops.difference(self._baseline, image)).mean[0]
        if score < self.change_threshold:
            self._candidate_hash = None
            self._candidate_count = 0
            return False
        perceptual = hashlib.sha256(image.tobytes()).hexdigest()
        if perceptual != self._candidate_hash:
            self._candidate_hash = perceptual
            self._candidate_count = 1
            return False
        self._candidate_count += 1
        if self._candidate_count < self.stable_frames:
            return False
        self._baseline = image
        self._candidate_hash = None
        self._candidate_count = 0
        return True
