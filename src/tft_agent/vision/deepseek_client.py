"""DeepSeek adapter for narrow, structured screenshot observation."""

import base64
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from tft_agent.vision.client import VisionDetail
from tft_agent.vision.exceptions import VisionExtractionError
from tft_agent.vision.schemas import (
    VisionExtraction,
    VisionObservation,
    VisionTokenUsage,
    VisionTrace,
)

VISION_PROMPT_VERSION = "v0.5.0"
VISION_PROMPT = """You are extracting observable TFT game-state facts.
Only report what is visibly present. Do not provide strategy. Do not infer hidden facts, unit
strength, meta tier, invisible items, or missing board units. Unknown or ambiguous unit text must
remain unresolved and be listed in uncertain_fields; never autocorrect it from game knowledge.
Use null for a scalar or collection whose UI region is not visible and include its field name in
not_visible_fields. A visible empty region is an empty list, not null. Do not output confidence.
Return one JSON object matching this shape:
{"stage":null,"level":null,"hp":null,"gold":null,"board_units":null,"bench_units":null,
"item_components":null,"completed_items":null,"augments":null,"win_streak":null,
"loss_streak":null,"uncertain_fields":[],"not_visible_fields":[]}.
Unit entries use {"name":string|null,"observed_label":string|null,"star_level":integer|null,
"count":integer,"items":[],"identity_status":"observed"|"uncertain"}.
Do not include observation_id or image_source; the application supplies them."""


class DeepSeekVisionClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def extract_game_state(
        self, image_path: Path, detail: VisionDetail = "high"
    ) -> VisionExtraction:
        path = image_path.resolve()
        image_bytes, mime, dimensions = _load_image(path)
        digest = hashlib.sha256(image_bytes).hexdigest()
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        started = perf_counter()
        try:
            messages: Any = [
                {"role": "system", "content": VISION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract only visible TFT facts as JSON."},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": detail},
                        },
                    ],
                },
            ]
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            raise _provider_error(exc) from exc
        latency_ms = (perf_counter() - started) * 1000
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise VisionExtractionError("MALFORMED_JSON", "Vision response contained no JSON")
        payload = _parse_json_once(content)
        payload = _normalize_visibility_once(payload)
        payload["observation_id"] = getattr(response, "id", None) or digest[:16]
        payload["image_source"] = path.name
        try:
            observation = VisionObservation.model_validate(payload)
        except ValidationError as exc:
            raise VisionExtractionError("VALIDATION_FAILED", str(exc)) from exc
        usage = getattr(response, "usage", None)
        token_usage = (
            VisionTokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            )
            if usage is not None
            else None
        )
        return VisionExtraction(
            observation=observation,
            trace=VisionTrace(
                observation_id=observation.observation_id,
                image_name=path.name,
                image_sha256=digest,
                dimensions=dimensions,
                provider="deepseek",
                model=self._model,
                detail=detail,
                extracted_fields=_extracted_fields(observation),
                uncertain_fields=observation.uncertain_fields,
                not_visible_fields=observation.not_visible_fields,
                latency_ms=latency_ms,
                token_usage=token_usage,
                validation_status="valid",
            ),
        )

    async def close(self) -> None:
        await self._client.close()


def _load_image(path: Path) -> tuple[bytes, str, tuple[int, int]]:
    if not path.is_file():
        raise VisionExtractionError("IMAGE_NOT_FOUND", f"Image does not exist: {path}")
    size = path.stat().st_size
    if size > 32 * 1024 * 1024:
        raise VisionExtractionError("IMAGE_TOO_LARGE", "Inline image exceeds 32 MiB")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image_format = image.format
            dimensions = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise VisionExtractionError("INVALID_IMAGE", "Invalid or unreadable image") from exc
    mime_by_format = {"PNG": "image/png", "JPEG": "image/jpeg"}
    if image_format not in mime_by_format:
        raise VisionExtractionError("UNSUPPORTED_IMAGE", "Only PNG and JPEG are supported")
    return path.read_bytes(), mime_by_format[image_format], dimensions


def _parse_json_once(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        normalized = content.strip()
        if normalized.startswith("```json"):
            normalized = normalized[7:]
        elif normalized.startswith("```"):
            normalized = normalized[3:]
        if normalized.endswith("```"):
            normalized = normalized[:-3]
        try:
            payload = json.loads(normalized.strip())
        except json.JSONDecodeError as exc:
            raise VisionExtractionError("MALFORMED_JSON", "Vision response is not JSON") from exc
    if not isinstance(payload, dict):
        raise VisionExtractionError("MALFORMED_JSON", "Vision response must be a JSON object")
    return payload


def _normalize_visibility_once(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve only the harmless [] + explicit not_visible representation mismatch."""

    normalized = dict(payload)
    not_visible = normalized.get("not_visible_fields")
    if not isinstance(not_visible, list):
        return normalized
    for path in not_visible:
        if not isinstance(path, str):
            continue
        root = path.split(".", 1)[0]
        if normalized.get(root) == []:
            normalized[root] = None
    uncertain = normalized.get("uncertain_fields")
    if isinstance(uncertain, list):
        normalized["uncertain_fields"] = [
            path
            for path in uncertain
            if path not in not_visible or normalized.get(path.split(".", 1)[0]) is not None
        ]
    return normalized


def _provider_error(exc: Exception) -> VisionExtractionError:
    if isinstance(exc, APITimeoutError):
        return VisionExtractionError("PROVIDER_TIMEOUT", "Vision provider timed out", True)
    if isinstance(exc, APIConnectionError):
        return VisionExtractionError("PROVIDER_UNAVAILABLE", "Vision provider unavailable", True)
    if isinstance(exc, APIStatusError):
        code = {402: "INSUFFICIENT_BALANCE", 429: "RATE_LIMITED"}.get(
            exc.status_code, "PROVIDER_ERROR"
        )
        retryable = exc.status_code >= 500 or exc.status_code == 429
        return VisionExtractionError(
            code, f"Vision provider returned HTTP {exc.status_code}", retryable
        )
    if isinstance(exc, httpx.TimeoutException):
        return VisionExtractionError("PROVIDER_TIMEOUT", "Vision provider timed out", True)
    return VisionExtractionError("PROVIDER_ERROR", "Vision provider request failed")


def _extracted_fields(observation: VisionObservation) -> list[str]:
    excluded = {"observation_id", "image_source", "uncertain_fields", "not_visible_fields"}
    values = observation.model_dump(mode="json")
    return [key for key, value in values.items() if key not in excluded and value is not None]
