"""One-shot local-image vision extraction CLI."""

import argparse
import asyncio
import json
from pathlib import Path

from tft_agent.core.config import Settings
from tft_agent.vision.deepseek_client import DeepSeekVisionClient


async def _extract(path: Path, settings: Settings) -> None:
    client = DeepSeekVisionClient(
        api_key=settings.require_deepseek_api_key().get_secret_value(),
        base_url=str(settings.deepseek_base_url).rstrip("/"),
        model=settings.vision_model,
    )
    try:
        result = await client.extract_game_state(path, settings.vision_detail)
    finally:
        await client.close()
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract observable TFT facts from a local image")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    asyncio.run(_extract(args.image, Settings()))


if __name__ == "__main__":
    main()
