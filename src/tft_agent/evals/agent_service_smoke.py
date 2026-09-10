"""Opt-in AgentService smoke cases with secret-safe observability."""

from __future__ import annotations

import asyncio
import json
from time import perf_counter

from tft_agent.app.aliases import AliasRegistry
from tft_agent.app.pipeline import ProductMode
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatRequest

CASES = [
    "当前版本S级阵容有哪些？",
    "当前版本强势阵容有哪些？",
    "当前版本常见装备和强化有哪些？",
    "地狱火怎么玩？",
    "艾希95怎么玩？",
]


async def run() -> None:
    service = AgentService(Settings())
    for index, question in enumerate(CASES, 1):
        started = perf_counter()
        answer = ""
        async for event in service.ask(ChatRequest(
            text=question,
            session_id=f"canonical-alias-smoke-{index}",
            product_mode=ProductMode.GENERAL_GUIDE,
        )):
            if event.type == "error":
                raise RuntimeError(event.message)
            if event.type == "answer":
                answer = event.message
        trace = service.last_trace
        assert trace is not None
        print(json.dumps({
            "case": index,
            "mode": trace.understanding.product_mode,
            "llm_call_count": trace.llm_call_count,
            "context_type": trace.understanding.knowledge_intent,
            "latency_ms": round((perf_counter() - started) * 1000, 1),
            "answer_summary": answer[:120],
        }, ensure_ascii=False))
    rows = await service._resolver().catalog()
    entities = [
        (str(row["canonical_id"]), str(row["entity_type"]),
         str(row["display_name"]), str(row["source"]))
        for row in rows
    ]
    registry = AliasRegistry.load_default()
    aliases = registry.materialize(
        entities, service.settings.current_set_number, service.settings.current_game_patch
    )
    print(json.dumps({
        "effective_alias_strings": len(aliases),
        "effective_alias_groups": len({(item.canonical_id, item.entity_type) for item in aliases}),
        "rejections": [item.model_dump(mode="json") for item in registry.rejections()],
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(run())
