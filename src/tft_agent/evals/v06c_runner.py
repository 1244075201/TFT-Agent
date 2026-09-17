"""Opt-in real DeepSeek smoke for state-aware and conditional strategy decisions."""
from __future__ import annotations

import asyncio
import json
import os
from time import perf_counter

from tft_agent.app.pipeline import ProductMode
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatRequest

SEQUENCES = [
    ["我如果大连败开 没血 经济又没办法支撑我直接上9怎么办"],
    ["厄斐琉斯没有迅射转能玩吗", "那有和没有都怎么构筑呢"],
    ["德莱文没有鱼骨头能玩吗", "那有和没有分别怎么玩"],
    ["升级咯 对冲基金 水乳交融 这三个开局选什么"],
]


async def run() -> None:
    service = AgentService(Settings())
    selected = os.getenv("V06C_SEQUENCE")
    for sequence, queries in enumerate(SEQUENCES, 1):
        if selected and sequence != int(selected):
            continue
        for turn, query in enumerate(queries, 1):
            started = perf_counter()
            answer = ""
            error = ""
            async for event in service.ask(ChatRequest(
                text=query, session_id=f"v06c-real-{sequence}",
                product_mode=ProductMode.GENERAL_GUIDE.value,
            )):
                if event.type == "answer":
                    answer = event.message
                elif event.type == "error":
                    error = event.message
            trace = service.last_trace
            print(json.dumps({
                "sequence": sequence, "turn": turn, "query": query,
                "route": trace.route if trace else None,
                "planner_fallback": trace.planner_used_fallback if trace else None,
                "planner_error": trace.planner_validation_error if trace else None,
                "plan": trace.validated_semantic_plan if trace else None,
                "actions": [call.model_dump(mode="json") for call in trace.repository_calls]
                if trace else [],
                "targets": trace.target_entities if trace else [],
                "conditions": trace.focal_conditions if trace else [],
                "scenarios": trace.scenario_evidence if trace else {},
                "strength": trace.evidence_strength if trace else {},
                "gaps": trace.knowledge_gaps if trace else [],
                "grounding": trace.grounding_violations if trace else [],
                "llm_calls": trace.llm_call_count if trace else 0,
                "latency_ms": round((perf_counter() - started) * 1000, 1),
                "answer": answer, "error": error,
            }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(run())
