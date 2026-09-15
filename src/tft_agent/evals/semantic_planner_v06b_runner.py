"""Opt-in real DeepSeek smoke for the bounded V0.6B planner experiment."""
from __future__ import annotations

import asyncio
import json
from time import perf_counter

from tft_agent.app.pipeline import ProductMode
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatRequest

CASES = [
    "这版本两费主c的阵容有什么",
    "三费主C能玩什么",
    "我想玩节奏快一点的阵容",
    "我想玩稳一点容易成型的",
    "有没有不太吃五费卡的阵容",
    "我不想赌三星，有什么低费核心阵容",
    "有哪些阵容能吃烂分 哪些能吃鸡",
    "主宰转怎么运营",
    "推荐20套阵容",
]


async def run() -> None:
    service = AgentService(Settings())
    for index, query in enumerate(CASES, 1):
        started = perf_counter()
        answer = ""
        error = ""
        async for event in service.ask(ChatRequest(
            text=query,
            session_id=f"v06b-real-{index}",
            product_mode=ProductMode.GENERAL_GUIDE.value,
        )):
            if event.type == "answer":
                answer = event.message
            elif event.type == "error":
                error = event.message
        trace = service.last_trace
        if trace is None:
            raise RuntimeError(error or "missing trace")
        print(json.dumps({
            "case": index,
            "query": query,
            "route": trace.route,
            "planner_invoked": trace.planner_invoked,
            "planner_fallback": trace.planner_used_fallback,
            "planner_validation_error": trace.planner_validation_error,
            "plan": trace.validated_semantic_plan,
            "patch": trace.retrieval_plan.game_patch,
            "snapshot": trace.retrieval_plan.active_snapshot_id,
            "candidates": trace.candidate_compositions,
            "evidence_counts": trace.evidence_counts_by_type,
            "grounding_passed": trace.grounding_passed,
            "grounding_violations": trace.grounding_violations,
            "grounding_fallback": trace.grounding_fallback_used,
            "llm_calls": trace.llm_call_count,
            "stage_latency_ms": {key: round(value, 1)
                                 for key, value in trace.stage_latency_ms.items()},
            "wall_latency_ms": round((perf_counter() - started) * 1000, 1),
            "answer": answer,
            "error": error,
        }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(run())
