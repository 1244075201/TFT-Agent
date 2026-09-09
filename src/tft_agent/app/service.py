"""Unified application service shared by CLI and desktop frontends."""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from tft_agent.agent.parser import GameStateParser
from tft_agent.agent.state import AgentMessage
from tft_agent.app.pipeline import (
    AnswerContext,
    ProductMode,
    QueryKind,
    RetrievalPlan,
    RetrievedKnowledge,
    build_retrieval_plan,
    comp_payload,
    context_gaps,
    entity_payload,
    understand_query,
)
from tft_agent.core.config import Settings
from tft_agent.core.schemas import (
    AdviceEvidence,
    AdviceOption,
    ChatEvent,
    ChatMode,
    ChatRequest,
    FinalResponse,
    GameState,
)
from tft_agent.llm.grounded_decision import GroundedDecisionClient
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.schemas import CompStatsRecord
from tft_agent.vision.deepseek_client import DeepSeekVisionClient
from tft_agent.vision.workflow import observe_and_merge


class AgentService:
    """Run one deterministic retrieval pipeline and, for decisions, one LLM call."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._cancelled_sessions: set[str] = set()
        self._repository: SQLiteGameDataRepository | None = None

    def _repo(self) -> SQLiteGameDataRepository:
        if self._repository is None:
            self._repository = SQLiteGameDataRepository(self.settings.database_path.resolve())
            self._repository.initialize_sync()
        return self._repository

    def cancel(self, session_id: str) -> None:
        self._cancelled_sessions.add(session_id)

    async def data_status(self) -> str:
        if not self.settings.database_path.resolve().is_file():
            return "DataTFT：尚未同步"
        latest = await self._repo().latest_datatft_snapshot()
        if latest is None:
            return "DataTFT：尚未同步"
        metadata, count = latest
        timestamp = metadata.retrieved_at.astimezone().strftime("%Y-%m-%d %H:%M")
        return (f"版本：{metadata.game_patch}｜DataTFT：已同步 {count} 套阵容｜"
                f"更新时间：{timestamp}｜状态：{metadata.status.value}")

    async def clear_session(self, session_id: str) -> None:
        await self._repo().clear_session(session_id)

    async def session_summary(self, session_id: str) -> str:
        stored = await self._repo().load_session_state(session_id)
        if stored is None:
            return "当前局面：尚未记录"
        state = GameState.model_validate_json(stored[0])
        parts = [value for value in (
            f"阵容 {state.comp_name}" if state.comp_name else None,
            f"阶段 {state.stage}" if state.stage else None,
            f"{state.level}级" if state.level else None,
            f"{state.gold}金币" if state.gold is not None else None,
            "强化 " + "/".join(state.augments) if state.augments else None,
        ) if value]
        return "当前局面：" + ("｜".join(parts) if parts else "尚未记录")

    async def load_game_state(self, session_id: str) -> GameState:
        stored = await self._repo().load_session_state(session_id)
        return GameState.model_validate_json(stored[0]) if stored else GameState()

    async def save_game_state(self, session_id: str, state: GameState) -> None:
        sources = {field: "user" for field in state.model_fields_set}
        await self._repo().save_session_state(
            session_id, state.model_dump_json(), json.dumps(sources, ensure_ascii=False)
        )

    async def ask(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        if (
            request.mode == ChatMode.EXPERIMENTAL_LIVE
            and not self.settings.experimental_live_enabled
        ):
            yield ChatEvent(type="error", message="实验实时模式默认关闭。")
            return
        self._cancelled_sessions.discard(request.session_id)
        yield ChatEvent(type="status", message="正在理解问题并规划检索…")
        if request.confirmed_game_state is not None:
            state = request.confirmed_game_state
        else:
            state = await self._merge_state(
                request.session_id, GameStateParser().parse_text(request.text)
            )
        vision: DeepSeekVisionClient | None = None
        try:
            if request.screenshot_path is not None:
                vision = DeepSeekVisionClient(
                    self.settings.require_deepseek_api_key().get_secret_value(),
                    str(self.settings.deepseek_base_url), self.settings.vision_model,
                )
                from tft_agent.agent.state import AgentState
                observed = await observe_and_merge(
                    AgentState(game_state=state, goal=request.text), request.screenshot_path,
                    vision, self.settings.vision_detail,
                )
                state = observed.game_state
                yield ChatEvent(type="observation", message="截图识别完成", game_state=state)
            if request.session_id in self._cancelled_sessions:
                yield ChatEvent(type="cancelled", message="请求已取消")
                return
            if not self.settings.database_path.resolve().is_file():
                yield ChatEvent(type="error", message="本地数据尚未初始化，请先运行数据同步命令。")
                return
            mode = ProductMode(request.product_mode)
            understanding = understand_query(request.text, state, mode)
            plan = build_retrieval_plan(understanding, state, self.settings.current_game_patch)
            history_rows = await self._repo().load_session_messages(request.session_id)
            history = [AgentMessage(role=role, content=content) for role, content in history_rows]
            knowledge = await self._retrieve(request.text, plan)
            await self.save_game_state(request.session_id, state)
            gaps = context_gaps(state)
            trace = plan.model_dump(mode="json")
            yield ChatEvent(type="status", message="检索完成", retrieval_plan=trace,
                            context_gaps=gaps)
            if understanding.kind == QueryKind.DETERMINISTIC_LOOKUP:
                answer = _deterministic_answer(knowledge, self.settings.current_game_patch)
                await self._record_answer(request, answer)
                yield ChatEvent(type="answer", message=answer.recommendation, response=answer,
                                advice_options=_advice_options(answer), retrieval_plan=trace,
                                context_gaps=gaps, llm_call_count=0)
                return
            if _critical_missing(request.text, state):
                yield ChatEvent(
                    type="error",
                    message="还需要目标棋子或费用，才能判断当前搜牌概率。",
                    retrieval_plan=trace,
                    context_gaps=gaps,
                    llm_call_count=0,
                )
                return
            context = AnswerContext(
                question=request.text, understanding=understanding, game_state=state,
                history=history, retrieval_plan=plan, knowledge=knowledge, context_gaps=gaps,
            )
            yield ChatEvent(type="status", message="正在生成一次有证据的建议…",
                            retrieval_plan=trace, context_gaps=gaps)
            client = GroundedDecisionClient(self.settings)
            try:
                outcome = await client.decide_context(context)
            finally:
                await client.close()
            await self._record_answer(request, outcome.response)
            yield ChatEvent(type="answer", message=outcome.response.recommendation,
                            response=outcome.response, advice_options=outcome.options,
                            retrieval_plan=trace, context_gaps=gaps, llm_call_count=1)
        except asyncio.CancelledError:
            yield ChatEvent(type="cancelled", message="请求已取消")
        except Exception as exc:
            yield ChatEvent(type="error", message=f"请求失败：{exc}")
        finally:
            if vision is not None:
                await vision.close()

    async def _merge_state(self, session_id: str, current: GameState) -> GameState:
        stored = await self._repo().load_session_state(session_id)
        if stored is None:
            return current
        previous = GameState.model_validate_json(stored[0])
        update = {field: getattr(current, field) for field in current.model_fields_set
                  if field != "user_question"}
        update["user_question"] = current.user_question
        return previous.model_copy(update=update)

    async def _retrieve(self, question: str, plan: RetrievalPlan) -> RetrievedKnowledge:
        comps_task = self._repo().search_comps(plan.comp_query, plan.game_patch,
                                                8 if plan.retrieve_all_comps else 3)
        entities_task = self._repo().search_static_entities(plan.static_entity_names)
        text_task = self._repo().search_knowledge_text(
            question, plan.game_patch, 6, plan.comp_query
        ) if plan.retrieve_strategy_text else asyncio.sleep(0, result=[])
        odds_task = self._repo().get_shop_odds(plan.shop_odds_level, plan.game_patch) \
            if plan.shop_odds_level else asyncio.sleep(0, result=None)
        comps, entities, chunks, odds = await asyncio.gather(
            comps_task, entities_task, text_task, odds_task
        )
        return RetrievedKnowledge(
            comps=[comp_payload(item) for item in comps],
            static_entities=[entity_payload(item) for item in entities],
            strategy_chunks=chunks,
            shop_odds=odds.model_dump(mode="json") if odds is not None else None,
        )

    async def _record_answer(self, request: ChatRequest, answer: FinalResponse) -> None:
        await self._repo().append_session_messages(request.session_id, [
            ("user", request.text), ("assistant", answer.recommendation),
        ])


def _critical_missing(question: str, state: GameState) -> bool:
    roll = bool(re.search(r"搜牌|要不要搜|该不该搜|roll", question, re.I))
    return roll and state.target_champion is None and state.target_cost is None


def _deterministic_answer(knowledge: RetrievedKnowledge, patch: str) -> FinalResponse:
    if not knowledge.comps:
        return FinalResponse(recommendation=f"本地数据库没有找到 {patch} 的对应阵容资料。")
    lines = []
    for comp in knowledge.comps:
        line = f"• {comp['name']}（{comp.get('tier') or '未分级'}级）"
        if comp.get("core_units"):
            line += "｜核心：" + "、".join(comp["core_units"])
        if comp.get("recommended_items"):
            line += "｜装备：" + "、".join(comp["recommended_items"][:4])
        if comp.get("augments"):
            line += "｜强化：" + "、".join(comp["augments"][:3])
        if comp.get("level_plan"):
            line += "｜运营：" + "；".join(comp["level_plan"][:3])
        lines.append(line)
    snapshot = knowledge.comps[0].get("snapshot", {})
    url = snapshot.get("source_url") if isinstance(snapshot, dict) else None
    return FinalResponse(
        recommendation=f"当前版本 {patch} 阵容资料：\n" + "\n".join(lines), confidence=1,
        used_tools=["structured_sqlite_retrieval"],
        sources=[f"DataTFT {patch} — {url}"] if url else [],
    )


def _advice_options(response: FinalResponse) -> list[AdviceOption]:
    evidence = [AdviceEvidence(label=s, url=s.rsplit(" — ", 1)[-1], source="retrieved")
                for s in response.sources if " — http" in s]
    return [AdviceOption(title="结果", action=response.recommendation, conditions=[],
                         expected_benefit="基于当前版本本地数据。",
                         risk="数据覆盖不足时需要补充信息。",
                         confidence=response.confidence or 0.5, evidence=evidence)]


# Compatibility helpers retained for one deprecation cycle; production does not call them.
def _datatft_direct_answer(question: str, records: list[CompStatsRecord]) -> FinalResponse | None:
    decision = any(term in question.casefold() for term in (
        "怎么玩", "为什么", "推荐", "比较", "选择", "怎么选", "该不该", "怎么办", "转阵"
    ))
    if decision or not records:
        return None
    return _deterministic_answer(
        RetrievedKnowledge(comps=[comp_payload(item) for item in records]),
        records[0].snapshot.game_patch,
    )


def _related_comp(question: str, history: list[AgentMessage],
                  records: list[CompStatsRecord]) -> CompStatsRecord | None:
    for text in [question, *(message.content for message in reversed(history)
                              if message.role == "user")]:
        match = next((item for item in records if item.name.casefold() in text.casefold()), None)
        if match:
            return match
    return None
