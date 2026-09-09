"""Unified application service with plan-driven, clean retrieval."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from time import perf_counter

from tft_agent.agent.parser import GameStateParser
from tft_agent.agent.state import AgentMessage, AgentState
from tft_agent.app.entity_resolution import CanonicalEntityResolver, ResolvedEntity
from tft_agent.app.pipeline import (
    AnswerContext,
    EvidenceRecord,
    ProductMode,
    QueryKind,
    QueryStateUnderstanding,
    RepositoryCall,
    RetrievalPlan,
    RetrievalTrace,
    RetrievedKnowledge,
    StateMutationIntent,
    apply_state_mutation,
    build_retrieval_plan,
    comp_evidence,
    context_gaps,
    static_evidence,
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
from tft_agent.vision.deepseek_client import DeepSeekVisionClient
from tft_agent.vision.workflow import observe_and_merge


class AgentService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._cancelled_sessions: set[str] = set()
        self._repository: SQLiteGameDataRepository | None = None
        self._entity_resolver: CanonicalEntityResolver | None = None
        self.last_trace: RetrievalTrace | None = None

    def _repo(self) -> SQLiteGameDataRepository:
        if self._repository is None:
            self._repository = SQLiteGameDataRepository(self.settings.database_path.resolve())
            self._repository.initialize_sync()
        return self._repository

    def _resolver(self) -> CanonicalEntityResolver:
        if self._entity_resolver is None:
            self._entity_resolver = CanonicalEntityResolver(self._repo(), self.settings)
        return self._entity_resolver

    def cancel(self, session_id: str) -> None:
        self._cancelled_sessions.add(session_id)

    async def data_status(self) -> str:
        if not self.settings.database_path.resolve().is_file():
            return "DataTFT：尚未同步"
        latest = await self._repo().latest_datatft_snapshot()
        if latest is None:
            return "DataTFT：尚未同步"
        metadata, count = latest
        return (f"版本：{metadata.game_patch}｜DataTFT：已同步 {count} 套阵容｜"
                f"更新时间：{metadata.retrieved_at.astimezone():%Y-%m-%d %H:%M}｜"
                f"状态：{metadata.status.value}")

    async def clear_session(self, session_id: str) -> None:
        await self._repo().clear_session(session_id)

    async def session_summary(self, session_id: str) -> str:
        state = await self.load_game_state(session_id)
        parts = [value for value in (
            f"阵容 {state.comp_name}" if state.comp_name else None,
            f"阶段 {state.stage}" if state.stage else None,
            f"{state.level}级" if state.level else None,
            f"{state.gold}金币" if state.gold is not None else None,
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
        started = perf_counter()
        yield ChatEvent(type="status", message="正在解析实体并规划检索…")
        persisted_state = await self.load_game_state(request.session_id)
        parsed_state = request.confirmed_game_state or GameStateParser().parse_text(request.text)
        state = request.confirmed_game_state or _merge_game_state(persisted_state, parsed_state)
        vision: DeepSeekVisionClient | None = None
        try:
            if request.screenshot_path is not None:
                vision = DeepSeekVisionClient(
                    self.settings.require_deepseek_api_key().get_secret_value(),
                    str(self.settings.deepseek_base_url), self.settings.vision_model,
                )
                observed = await observe_and_merge(
                    AgentState(game_state=state, goal=request.text), request.screenshot_path,
                    vision, self.settings.vision_detail,
                )
                state = observed.game_state
                yield ChatEvent(type="observation", message="截图识别完成", game_state=state)
            entity_start = perf_counter()
            resolution = await self._resolver().resolve(request.text)
            understanding = understand_query(
                request.text, state, ProductMode(request.product_mode), resolution
            )
            if (
                understanding.state_mutation == StateMutationIntent.RETAIN
                and state.comp_name
            ):
                retained = await self._resolver().resolve(state.comp_name)
                retained_comp = next((item for item in retained.entities
                                      if item.entity_type == "composition"), None)
                if retained_comp:
                    understanding = understanding.model_copy(
                        update={"target_comp": retained_comp}
                    )
            state = apply_state_mutation(state, understanding, request.text)
            entity_end = perf_counter()
            plan = build_retrieval_plan(
                understanding, state, self.settings.current_game_patch,
                self.settings.current_set_number, self.settings.current_set_name,
            )
            plan_end = perf_counter()
            knowledge, calls = await self._retrieve(request.text, plan)
            retrieval_end = perf_counter()
            history_rows = await self._repo().load_session_messages(request.session_id)
            history = _relevant_history(history_rows, understanding)
            gaps = context_gaps(state)
            context = AnswerContext(
                question=request.text, understanding=understanding, game_state=state,
                history=history, retrieval_plan=plan, evidence=knowledge.evidence,
                context_gaps=gaps, knowledge_gaps=knowledge.knowledge_gaps,
            )
            context_summary = context.readable_summary()
            context_end = perf_counter()
            trace = RetrievalTrace(
                original_query=request.text, resolved_entities=resolution.entities,
                understanding=understanding, retrieval_plan=plan, repository_calls=calls,
                retrieved_knowledge=knowledge, knowledge_gaps=knowledge.knowledge_gaps,
                clean_context_summary=context_summary, llm_call_count=0,
                persisted_state_before=persisted_state, parsed_state=parsed_state,
                game_state_after_merge=state, injected_history=history,
                stage_latency_ms={
                    "entity_resolution": (entity_end - entity_start) * 1000,
                    "planning": (plan_end - entity_end) * 1000,
                    "retrieval": (retrieval_end - plan_end) * 1000,
                    "context": (context_end - retrieval_end) * 1000,
                    "local_total": (context_end - started) * 1000,
                },
            )
            self.last_trace = trace
            await self.save_game_state(request.session_id, state)
            payload = trace.model_dump(mode="json")
            yield ChatEvent(type="status", message="检索完成", retrieval_plan=payload,
                            context_gaps=[*gaps, *knowledge.knowledge_gaps])
            if understanding.kind == QueryKind.DETERMINISTIC_LOOKUP:
                answer = _deterministic_answer(knowledge, self.settings.current_game_patch)
                await self._record_answer(request, answer)
                yield ChatEvent(type="answer", message=answer.recommendation, response=answer,
                                advice_options=_advice_options(answer), retrieval_plan=payload,
                                context_gaps=gaps, llm_call_count=0)
                return
            client = GroundedDecisionClient(self.settings)
            llm_started = perf_counter()
            try:
                outcome = await client.decide_context(context)
            finally:
                await client.close()
            trace.llm_call_count = 1
            trace.stage_latency_ms["llm"] = (perf_counter() - llm_started) * 1000
            self.last_trace = trace
            await self._record_answer(request, outcome.response)
            yield ChatEvent(type="answer", message=outcome.response.recommendation,
                            response=outcome.response, advice_options=outcome.options,
                            retrieval_plan=trace.model_dump(mode="json"), context_gaps=gaps,
                            llm_call_count=1)
        except asyncio.CancelledError:
            yield ChatEvent(type="cancelled", message="请求已取消")
        except Exception as exc:
            yield ChatEvent(type="error", message=f"请求失败：{exc}")
        finally:
            if vision is not None:
                await vision.close()

    async def _retrieve(
        self, question: str, plan: RetrievalPlan
    ) -> tuple[RetrievedKnowledge, list[RepositoryCall]]:
        evidence: list[EvidenceRecord] = []
        gaps: list[str] = []
        calls: list[RepositoryCall] = []
        requested_comps = ([plan.target_comp] if plan.target_comp else plan.candidate_comps)
        for requested_comp in requested_comps:
            start = perf_counter()
            comp = await self._repo().find_comp_by_id(
                requested_comp.canonical_id, plan.game_patch
            )
            calls.append(RepositoryCall(
                operation="find_comp_by_id",
                filters={"canonical_id": requested_comp.canonical_id,
                         "game_patch": plan.game_patch}, result_count=int(comp is not None),
                latency_ms=(perf_counter() - start) * 1000,
            ))
            if comp:
                evidence.extend(item for item in comp_evidence(comp, plan.game_patch)
                                if item.section in plan.knowledge_needs)
            else:
                gaps.append(
                    f"composition {requested_comp.display_name} is absent for the exact patch"
                )
        if not requested_comps and plan.retrieve_all_comps:
            start = perf_counter()
            comps = await self._repo().search_comps(None, plan.game_patch, 8)
            calls.append(RepositoryCall(
                operation="browse_comps", filters={"game_patch": plan.game_patch, "limit": 8},
                result_count=len(comps), latency_ms=(perf_counter() - start) * 1000,
            ))
            for comp in comps:
                evidence.extend(comp_evidence(comp, plan.game_patch))
        elif not requested_comps and plan.candidate_selection_inputs:
            gaps.append("no structured item/augment-to-composition candidate index is available")
        static_entities = [item for item in plan.entities
                           if item.entity_type in {"augment", "emblem", "item", "champion"}]
        if static_entities:
            start = perf_counter()
            records = await self._repo().find_static_entities_by_ids(
                [item.canonical_id for item in static_entities]
            )
            calls.append(RepositoryCall(
                operation="find_static_entities_by_ids",
                filters={"canonical_ids": [item.canonical_id for item in static_entities],
                         "set_number": plan.set_number}, result_count=len(records),
                latency_ms=(perf_counter() - start) * 1000,
            ))
            by_id = {record.entity_id: record for record in records}
            for resolved in static_entities:
                record = by_id.get(resolved.canonical_id)
                item = static_evidence(record, resolved, plan.game_patch) if record else None
                if item:
                    evidence.append(item)
                else:
                    gaps.append(f"missing usable definition for {resolved.display_name}")
        if plan.retrieve_strategy_text:
            text_targets: list[ResolvedEntity | None] = [*requested_comps] or [None]
            for text_target in text_targets:
                start = perf_counter()
                target_name = text_target.display_name if text_target else None
                chunks = await self._repo().search_knowledge_text(
                    question, plan.game_patch, 6, target_name,
                )
                calls.append(RepositoryCall(
                    operation="search_strategy_text",
                    filters={"requested_patch": plan.game_patch,
                             "set_name": plan.set_name, "target_comp": target_name,
                             "sections": plan.knowledge_needs}, result_count=len(chunks),
                    latency_ms=(perf_counter() - start) * 1000,
                ))
                chunk_evidence = [_chunk_evidence(chunk, plan) for chunk in chunks]
                evidence.extend(item for item in chunk_evidence
                                if item.section in plan.knowledge_needs)
        available = {item.section for item in evidence}
        for need in plan.knowledge_needs:
            if need not in available and need not in {"composition", "leveling"}:
                gaps.append(f"knowledge coverage missing: {need}")
        unique = {item.evidence_id: item for item in evidence}
        return RetrievedKnowledge(evidence=list(unique.values()),
                                  knowledge_gaps=list(dict.fromkeys(gaps))), calls

    async def _record_answer(self, request: ChatRequest, answer: FinalResponse) -> None:
        await self._repo().append_session_messages(request.session_id, [
            ("user", request.text), ("assistant", answer.recommendation),
        ])


def _merge_game_state(previous: GameState, current: GameState) -> GameState:
    update = {field: getattr(current, field) for field in current.model_fields_set
              if field != "user_question"}
    update["user_question"] = current.user_question
    return previous.model_copy(update=update)


def _relevant_history(
    rows: list[tuple[str, str]], understanding: QueryStateUnderstanding
) -> list[AgentMessage]:
    names = [item.display_name for item in (
        [understanding.target_comp] if understanding.target_comp
        else understanding.candidate_comps
    )]
    recent = rows[-8:]
    selected: list[tuple[str, str]] = []
    for index, row in enumerate(recent):
        role, content = row
        if role != "user" or not any(name in content for name in names):
            continue
        selected.append(row)
        if index + 1 < len(recent) and recent[index + 1][0] == "assistant":
            selected.append(recent[index + 1])
    return [AgentMessage(role=role, content=content) for role, content in selected[-4:]]


def _chunk_evidence(chunk: dict[str, object], plan: RetrievalPlan) -> EvidenceRecord:
    source_patch = str(chunk.get("game_patch") or "") or None
    identifier = str(chunk.get("document_id") or chunk.get("title"))
    section = _strategy_chunk_section(
        str(chunk.get("heading") or ""), str(chunk.get("text") or "")
    )
    source = str(chunk.get("source") or "strategy knowledge")
    return EvidenceRecord(
        evidence_id=":".join((source, source_patch or "unversioned", "strategy", identifier,
                              section)),
        source=source, requested_patch=plan.game_patch, source_patch=source_patch,
        patch_match=str(chunk.get("patch_match") or "exact"), entity_type="strategy",
        canonical_entity_id=identifier, display_name=str(chunk.get("title") or "攻略"),
        section=section, semantic_content=str(chunk.get("text") or ""),
        relevance_score=float(str(chunk.get("relevance_score") or 0)),
        url=str(chunk.get("url") or ""),
    )


def _strategy_chunk_section(heading: str, text: str) -> str:
    value = f"{heading} {text}".casefold()
    if any(term in value for term in ("opening", "condition", "开局", "条件")):
        return "starting_conditions"
    if any(term in value for term in ("roll", "搜牌", "d牌")):
        return "reroll_pattern"
    if any(term in value for term in ("item", "装备", "flexibility")):
        return "items"
    if any(term in value for term in ("level", "人口", "transition")):
        return "leveling"
    if any(term in value for term in ("early", "stage 2", "前期")):
        return "early_game"
    if any(term in value for term in ("late", "stage 5", "成型", "后期")):
        return "late_game"
    if any(term in value for term in ("mid", "stage 3", "stage 4", "中期")):
        return "mid_game"
    return "core"


def _deterministic_answer(knowledge: RetrievedKnowledge, patch: str) -> FinalResponse:
    relevant = [item for item in knowledge.evidence if item.section == "core_units"]
    if relevant:
        item = relevant[0]
        values = item.semantic_content if isinstance(item.semantic_content, list) else []
        return FinalResponse(
            recommendation=f"{item.display_name}的核心棋子：{'、'.join(values)}。",
            used_tools=["structured_sqlite_retrieval"],
            sources=[f"{item.display_name} · {item.section} — {item.url}"],
        )
    comps = list(dict.fromkeys(item.display_name for item in knowledge.evidence
                               if item.entity_type == "composition"))
    message = f"当前版本 {patch} 阵容：" + "、".join(comps) if comps else (
        f"本地数据库没有找到 {patch} 的对应事实。"
    )
    return FinalResponse(recommendation=message, used_tools=["structured_sqlite_retrieval"])


def _advice_options(response: FinalResponse) -> list[AdviceOption]:
    evidence = [AdviceEvidence(label=value.rsplit(" — ", 1)[0],
                               url=value.rsplit(" — ", 1)[-1], source="retrieved")
                for value in response.sources if " — http" in value]
    return [AdviceOption(title="结果", action=response.recommendation, conditions=[],
                         expected_benefit="基于当前版本本地数据。",
                         risk="数据覆盖不足时已明确标注。", confidence=1, evidence=evidence)]
