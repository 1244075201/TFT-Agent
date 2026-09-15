"""Unified application service with plan-driven, clean retrieval."""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable
from time import perf_counter
from typing import Any

from tft_agent.agent.parser import GameStateParser
from tft_agent.agent.state import AgentMessage, AgentState
from tft_agent.app.aliases import EntityTextNormalizer
from tft_agent.app.entity_resolution import CanonicalEntityResolver, ResolvedEntity
from tft_agent.app.pipeline import (
    AnswerContext,
    ChampionCompositionMatch,
    ConversationSubject,
    DecisionIntent,
    EvidenceRecord,
    KnowledgeIntent,
    ProductMode,
    QueryKind,
    QueryStateUnderstanding,
    RepositoryCall,
    ResolutionCategory,
    RetrievalPlan,
    RetrievalTrace,
    RetrievedKnowledge,
    StateMutationIntent,
    StrategyTurnState,
    apply_state_mutation,
    build_retrieval_plan,
    clean_template,
    comp_evidence,
    context_gaps,
    explicit_entity_attempt,
    format_item_plans,
    is_global_browse_query,
    is_strategy_open_query,
    is_topic_compatible_query,
    static_evidence,
    structured_composition_list_limit,
    understand_query,
    unresolved_entity_fragment,
)
from tft_agent.app.semantic_retrieval import (
    PlannerOutcome,
    RetrievalCapability,
    RetrievalPlannerClient,
    SemanticRetrievalPlan,
    is_semantic_discovery_query,
)
from tft_agent.app.session_scope import session_scope
from tft_agent.core.config import Settings
from tft_agent.core.patches import PatchService
from tft_agent.core.schemas import (
    AdviceEvidence,
    AdviceOption,
    ChatEvent,
    ChatMode,
    ChatRequest,
    FinalResponse,
    GameState,
)
from tft_agent.desktop.catalog import CanonicalEntityCatalog, CompositionCard
from tft_agent.llm.grounded_decision import (
    GroundedDecisionClient,
    enforce_closed_world_grounding,
)
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.schemas import ActivePatchContext, CompStatsRecord
from tft_agent.vision.deepseek_client import DeepSeekVisionClient
from tft_agent.vision.workflow import observe_and_merge


class AgentService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._cancelled_sessions: set[str] = set()
        self._repository: SQLiteGameDataRepository | None = None
        self._entity_resolver: CanonicalEntityResolver | None = None
        self._active_context: ActivePatchContext | None = None
        self._general_guide_resets: set[str] = set()
        self._strategy_turn_states: dict[str, StrategyTurnState] = {}
        self.last_trace: RetrievalTrace | None = None

    def _repo(self) -> SQLiteGameDataRepository:
        if self._repository is None:
            self._repository = SQLiteGameDataRepository(self.settings.database_path.resolve())
            self._repository.initialize_sync()
        return self._repository

    async def _context(self) -> ActivePatchContext:
        resolved = await PatchService(self._repo(), self.settings).active_context()
        if resolved != self._active_context:
            self._active_context = resolved
            self._entity_resolver = None
        return self._active_context

    @property
    def active_context(self) -> ActivePatchContext | None:
        return self._active_context

    def _resolver(
        self, context: ActivePatchContext | None = None
    ) -> CanonicalEntityResolver:
        if context is None:
            context = self._active_context or ActivePatchContext(
                game_patch=self.settings.current_game_patch,
                patch_family=self.settings.current_game_patch.rstrip("abcdefghijklmnopqrstuvwxyz"),
                set_number=self.settings.current_set_number,
                set_name=self.settings.current_set_name,
                active_snapshot_id="",
            )
        if self._entity_resolver is None or self._entity_resolver.context != context:
            self._entity_resolver = CanonicalEntityResolver(
                self._repo(), context, self.settings.entity_fuzzy_threshold
            )
        return self._entity_resolver

    def cancel(self, session_id: str) -> None:
        self._cancelled_sessions.add(session_id)

    async def data_status(self) -> str:
        if not self.settings.database_path.resolve().is_file():
            return "DataTFT：尚未同步"
        await self._context()
        latest = await self._repo().latest_datatft_snapshot()
        if latest is None:
            return "DataTFT：尚未同步"
        metadata, count = latest
        return (f"版本：{metadata.game_patch}｜DataTFT：已同步 {count} 套阵容｜"
                f"更新时间：{metadata.retrieved_at.astimezone():%Y-%m-%d %H:%M}｜"
                f"状态：{metadata.status.value}")

    async def clear_session(self, session_id: str) -> None:
        await self._repo().clear_session(session_id)
        self._strategy_turn_states.pop(session_id, None)

    async def clear_product_session(self, session_id: str, product_mode: ProductMode) -> None:
        scope = session_scope(session_id, product_mode)
        await self._repo().clear_session(scope.session_id)
        self._strategy_turn_states.pop(scope.session_id, None)
        if product_mode == ProductMode.GENERAL_GUIDE:
            self._general_guide_resets.add(scope.session_id)

    async def clear_general_guide_conversation(self, session_id: str) -> None:
        await self.clear_product_session(session_id, ProductMode.GENERAL_GUIDE)

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

    async def entity_catalog(self) -> CanonicalEntityCatalog:
        """Return a presentation-safe, active-patch catalog for desktop controls."""
        context = await self._context()
        rows = await self._resolver(context).catalog()
        static_ids = [
            str(row["canonical_id"])
            for row in rows
            if row.get("entity_type") != "composition"
        ]
        static = []
        for offset in range(0, len(static_ids), 500):
            static.extend(
                await self._repo().find_static_entities_by_ids(static_ids[offset : offset + 500])
            )
        return CanonicalEntityCatalog.from_repository(rows, static)

    async def composition_cards(self, limit: int = 6) -> list[CompositionCard]:
        """Return current-patch composition summaries without repository internals."""
        context = await self._context()
        records = await self._repo().search_comps(None, context.game_patch, limit=limit)
        return [CompositionCard(
            canonical_id=record.comp_id,
            display_name=record.name,
            tier=record.tier,
            core_units=record.core_units,
            summary=record.level_plan[0] if record.level_plan else None,
        ) for record in records]

    async def ask(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        active_context = await self._context()
        if (
            request.mode == ChatMode.EXPERIMENTAL_LIVE
            and not self.settings.experimental_live_enabled
        ):
            yield ChatEvent(type="error", message="实验实时模式默认关闭。")
            return
        product_mode = ProductMode(request.product_mode)
        scope = session_scope(request.session_id, product_mode)
        previous_strategy = self._strategy_turn_states.get(scope.session_id)
        if previous_strategy and (
            previous_strategy.patch != active_context.game_patch
            or previous_strategy.workspace != product_mode
        ):
            self._strategy_turn_states.pop(scope.session_id, None)
            previous_strategy = None
        potential_strategy_followup = bool(
            previous_strategy and _is_strategy_followup(request.text)
        )
        retrieval_first = (
            is_strategy_open_query(request.text, product_mode)
            or potential_strategy_followup
        )
        conversation_reset = scope.session_id in self._general_guide_resets
        self._cancelled_sessions.discard(scope.session_id)
        started = perf_counter()
        yield ChatEvent(
            type="status",
            message=("正在检索当前版本策略…" if retrieval_first else "正在解析实体并规划检索…"),
        )
        persisted_state = (
            await self.load_game_state(scope.session_id) if scope.uses_game_state else GameState()
        )
        parsed_state = (
            request.confirmed_game_state or GameStateParser().parse_text(request.text)
            if scope.uses_game_state
            else GameState(user_question=request.text)
        )
        state = (
            request.confirmed_game_state or _merge_game_state(persisted_state, parsed_state)
            if scope.uses_game_state
            else parsed_state
        )
        history_rows = await self._repo().load_session_messages(scope.session_id)
        conversational_subject = (
            await self._conversation_subject(history_rows, active_context)
            if product_mode == ProductMode.GENERAL_GUIDE else None
        )
        vision: DeepSeekVisionClient | None = None
        try:
            if request.screenshot_path is not None and scope.accepts_screenshot:
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
            resolution = await self._resolver(active_context).resolve(request.text)
            explicit_entities = list(resolution.entities)
            preliminary_understanding = understand_query(
                request.text, state, product_mode, resolution
            )
            semantic_discovery = (
                not potential_strategy_followup
                and structured_composition_list_limit(request.text) is None
                and preliminary_understanding.knowledge_intent not in {
                    KnowledgeIntent.ENTITY_FACT,
                    KnowledgeIntent.GENERAL_KNOWLEDGE,
                }
                and is_semantic_discovery_query(
                    request.text, {item.entity_type for item in resolution.entities}
                )
            )
            retrieval_first = retrieval_first or semantic_discovery
            strategy_state_inherited = bool(
                potential_strategy_followup
                and previous_strategy
                and not resolution.entities
            )
            entity_attempt = explicit_entity_attempt(request.text)
            subject_inherited = False
            inheritance_reason: str | None = None
            inheritance_blocked_reason: str | None = None
            explicit_subjects = [item for item in resolution.entities
                                 if item.entity_type in {"composition", "champion"}]
            unresolved_attempt = (
                unresolved_entity_fragment(entity_attempt, resolution.entities)
                if entity_attempt else None
            )
            if (
                not strategy_state_inherited
                and
                conversational_subject
                and not resolution.entities
                and unresolved_attempt is None
                and not is_global_browse_query(request.text, product_mode)
                and is_topic_compatible_query(request.text)
            ):
                inherited = ResolvedEntity(
                    canonical_id=conversational_subject.canonical_id,
                    entity_type=conversational_subject.entity_type,
                    display_name=conversational_subject.display_name,
                    game_patch=active_context.game_patch,
                    set_number=active_context.set_number,
                    set_name=active_context.set_name,
                    matched_alias=conversational_subject.display_name,
                    match_kind="conversational_subject",
                    match_score=1.0,
                )
                resolution = resolution.model_copy(
                    update={"entities": [*resolution.entities, inherited]}
                )
                subject_inherited = True
                inheritance_reason = "compatible_followup_without_explicit_subject"
            elif unresolved_attempt:
                inheritance_blocked_reason = "explicit_unresolved_entity_attempt"
            elif explicit_subjects:
                inheritance_blocked_reason = "current_explicit_resolved_entity"
            elif resolution.entities:
                inheritance_blocked_reason = "current_explicit_resolved_entity"
            elif is_global_browse_query(request.text, product_mode):
                inheritance_blocked_reason = "current_global_browse_intent"
            elif conversational_subject:
                inheritance_blocked_reason = "query_not_topic_compatible"
            effective_subject = (
                _conversation_subject_from_entities(explicit_subjects)
                or (conversational_subject if subject_inherited else None)
            )
            resolved_comps = [
                item for item in resolution.entities if item.entity_type == "composition"
            ]
            if (
                scope.uses_game_state
                and state.comp_name
                and not resolved_comps
                and any(item.entity_type != "composition" for item in resolution.entities)
            ):
                state = state.model_copy(
                    update={
                        "comp_name": persisted_state.comp_name,
                        "active_comp_id": persisted_state.active_comp_id,
                    }
                )
            # Routing needs an early read, but conversational inheritance can amend
            # resolution. Build the operative understanding from that final resolution.
            understanding = understand_query(request.text, state, product_mode, resolution)
            champion_matches, champion_calls = await self._champion_composition_matches(
                resolution.entities, active_context
            )
            understanding = _apply_champion_strategy_route(understanding, champion_matches)
            inherited_candidates: list[ResolvedEntity] = []
            if strategy_state_inherited and previous_strategy:
                inherited_candidates = _select_followup_candidates(
                    request.text, previous_strategy
                )
                if _is_candidate_comparison(request.text):
                    understanding = understanding.model_copy(update={
                        "target_comp": None,
                        "candidate_comps": inherited_candidates,
                        "decision_intent": DecisionIntent.COMPOSITION_COMPARISON,
                        "state_mutation": StateMutationIntent.CANDIDATES,
                        "resolved_entities": previous_strategy.retrieval_seeds,
                    })
                elif len(inherited_candidates) == 1:
                    understanding = understanding.model_copy(update={
                        "target_comp": inherited_candidates[0],
                        "candidate_comps": [],
                        "decision_intent": DecisionIntent.FULL_GUIDE,
                        "state_mutation": StateMutationIntent.RETAIN,
                        "resolved_entities": previous_strategy.retrieval_seeds,
                    })
                else:
                    understanding = understanding.model_copy(update={
                        "target_comp": None,
                        "candidate_comps": inherited_candidates,
                        "decision_intent": DecisionIntent.FULL_GUIDE,
                        "state_mutation": StateMutationIntent.CANDIDATES,
                        "resolved_entities": previous_strategy.retrieval_seeds,
                    })
            if retrieval_first:
                understanding = understanding.model_copy(update={
                    "kind": QueryKind.STRATEGY_DECISION,
                    "unresolved_terms": [],
                    "resolution_category": ResolutionCategory.RESOLVED_STRATEGY,
                })
            if (
                understanding.state_mutation == StateMutationIntent.RETAIN
                and state.comp_name
            ):
                retained = await self._resolver(active_context).resolve(state.comp_name)
                retained_comp = next((item for item in retained.entities
                                      if item.entity_type == "composition"), None)
                if retained_comp:
                    understanding = understanding.model_copy(
                        update={"target_comp": retained_comp}
                    )
            state = apply_state_mutation(state, understanding, request.text)
            entity_end = perf_counter()
            planner_outcome: PlannerOutcome | None = None
            if semantic_discovery:
                planner = RetrievalPlannerClient(self.settings)
                try:
                    planner_outcome = await planner.plan(
                        request.text, state if scope.uses_game_state else None
                    )
                finally:
                    await planner.close()
            plan = build_retrieval_plan(
                understanding, state, active_context.game_patch,
                active_context.set_number, active_context.set_name,
                active_context.active_snapshot_id,
            )
            plan_end = perf_counter()
            retrieval_queries: list[str] = []
            structured_seeds: list[dict[str, str]] = []
            structured_metadata: dict[str, Any] = {}
            if strategy_state_inherited and previous_strategy:
                retrieval_queries = [
                    request.text,
                    previous_strategy.raw_query,
                    *(item.display_name for item in inherited_candidates),
                ]
                knowledge, calls = await self._retrieve(request.text, plan)
                structured_metadata = {
                    "candidate_comps": [
                        _candidate_trace_record(item) for item in inherited_candidates
                    ]
                }
            elif semantic_discovery and planner_outcome is not None:
                retrieval_queries = [
                    *planner_outcome.plan.strategy_searches,
                    *planner_outcome.plan.strategy_exclusions,
                    *planner_outcome.plan.item_searches,
                ]
                knowledge, calls, structured_metadata = await self._retrieve_semantic(
                    planner_outcome.plan, plan
                )
            elif retrieval_first:
                seed_entities = list(resolution.entities)
                if scope.uses_game_state:
                    state_seed_text = _state_relation_seed_text(state)
                    if state_seed_text:
                        state_resolution = await self._resolver(active_context).resolve(
                            state_seed_text
                        )
                        seed_entities.extend(state_resolution.entities)
                structured_seeds = _structured_seed_records(seed_entities)
                retrieval_queries = _broad_retrieval_queries(
                    request.text, state if scope.uses_game_state else None, seed_entities
                )
                knowledge, calls, structured_metadata = await self._retrieve_broad_strategy(
                    retrieval_queries, structured_seeds, plan
                )
            else:
                knowledge, calls = await self._retrieve(request.text, plan)
            calls = [*champion_calls, *calls]
            retrieval_end = perf_counter()
            history = [] if retrieval_first else _relevant_history(history_rows, understanding)
            gaps = context_gaps(state) if scope.uses_game_state else []
            context = AnswerContext(
                question=request.text, understanding=understanding, game_state=state,
                history=history, retrieval_plan=plan, evidence=knowledge.evidence,
                context_gaps=gaps, knowledge_gaps=knowledge.knowledge_gaps,
                retrieval_first=retrieval_first,
                previous_topic=(previous_strategy.raw_query
                                if strategy_state_inherited and previous_strategy else None),
                current_candidates=inherited_candidates,
                retrieval_interpretation=(
                    planner_outcome.plan.user_goal if planner_outcome else None
                ),
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
                explicit_entities=explicit_entities,
                conversational_subject=effective_subject,
                subject_inherited=subject_inherited,
                inheritance_reason=inheritance_reason,
                champion_candidates=champion_matches,
                final_resolution_category=understanding.resolution_category,
                explicit_entity_attempt=entity_attempt,
                explicit_unresolved_term=unresolved_attempt,
                inheritance_blocked_reason=inheritance_blocked_reason,
                conversation_reset=conversation_reset,
                retrieval_first_strategy=retrieval_first,
                workspace=product_mode,
                game_state_present=(scope.uses_game_state and _has_game_state(state)),
                retrieval_queries=retrieval_queries,
                retrieved_evidence_kinds=list(dict.fromkeys(
                    item.section for item in knowledge.evidence
                )),
                text_retrieval_hits=int(structured_metadata.get("text_hits", 0)),
                structured_seeds=structured_seeds,
                structured_expansions=list(structured_metadata.get("expansions", [])),
                structured_candidate_comps=list(
                    structured_metadata.get("candidate_comps", [])
                ),
                evidence_counts_by_type=_evidence_counts(knowledge.evidence),
                strategy_state_inherited=strategy_state_inherited,
                inherited_topic=(previous_strategy.raw_query
                                 if strategy_state_inherited and previous_strategy else None),
                inherited_candidate_ids=[
                    item.canonical_id for item in inherited_candidates
                ],
                route=("semantic_discovery" if semantic_discovery else
                       "strategy_followup" if strategy_state_inherited else
                       "explicit_strategy" if retrieval_first else
                       "deterministic_fact"
                       if understanding.kind == QueryKind.DETERMINISTIC_LOOKUP else
                       "legacy_open_strategy"),
                planner_invoked=planner_outcome is not None,
                planner_model=(planner_outcome.model if planner_outcome else None),
                planner_used_fallback=(
                    planner_outcome.used_fallback if planner_outcome else False
                ),
                planner_validation_error=(
                    planner_outcome.validation_error if planner_outcome else None
                ),
                validated_semantic_plan=(
                    planner_outcome.plan.model_dump(mode="json") if planner_outcome else None
                ),
                planner_generated_search_concepts=(
                    [*planner_outcome.plan.strategy_searches,
                     *planner_outcome.plan.strategy_exclusions,
                     *planner_outcome.plan.item_searches] if planner_outcome else []
                ),
                semantic_structured_filters=(
                    planner_outcome.plan.composition_filters.model_dump(mode="json")
                    if planner_outcome else {}
                ),
                retrieval_actions=(
                    [item.value for item in planner_outcome.plan.capabilities]
                    if planner_outcome else [call.operation for call in calls]
                ),
                candidate_compositions=[
                    str(item.get("name")) for item in
                    structured_metadata.get("candidate_comps", [])
                    if isinstance(item, dict) and item.get("name")
                ],
                stage_latency_ms={
                    "entity_resolution": (entity_end - entity_start) * 1000,
                    "planning": (plan_end - entity_end) * 1000,
                    "retrieval": (retrieval_end - plan_end) * 1000,
                    "context": (context_end - retrieval_end) * 1000,
                    "local_total": (context_end - started) * 1000,
                    **({"planner": planner_outcome.latency_ms}
                       if planner_outcome else {}),
                },
            )
            self.last_trace = trace
            self._general_guide_resets.discard(scope.session_id)
            if scope.uses_game_state:
                await self.save_game_state(scope.session_id, state)
            payload = trace.model_dump(mode="json")
            yield ChatEvent(type="status", message="检索完成", retrieval_plan=payload,
                            context_gaps=[*gaps, *knowledge.knowledge_gaps])
            if not retrieval_first and (
                understanding.kind == QueryKind.DETERMINISTIC_LOOKUP
                or understanding.resolution_category
                == ResolutionCategory.RESOLVED_ENTITY_NO_KNOWLEDGE
                or any(gap.startswith("recognized entities lack exact relation:")
                       for gap in knowledge.knowledge_gaps)
            ):
                answer = _deterministic_answer(
                    knowledge, plan.game_patch, understanding, plan
                )
                await self._record_answer(request, answer, scope.session_id)
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
            catalog = await self._repo().grounding_entity_catalog()
            grounding = enforce_closed_world_grounding(outcome, context, catalog)
            outcome = grounding.outcome
            trace.llm_call_count = 1
            if planner_outcome:
                trace.llm_call_count += 1
            trace.final_llm_calls = 1
            trace.grounding_passed = not grounding.violations
            trace.grounding_violations = grounding.violations
            trace.grounding_fallback_used = grounding.used_fallback
            trace.stage_latency_ms["llm"] = (perf_counter() - llm_started) * 1000
            trace.stage_latency_ms["answer"] = trace.stage_latency_ms["llm"]
            trace.stage_latency_ms["total"] = (perf_counter() - started) * 1000
            self.last_trace = trace
            await self._record_answer(request, outcome.response, scope.session_id)
            if retrieval_first and not strategy_state_inherited:
                turn_state = _strategy_turn_state_from_trace(trace, active_context)
                if turn_state is not None:
                    self._strategy_turn_states[scope.session_id] = turn_state
            yield ChatEvent(type="answer", message=outcome.response.recommendation,
                            response=outcome.response, advice_options=outcome.options,
                            retrieval_plan=trace.model_dump(mode="json"), context_gaps=gaps,
                            llm_call_count=trace.llm_call_count)
        except asyncio.CancelledError:
            yield ChatEvent(type="cancelled", message="请求已取消")
        except Exception as exc:
            yield ChatEvent(type="error", message=f"请求失败：{exc}")
        finally:
            if vision is not None:
                await vision.close()

    async def _retrieve_broad_strategy(
        self,
        queries: list[str],
        seeds: list[dict[str, str]],
        plan: RetrievalPlan,
    ) -> tuple[RetrievedKnowledge, list[RepositoryCall], dict[str, Any]]:
        started = perf_counter()
        text_rows, structured = await asyncio.gather(
            self._repo().search_normalized_knowledge(
                plan.active_snapshot_id, plan.game_patch, queries, limit=12
            ),
            self._repo().expand_normalized_relations(
                plan.active_snapshot_id, plan.game_patch, seeds, candidate_limit=4
            ),
        )
        raw_structured_rows = structured.get("evidence", [])
        structured_rows = (
            list(raw_structured_rows) if isinstance(raw_structured_rows, list) else []
        )
        raw_candidates = structured.get("candidate_comps", [])
        candidate_rows = raw_candidates if isinstance(raw_candidates, list) else []
        candidate_ids = {
            str(item.get("comp_id")) for item in candidate_rows if isinstance(item, dict)
        }
        candidate_ids.update(
            item.canonical_id for item in (
                [plan.target_comp] if plan.target_comp else plan.candidate_comps
            ) if item is not None
        )
        seed_names = [_normalized(seed["display_name"]) for seed in seeds]
        seed_names.extend(
            _normalized(seed["display_name"].removesuffix("纹章"))
            for seed in seeds
            if seed["entity_type"] == "emblem" and seed["display_name"].endswith("纹章")
        )
        if candidate_ids:
            text_rows = [row for row in text_rows if (
                str(row.get("comp_id")) in candidate_ids
                or any(name and name in _normalized(str(row.get("display_name") or ""))
                       for name in seed_names)
            )]
        rows = [*structured_rows, *text_rows]
        unique_rows = {str(row["evidence_id"]): row for row in rows}
        call = RepositoryCall(
            operation="hybrid_normalized_retrieval",
            filters={
                "queries": queries,
                "structured_seeds": seeds,
                "game_patch": plan.game_patch,
                "snapshot_id": plan.active_snapshot_id,
                "text_limit": 12,
                "candidate_limit": 4,
            },
            result_count=len(unique_rows),
            latency_ms=(perf_counter() - started) * 1000,
        )
        evidence = [EvidenceRecord(
            requested_patch=plan.game_patch,
            **row,
        ) for row in unique_rows.values()]
        gaps = [] if evidence else [
            "no relevant evidence found in active normalized knowledge"
        ]
        if not evidence and not seeds:
            gaps.append(
                "no unique structured entity seed was resolved; request the full entity name"
            )
        if evidence and all(item.section == "definition" for item in evidence):
            gaps.append(
                "retrieval found entity definitions but no composition strategy relation"
            )
        metadata = {
            "text_hits": len(text_rows),
            "structured_hits": len(structured_rows),
            "expansions": structured.get("expansions", []),
            "candidate_comps": candidate_rows,
        }
        return RetrievedKnowledge(evidence=evidence, knowledge_gaps=gaps), [call], metadata

    async def _retrieve_semantic(
        self, semantic: SemanticRetrievalPlan, plan: RetrievalPlan,
    ) -> tuple[RetrievedKnowledge, list[RepositoryCall], dict[str, Any]]:
        """Execute one validated discovery plan against the active snapshot."""
        started = perf_counter()
        filters = semantic.composition_filters
        capabilities = set(semantic.capabilities)
        use_cost = RetrievalCapability.FILTER_BY_MAIN_UNIT_COST in capabilities
        use_strategy = RetrievalCapability.SEARCH_STRATEGY_TEXT in capabilities
        use_item_search = RetrievalCapability.FIND_BY_ITEM_OR_ARTIFACT in capabilities
        requested_fields = [field for field in semantic.requested_fields if (
            field == "tier" or field == "main_units"
            or field == "roster" and RetrievalCapability.GET_COMPOSITION_ROSTER in capabilities
            or field == "items" and RetrievalCapability.GET_COMPOSITION_ITEMS in capabilities
            or field in {"starting_conditions", "core", "early_game", "mid_game",
                         "late_game", "tips"}
            and bool(capabilities & {
                RetrievalCapability.GET_COMPOSITION_STRATEGY,
                RetrievalCapability.SEARCH_STRATEGY_TEXT,
            })
        )]
        result = await self._repo().discover_compositions(
            plan.active_snapshot_id,
            plan.game_patch,
            main_unit_cost_min=filters.main_unit_cost_min if use_cost else None,
            main_unit_cost_max=filters.main_unit_cost_max if use_cost else None,
            cost_quantifier=filters.cost_quantifier.value,
            strategy_terms=semantic.strategy_searches if use_strategy else [],
            strategy_exclusions=semantic.strategy_exclusions if use_strategy else [],
            item_terms=semantic.item_searches if use_item_search else [],
            require_strategy_match=filters.require_strategy_match and use_strategy,
            requested_fields=requested_fields,
            limit=semantic.limit,
        )
        raw_evidence = result.get("evidence", [])
        rows = raw_evidence if isinstance(raw_evidence, list) else []
        raw_candidates = result.get("candidate_comps", [])
        candidates = raw_candidates if isinstance(raw_candidates, list) else []
        evidence = [EvidenceRecord(requested_patch=plan.game_patch, **row)
                    for row in rows if isinstance(row, dict)]
        gaps: list[str] = []
        if not evidence:
            gaps.append("validated semantic plan found no qualifying active-snapshot evidence")
        if any(
            field in semantic.requested_fields
            for field in ("early_game", "starting_conditions")
        ):
            gaps.append(
                "no stage win-rate evidence is available; early strength can only be inferred "
                "from retrieved guide descriptions"
            )
        call = RepositoryCall(
            operation="execute_semantic_retrieval_plan",
            filters={
                "game_patch": plan.game_patch,
                "snapshot_id": plan.active_snapshot_id,
                "capabilities": [item.value for item in semantic.capabilities],
                "composition_filters": filters.model_dump(mode="json"),
                "strategy_searches": semantic.strategy_searches,
                "strategy_exclusions": semantic.strategy_exclusions,
                "item_searches": semantic.item_searches,
                "requested_fields": semantic.requested_fields,
                "limit": semantic.limit,
            },
            result_count=len(evidence), latency_ms=(perf_counter() - started) * 1000,
        )
        return (
            RetrievedKnowledge(evidence=evidence, knowledge_gaps=gaps),
            [call],
            {"candidate_comps": candidates, "text_hits": sum(
                item.section not in {"candidate_composition", "composition_roster", "item_plan"}
                for item in evidence
            )},
        )

    async def _conversation_subject(
        self, rows: list[tuple[str, str]], context: ActivePatchContext
    ) -> ConversationSubject | None:
        user_turns = [content for role, content in rows[-8:] if role == "user"]
        for offset, content in enumerate(reversed(user_turns), start=1):
            if is_global_browse_query(content):
                return None
            resolution = await self._resolver(context).resolve(content)
            subjects = [item for item in resolution.entities
                        if item.entity_type in {"composition", "champion"}]
            compositions = [item for item in subjects if item.entity_type == "composition"]
            selected = compositions[0] if len(compositions) == 1 else (
                subjects[0] if len(subjects) == 1 else None
            )
            if selected:
                return ConversationSubject(
                    canonical_id=selected.canonical_id,
                    entity_type=selected.entity_type,
                    display_name=selected.display_name,
                    source_turn_offset=offset,
                )
        return None

    async def _champion_composition_matches(
        self, entities: list[ResolvedEntity], context: ActivePatchContext
    ) -> tuple[list[ChampionCompositionMatch], list[RepositoryCall]]:
        champions = [item for item in entities if item.entity_type == "champion"]
        if not champions:
            return [], []
        started = perf_counter()
        comps = await self._repo().search_comps(
            None, context.game_patch, limit=50
        )
        catalog = await self._resolver(context).catalog()
        catalog_terms: dict[str, set[str]] = {}
        for row in catalog:
            if row["entity_type"] != "champion":
                continue
            aliases = row["aliases"]
            alias_values = aliases if isinstance(aliases, list) else []
            catalog_terms[str(row["canonical_id"])] = {
                _normalized(str(row["display_name"])),
                *(_normalized(str(alias)) for alias in alias_values),
            }
        matches: list[ChampionCompositionMatch] = []
        for champion in champions:
            identity_terms = catalog_terms.get(champion.canonical_id, {
                _normalized(champion.display_name), _normalized(champion.matched_alias),
            })
            for comp in comps:
                sources: list[str] = []
                if context.active_snapshot_id:
                    board = await self._repo().get_comp_board_units(
                        context.active_snapshot_id, comp.comp_id, "FINAL"
                    )
                    items = await self._repo().get_comp_items(
                        context.active_snapshot_id, comp.comp_id
                    )
                    if any(_champion_row_matches(row, champion) for row in board):
                        sources.append("normalized_final_board")
                    if any(_holder_matches(row, champion) for row in items):
                        sources.append("normalized_item_holder")
                if not sources:
                    if champion.display_name in comp.core_units:
                        sources.append("legacy_core_units")
                    if any(
                        _recommended_item_owner(value) == champion.display_name
                        for value in comp.recommended_items
                    ):
                        sources.append("legacy_recommended_item_owner")
                if _normalized(comp.name) in identity_terms:
                    sources.append("exact_composition_title_identity")
                if not sources:
                    continue
                matches.append(ChampionCompositionMatch(
                    champion_id=champion.canonical_id,
                    champion_name=champion.display_name,
                    composition=ResolvedEntity(
                        canonical_id=comp.comp_id,
                        entity_type="composition",
                        display_name=comp.name,
                        game_patch=context.game_patch,
                        set_number=context.set_number,
                        set_name=context.set_name,
                        matched_alias=comp.name,
                        match_kind="structured_champion_bridge",
                        match_score=1.0,
                    ),
                    confidence="high",
                    sources=sources,
                ))
        call = RepositoryCall(
            operation="find_compositions_by_champion_signals",
            filters={
                "champion_ids": [item.canonical_id for item in champions],
                "game_patch": context.game_patch,
                "snapshot_id": context.active_snapshot_id,
                "signals": ["normalized_final_board", "normalized_item_holder",
                            "exact_composition_title_identity"],
            },
            result_count=len(matches),
            latency_ms=(perf_counter() - started) * 1000,
        )
        return matches, [call]

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
                normalized, normalized_calls = await self._normalized_comp_evidence(
                    comp, plan
                )
                evidence.extend(normalized)
                calls.extend(normalized_calls)
                if not normalized:
                    legacy_needs = set(plan.knowledge_needs)
                    if "composition_roster" in legacy_needs:
                        legacy_needs.add("core_units")
                    evidence.extend(item for item in comp_evidence(comp, plan.game_patch)
                                    if item.section in legacy_needs)
            else:
                gaps.append(
                    f"composition {requested_comp.display_name} is absent for the exact patch"
                )
        if not requested_comps and plan.retrieve_all_comps:
            start = perf_counter()
            comps = await self._repo().search_comps(
                None, plan.game_patch, plan.composition_limit
            )
            if plan.tier_filter:
                comps = [comp for comp in comps if comp.tier in plan.tier_filter]
            comps = comps[:plan.composition_limit]
            calls.append(RepositoryCall(
                operation="browse_comps",
                filters={"game_patch": plan.game_patch, "limit": plan.composition_limit},
                result_count=len(comps), latency_ms=(perf_counter() - start) * 1000,
            ))
            for comp in comps:
                evidence.append(_composition_summary_evidence(comp, plan))
        elif not requested_comps and plan.candidate_selection_inputs:
            gaps.append("no structured item/augment-to-composition candidate index is available")
        static_entities = [item for item in plan.entities
                           if item.entity_type in {"augment", "emblem", "item"}]
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
        normalized_strategy = any(
            item.source == "DataTFT normalized" and item.section in {
                "starting_conditions", "core", "early_game", "mid_game", "late_game", "tip",
            }
            for item in evidence
        )
        if plan.retrieve_strategy_text and not normalized_strategy:
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
        gaps.extend(_combination_relation_gaps(plan, evidence))
        available = {item.section for item in evidence}
        comparison_subject = (
            set(plan.knowledge_needs) & {"augment", "artifact", "radiant"}
            or any(item.entity_type == "augment" for item in plan.entities)
        )
        if plan.comparison_requested and comparison_subject:
            gaps.append(
                "comparison evidence unavailable: associations do not provide rank, priority, "
                "score, win rate, or stage preference"
            )
        for need in plan.knowledge_needs:
            if need not in available and need not in {"composition", "leveling"}:
                gaps.append(f"knowledge coverage missing: {need}")
        unique = {item.evidence_id: item for item in evidence}
        return RetrievedKnowledge(evidence=list(unique.values()),
                                  knowledge_gaps=list(dict.fromkeys(gaps))), calls

    async def _normalized_comp_evidence(
        self, comp: Any, plan: RetrievalPlan
    ) -> tuple[list[EvidenceRecord], list[RepositoryCall]]:
        if not plan.active_snapshot_id:
            return [], []
        comp_id = str(comp.comp_id)
        comp_name = str(comp.name)
        source_url = str(comp.snapshot.source_url)
        evidence: list[EvidenceRecord] = []
        calls: list[RepositoryCall] = []

        async def read(
            operation: str, function: Awaitable[list[dict[str, Any]]]
        ) -> list[dict[str, Any]]:
            started = perf_counter()
            rows = await function
            calls.append(RepositoryCall(
                operation=operation,
                filters={"snapshot_id": plan.active_snapshot_id, "comp_id": comp_id},
                result_count=len(rows), latency_ms=(perf_counter() - started) * 1000,
            ))
            return rows

        needs = set(plan.knowledge_needs)
        if needs & {"composition_roster", "positioning"}:
            board = await read(
                "get_comp_board_units",
                self._repo().get_comp_board_units(plan.active_snapshot_id, comp_id, "FINAL"),
            )
            selected = board
            if plan.target_champion and "positioning" in needs:
                selected = [
                    row for row in board
                    if _champion_row_matches(row, plan.target_champion)
                ]
            section = "positioning" if "positioning" in needs else "composition_roster"
            if selected:
                evidence.append(_normalized_evidence(
                    plan, comp_id, comp_name, section, selected, source_url,
                    source_response_id=_shared_source_response(selected),
                ))
        category = next((value for value in ("ARTIFACT", "RADIANT", "EMBLEM")
                         if value.casefold() in needs), None)
        if needs & {"item_plan", "artifact", "radiant", "emblem"}:
            operation = "get_comp_items_by_category" if category else "get_comp_items"
            query = (
                self._repo().get_comp_items_by_category(
                    plan.active_snapshot_id, comp_id, category
                ) if category else
                self._repo().get_comp_items(plan.active_snapshot_id, comp_id)
            )
            items = await read(operation, query)
            if plan.target_champion:
                items = [row for row in items if _holder_matches(row, plan.target_champion)]
            section = category.casefold() if category else "item_plan"
            if items:
                evidence.append(_normalized_evidence(
                    plan, comp_id, comp_name, section, items, source_url,
                    source_response_id=_shared_source_response(items),
                ))
        if "augment" in needs:
            augments = await read(
                "get_comp_augments",
                self._repo().get_comp_augments(plan.active_snapshot_id, comp_id),
            )
            if augments:
                evidence.append(_normalized_evidence(
                    plan, comp_id, comp_name, "augment", augments, source_url,
                    source_response_id=_shared_source_response(augments),
                ))
        if "condition" in needs:
            conditions = await read(
                "get_comp_conditions",
                self._repo().get_comp_conditions(plan.active_snapshot_id, comp_id),
            )
            if conditions:
                evidence.append(_normalized_evidence(
                    plan, comp_id, comp_name, "condition", conditions, source_url,
                    source_response_id=_shared_source_response(conditions),
                ))
        strategy_needs = needs & {
            "starting_conditions", "core", "early_game", "mid_game", "late_game", "tip",
        }
        if strategy_needs:
            sections = await read(
                "get_comp_strategy_sections",
                self._repo().get_comp_strategy_sections(plan.active_snapshot_id, comp_id),
            )
            for row in sections:
                section = str(row.get("section_type") or "")
                if section not in strategy_needs and section != "tip":
                    continue
                evidence.append(_normalized_evidence(
                    plan, comp_id, comp_name, section,
                    str(row.get("text") or ""), source_url,
                    source_response_id=str(row.get("source_response_id") or "") or None,
                    suffix=str(row.get("ordinal") or 0),
                ))
        return evidence, calls

    async def _record_answer(
        self, request: ChatRequest, answer: FinalResponse, session_id: str | None = None
    ) -> None:
        await self._repo().append_session_messages(session_id or request.session_id, [
            ("user", request.text), ("assistant", answer.recommendation),
        ])


def _merge_game_state(previous: GameState, current: GameState) -> GameState:
    update = {field: getattr(current, field) for field in current.model_fields_set
              if field != "user_question"}
    update["user_question"] = current.user_question
    return previous.model_copy(update=update)


def _has_game_state(state: GameState) -> bool:
    return bool(state.model_dump(
        exclude={"user_question"}, exclude_none=True, exclude_defaults=True
    ))


def _state_relation_seed_text(state: GameState) -> str:
    values = [
        *state.items, *state.item_components, *state.completed_items,
        *state.artifacts, *state.radiant_items, *state.emblems,
        *state.augments, *state.augment_choices,
    ]
    return " ".join(dict.fromkeys(value for value in values if value.strip()))


def _structured_seed_records(
    entities: list[ResolvedEntity],
) -> list[dict[str, str]]:
    seeds: dict[tuple[str, str], dict[str, str]] = {}
    for entity in entities:
        if entity.entity_type not in {"trait", "emblem", "item", "artifact", "augment"}:
            continue
        key = (entity.entity_type, entity.canonical_id)
        seeds[key] = {
            "entity_type": entity.entity_type,
            "canonical_id": entity.canonical_id,
            "display_name": entity.display_name,
            "matched_alias": entity.matched_alias,
        }
    return list(seeds.values())


def _evidence_counts(evidence: list[EvidenceRecord]) -> dict[str, int]:
    categories = {
        "composition": "DIRECT FACT",
        "definition": "STATIC DEFINITION",
        "structured_relation": "STRUCTURED RELATION",
        "candidate_composition": "CANDIDATE COMPOSITION",
    }
    output: dict[str, int] = {}
    for item in evidence:
        category = categories.get(item.section, "STRATEGY TEXT")
        output[category] = output.get(category, 0) + 1
    return output


def _is_strategy_followup(text: str) -> bool:
    normalized = _normalized(text)
    return bool(re.search(
        r"(?:其他|另外|剩下|还有|这几个|这些|它们|几个|那.+呢|"
        r"第(?:一|二|三|四|五|\d+)(?:个|套)?|"
        r"哪个.*(?:前期|后期|适合)|有什么区别|区别)",
        normalized,
    ))


def _is_candidate_comparison(text: str) -> bool:
    return bool(re.search(r"这几个|这些|它们|区别|比较|哪个", text))


def _select_followup_candidates(
    text: str,
    previous: StrategyTurnState,
) -> list[ResolvedEntity]:
    candidates = previous.candidate_compositions
    ordinal = re.search(r"第(一|二|三|四|五|\d+)(?:个|套)?", text)
    if ordinal:
        value = ordinal.group(1)
        index = ({"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}.get(value)
                 or int(value)) - 1
        return [candidates[index]] if 0 <= index < len(candidates) else []
    if re.search(r"其他|另外|剩下|还有", text):
        primary_id = (
            previous.primary_composition.canonical_id
            if previous.primary_composition else None
        )
        return [item for item in candidates if item.canonical_id != primary_id]
    return list(candidates)


def _candidate_trace_record(entity: ResolvedEntity) -> dict[str, object]:
    return {
        "comp_id": entity.canonical_id,
        "name": entity.display_name,
        "source": "previous_strategy_turn",
    }


def _strategy_turn_state_from_trace(
    trace: RetrievalTrace,
    context: ActivePatchContext,
) -> StrategyTurnState | None:
    candidates: dict[str, ResolvedEntity] = {}
    ordered = ([trace.understanding.target_comp]
               if trace.understanding.target_comp else trace.understanding.candidate_comps)
    for entity in ordered:
        if entity is not None:
            candidates[entity.canonical_id] = entity
    for row in trace.structured_candidate_comps:
        comp_id = str(row.get("comp_id") or "")
        name = str(row.get("name") or "")
        if not comp_id or not name or comp_id in candidates:
            continue
        candidates[comp_id] = ResolvedEntity(
            canonical_id=comp_id,
            entity_type="composition",
            display_name=name,
            game_patch=context.game_patch,
            set_number=context.set_number,
            set_name=context.set_name,
            matched_alias=name,
            match_kind="retrieval_candidate",
            match_score=1.0,
        )
    if not candidates:
        return None
    values = list(candidates.values())
    seeds = [item for item in trace.understanding.resolved_entities
             if item.entity_type != "composition"]
    return StrategyTurnState(
        workspace=trace.workspace or trace.understanding.product_mode,
        patch=context.game_patch,
        raw_query=trace.original_query,
        primary_composition=trace.understanding.target_comp or values[0],
        candidate_compositions=values,
        retrieval_seeds=seeds,
    )


def _broad_retrieval_queries(
    raw_query: str,
    state: GameState | None,
    entities: list[ResolvedEntity],
) -> list[str]:
    """Preserve the raw query and add only existing deterministic recall hints."""
    queries = [raw_query]
    for entity in entities:
        queries.extend((entity.matched_alias, entity.display_name))
        if entity.entity_type == "emblem" and entity.display_name.endswith("纹章"):
            queries.append(entity.display_name.removesuffix("纹章"))
    if state is not None:
        queries.extend(value for value in (
            state.comp_name, state.target_champion, *state.core_units,
            *state.items, *state.item_components, *state.completed_items,
            *state.artifacts, *state.radiant_items, *state.emblems,
            *state.augments, *state.augment_choices,
        ) if value)
        queries.extend(unit.name for unit in [*state.board, *state.bench] if unit.name)
        queries.extend(trait.name for trait in state.traits)
    return list(dict.fromkeys(value.strip() for value in queries if value.strip()))


def _apply_champion_strategy_route(
    understanding: QueryStateUnderstanding,
    matches: list[ChampionCompositionMatch],
) -> QueryStateUnderstanding:
    champions = [item for item in understanding.resolved_entities
                 if item.entity_type == "champion"]
    if not champions:
        return understanding
    if understanding.target_comp:
        return understanding.model_copy(
            update={"resolution_category": ResolutionCategory.RESOLVED_STRATEGY}
        )
    champion_ids = {item.canonical_id for item in champions}
    unique: dict[str, ResolvedEntity] = {}
    for match in matches:
        if match.champion_id in champion_ids and match.confidence == "high":
            unique[match.composition.canonical_id] = match.composition
    candidates = list(unique.values())
    if len(candidates) == 1:
        return understanding.model_copy(update={
            "target_comp": candidates[0],
            "candidate_comps": [],
            "state_mutation": StateMutationIntent.REPLACE,
            "resolution_category": ResolutionCategory.RESOLVED_STRATEGY,
        })
    if len(candidates) > 1:
        return understanding.model_copy(update={
            "target_comp": None,
            "candidate_comps": candidates,
            "state_mutation": StateMutationIntent.CANDIDATES,
            "decision_intent": DecisionIntent.COMPOSITION_COMPARISON,
            "resolution_category": ResolutionCategory.RESOLVED_STRATEGY,
        })
    return understanding.model_copy(update={
        "resolution_category": ResolutionCategory.RESOLVED_ENTITY_NO_KNOWLEDGE,
    })


def _normalized(value: str) -> str:
    return EntityTextNormalizer.normalize(value)


def _conversation_subject_from_entities(
    entities: list[ResolvedEntity],
) -> ConversationSubject | None:
    compositions = [item for item in entities if item.entity_type == "composition"]
    selected = compositions[0] if len(compositions) == 1 else (
        entities[0] if len(entities) == 1 else None
    )
    if selected is None:
        return None
    return ConversationSubject(
        canonical_id=selected.canonical_id,
        entity_type=selected.entity_type,
        display_name=selected.display_name,
        source_turn_offset=0,
    )


def _recommended_item_owner(value: str) -> str:
    owner, separator, _ = value.partition("方案")
    if separator:
        return owner.strip()
    owner, separator, _ = value.partition(":")
    return owner.strip() if separator else ""


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


def _champion_row_matches(row: dict[str, Any], champion: ResolvedEntity) -> bool:
    return (
        row.get("canonical_champion_id") == champion.canonical_id
        or _normalized(str(row.get("display_name") or ""))
        == _normalized(champion.display_name)
    )


def _holder_matches(row: dict[str, Any], champion: ResolvedEntity) -> bool:
    return (
        row.get("holder_canonical_champion_id") == champion.canonical_id
        or _normalized(str(row.get("holder_display_name") or ""))
        == _normalized(champion.display_name)
    )


def _shared_source_response(rows: list[dict[str, Any]]) -> str | None:
    values = {str(row.get("source_response_id")) for row in rows
              if row.get("source_response_id")}
    return next(iter(values)) if len(values) == 1 else None


def _combination_relation_gaps(
    plan: RetrievalPlan, evidence: list[EvidenceRecord]
) -> list[str]:
    if not (plan.target_comp or plan.candidate_comps):
        return []
    section_by_type = {
        "trait": "key_traits", "emblem": "emblem", "item": "item_plan",
        "augment": "augment",
    }
    gaps: list[str] = []
    for entity in plan.entities:
        section = section_by_type.get(entity.entity_type)
        if section is None:
            continue
        related = any(
            item.entity_type == "composition"
            and item.section == section
            and _normalized(entity.display_name) in _normalized(str(item.semantic_content))
            for item in evidence
        )
        if not related:
            gaps.append(
                "recognized entities lack exact relation: "
                f"{entity.display_name} has no current-patch composition association"
            )
    return gaps


def _normalized_evidence(
    plan: RetrievalPlan,
    comp_id: str,
    comp_name: str,
    section: str,
    content: str | list[dict[str, Any]],
    source_url: str,
    *,
    source_response_id: str | None,
    suffix: str = "0",
) -> EvidenceRecord:
    semantic: str | list[str] | dict[str, Any]
    statuses: set[str] = set()
    if isinstance(content, str):
        semantic = content
    elif section in {"composition_roster", "positioning"}:
        units = []
        for row in content:
            statuses.add(str(row.get("resolution_status") or "UNKNOWN"))
            unit: dict[str, Any] = {"champion": row.get("display_name") or "未解析棋子"}
            if section == "positioning":
                unit["position"] = {"x": row.get("x"), "y": row.get("y")}
                if row.get("stars") is not None:
                    unit["stars"] = row.get("stars")
            units.append(unit)
        semantic = {"units": units, "phase": "FINAL"}
    elif section in {"item_plan", "artifact", "radiant", "emblem"}:
        plans = []
        for row in content:
            statuses.add(str(row.get("identity_resolution_status") or "UNKNOWN"))
            definition = row.get("definition")
            raw_description = (
                definition.get("description") if isinstance(definition, dict) else None
            )
            description = clean_template(str(raw_description)) if raw_description else None
            plans.append({
                "holder": row.get("holder_display_name") or "未解析棋子",
                "plan": row.get("plan_index"), "slot": row.get("slot_index"),
                "item": row.get("item_display_name") or "未解析装备",
                "category": row.get("item_category"),
                "description": description,
                "resolution_status": row.get("identity_resolution_status"),
            })
        semantic = {"item_plans": plans, "association_is_not_ranking": True}
    elif section == "augment":
        associations = []
        for row in content:
            statuses.add(str(row.get("identity_resolution_status") or "UNKNOWN"))
            definition = row.get("definition")
            raw_description = (
                definition.get("description") if isinstance(definition, dict) else None
            )
            description = clean_template(str(raw_description)) if raw_description else None
            associations.append({
                "augment": row.get("display_name") or "未解析强化",
                "description": description, "source_kind": row.get("source_kind"),
                "resolution_status": row.get("identity_resolution_status"),
                "rank": None, "priority": None, "score": None,
            })
        semantic = {"augments": associations, "association_is_not_ranking": True}
    else:
        semantic = {"conditions": [{
            "kind": row.get("condition_kind"), "type": row.get("condition_type"),
            "name": row.get("display_name"), "level": row.get("level_value"),
            "trait_level": row.get("trait_level_text"),
        } for row in content]}
    return EvidenceRecord(
        evidence_id=":".join(("DataTFT-normalized", plan.game_patch, comp_id, section, suffix)),
        source="DataTFT normalized", requested_patch=plan.game_patch,
        source_patch=plan.game_patch, patch_match="exact", entity_type="composition",
        canonical_entity_id=comp_id, display_name=comp_name, section=section,
        semantic_content=semantic, relevance_score=1.0, url=source_url,
        snapshot_id=plan.active_snapshot_id, comp_id=comp_id,
        source_response_id=source_response_id,
        resolution_status=(next(iter(statuses)) if len(statuses) == 1 else
                           "MIXED" if statuses else None),
    )


def _composition_summary_evidence(
    comp: CompStatsRecord, plan: RetrievalPlan
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=":".join(("DataTFT-normalized", plan.game_patch, comp.comp_id,
                              "composition")),
        source="DataTFT normalized", requested_patch=plan.game_patch,
        source_patch=comp.snapshot.game_patch, patch_match="exact",
        entity_type="composition", canonical_entity_id=comp.comp_id,
        display_name=comp.name, section="composition",
        semantic_content={"tier": comp.tier}, relevance_score=1.0,
        url=comp.snapshot.source_url, snapshot_id=plan.active_snapshot_id,
        comp_id=comp.comp_id, source_response_id=comp.source_response_id,
    )


def _deterministic_answer(
    knowledge: RetrievedKnowledge, patch: str,
    understanding: QueryStateUnderstanding | None = None,
    plan: RetrievalPlan | None = None,
) -> FinalResponse:
    from tft_agent.app.pipeline import KnowledgeIntent
    if understanding and understanding.unresolved_terms:
        terms = "、".join(understanding.unresolved_terms)
        return FinalResponse(
            recommendation=f"当前版本实体目录无法唯一识别“{terms}”。请补充完整阵容、棋子、装备或强化名称。",
            used_tools=["canonical_entity_registry"],
        )
    relation_gaps = [gap for gap in knowledge.knowledge_gaps
                     if gap.startswith("recognized entities lack exact relation:")]
    if relation_gaps and understanding:
        names = "、".join(dict.fromkeys(
            item.display_name for item in understanding.resolved_entities
        ))
        return FinalResponse(
            recommendation=(f"已识别到实体：{names}。但当前版本 {patch} 的证据没有建立"
                            "这些实体之间的阵容玩法 relation，因此无法给出确定玩法；"
                            "我不会补造候选路线、适配结论或优先级。"),
            used_tools=["canonical_entity_registry", "structured_sqlite_retrieval"],
        )
    if (
        understanding
        and understanding.resolution_category
        == ResolutionCategory.RESOLVED_ENTITY_NO_KNOWLEDGE
    ):
        champions = [item.display_name for item in understanding.resolved_entities
                     if item.entity_type == "champion"]
        label = "、".join(dict.fromkeys(champions))
        return FinalResponse(
            recommendation=(f"已识别到英雄“{label}”，但当前知识库缺少足够可靠的阵容关联数据。"
                            "我不会仅凭攻略文本中的一次提及自动绑定阵容。"),
            used_tools=["canonical_entity_registry", "structured_sqlite_retrieval"],
        )
    if understanding and understanding.knowledge_intent == KnowledgeIntent.GENERAL_KNOWLEDGE:
        return FinalResponse(
            recommendation=(f"当前版本 {patch} 的本地数据没有装备或强化流行度统计，"
                            "因此不能把全量静态实体当作“常见”名单。"),
            used_tools=["structured_sqlite_retrieval"],
        )
    if understanding and understanding.knowledge_intent == KnowledgeIntent.COMPOSITION_BROWSE:
        comps = list(dict.fromkeys(item.display_name for item in knowledge.evidence
                                   if item.entity_type == "composition"))
        label = "S级阵容" if plan and plan.tier_filter else "按数据库 tier 优先列出的阵容"
        message = f"当前版本 {patch} {label}：" + "、".join(comps) if comps else (
            f"本地数据库没有找到 {patch} 的对应阵容事实。"
        )
        return FinalResponse(recommendation=message, used_tools=["structured_sqlite_retrieval"])
    for section, label in (
        ("composition_roster", "完整阵容"), ("positioning", "站位"),
        ("item_plan", "装备方案"), ("artifact", "神器"), ("radiant", "光明装备"),
        ("emblem", "纹章"), ("augment", "强化"),
    ):
        relevant = [item for item in knowledge.evidence if item.section == section]
        if not relevant:
            continue
        item = relevant[0]
        return FinalResponse(
            recommendation=(f"{item.display_name}的{label}："
                            f"{_deterministic_semantic(section, item.semantic_content)}"),
            used_tools=["structured_sqlite_retrieval"],
            sources=[f"{item.display_name} · {section} — {item.url}"],
        )
    requested = set(plan.knowledge_needs) if plan else set()
    missing_label = next((label for need, label in (
        ("radiant", "光明装备"), ("artifact", "神器"), ("emblem", "纹章"),
        ("augment", "强化"), ("positioning", "站位"), ("composition_roster", "完整阵容"),
        ("item_plan", "装备方案"),
    ) if need in requested), None)
    if missing_label and plan and plan.target_comp:
        return FinalResponse(
            recommendation=(f"当前版本 {patch} 的 DataTFT exact 数据中，"
                            f"{plan.target_comp.display_name}没有可用的{missing_label} relation。"),
            used_tools=["structured_sqlite_retrieval"],
        )
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


def _deterministic_semantic(section: str, content: object) -> str:
    if not isinstance(content, dict):
        return str(content)
    if section in {"composition_roster", "positioning"}:
        units = content.get("units", [])
        if not isinstance(units, list):
            return "无"
        values = []
        for unit in units:
            if not isinstance(unit, dict):
                continue
            name = str(unit.get("champion") or "未解析棋子")
            position = unit.get("position")
            if section == "positioning" and isinstance(position, dict):
                name += f"（x={position.get('x')}, y={position.get('y')}）"
            values.append(name)
        return "、".join(values) + "。"
    key = "augments" if section == "augment" else "item_plans"
    rows = content.get(key, [])
    if not isinstance(rows, list):
        return "无"
    if section != "augment":
        return format_item_plans(rows)
    values = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if section == "augment":
            value = str(row.get("augment") or "未解析强化")
            if row.get("description"):
                value += "：" + str(row["description"])
        values.append(value)
    return "；".join(values) + "。"


def _advice_options(response: FinalResponse) -> list[AdviceOption]:
    evidence = [AdviceEvidence(label=value.rsplit(" — ", 1)[0],
                               url=value.rsplit(" — ", 1)[-1], source="retrieved")
                for value in response.sources if " — http" in value]
    return [AdviceOption(title="结果", action=response.recommendation, conditions=[],
                         expected_benefit="基于当前版本本地数据。",
                         risk="数据覆盖不足时已明确标注。", confidence=1, evidence=evidence)]
