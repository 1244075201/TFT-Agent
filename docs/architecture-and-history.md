> 历史设计与迭代记录：不同版本的说明可能并存。当前入口、架构与运行方法以 [项目 README](../README.md) 为准。

# TFT Decision Agent

## Project Goal

TFT Decision Agent is a maintainable, version-aware foundation for turning a user's
Teamfight Tactics game state into an evidence-backed decision. The current implementation
includes two narrow structured-data tools plus one version-filtered knowledge retrieval tool.

The project now also includes an optional Chinese Windows desktop client, versioned DataTFT
composition snapshots, and an opt-in screenshot monitor. The monitor captures pixels only: it does
not read game memory, inject into the client, inspect opponents, or automate input.

> Riot's published TFT policy does not permit dynamic real-time recommendations based on current
> game state. The desktop application therefore starts in compliant mode. Experimental monitoring
> is local-only, disabled by default, and must not be publicly distributed without Riot approval.

## Architecture

```text
User Query + GameState
  -> scoped entity resolution + bounded semantic guide planning
  -> structured SQLite / text retrieval + optional verified web evidence
  -> deterministic factual answer, or Context Builder -> one DeepSeek answer call
```

`AgentService` is the single composition root for CLI and desktop. Explicit factual browsing uses
the local SQLite path without an LLM. Guide questions, including named heroes, augments, emerging
combinations and follow-ups, make one semantic planning call before retrieval and one answer call
afterwards. Planner failures fall back to local query/entity retrieval. The production path
does not use model-selected tools, an agent loop, or an LLM repair call. Local parsing and output
normalization may recover fenced JSON, partial envelopes, aliases, and useful plain text.

The legacy `AgentRuntime` and stdio MCP integration remain available for compatibility and
regression tests, but neither is referenced by the production CLI or desktop answer path.

Function calling is the model-facing representation of an action. MCP is the transport
and discovery boundary used to execute that action; they are related, but not identical.

## V0.5 Runtime Workflow

The version-numbered sections below document earlier milestones; the current guide workflow is
described above and in the following section.

### Guide language and continuously updated evidence

Guide planning separates goals, preferences, constraints, mentioned entities, unknown names and
relationships. For example, “上限” describes late-game potential, “赌狗” describes reroll/three-star
play (not necessarily a fixed unit cost), and “吃烂分” describes a placement-oriented goal. These
are retrieval objectives, never measured win rates or evidence that a player owns an augment.
Names such as “地狱火艾希” can be searched as a combination without creating a composition ID.

Optional live supplementation uses Brave Web Search to discover URLs, then fetches article bodies
from the HTTPS domain allowlist in `.env.example`. Configure
`TFT_AGENT_KNOWLEDGE_SEARCH_API_KEY` with a Brave Search API key to enable it. The adapter is
replaceable through `SearchAdapter` / `PageFetcher`; no new package is needed. A missing key or
network failure is explicitly disclosed in the answer. Search descriptions are never evidence.
The page must contain an explicit exact patch, sufficient body text and the requested entities.
Cross-patch, ambiguous-version and untrusted pages are rejected; original data is not relabelled.
JavaScript-only pages, videos and unavailable/blocked articles are not bypassed or interpreted.

The desktop's single update coordinator checks at startup and hourly for a daily update
(24-hour default). It checks official CN announcements, fetches a candidate DataTFT snapshot,
validates it and publishes its canonical catalog and active pointer in one transaction. It then
revisits up to 12 tracked guide questions and a discovery query. Questions retain independent
on-demand supplementation. No operating-system scheduled task is installed; the worker only
runs while the app is open. CLI questions use on-demand supplementation.

Server truth is stored separately from source snapshots in `application_updates`. Confirmed
server upgrades exclude old-patch recommendations even if the new DataTFT snapshot is missing.
An unverified server check may refresh the already active source version, but cannot promote a
different version. Answers disclose that server status is unverified. Official previews, foreign
articles and community patch labels cannot confirm a CN rollout; announcements without an exact
hotfix identity or verifiable effective time remain unverified. `TFT_AGENT_SERVER_VERSION_URLS`
can supply official announcement URLs alongside the official index and search discovery.
Static CommunityDragon imports retain their explicit ingestion workflow; old facts are not
relabeled as a new patch. Failed updates preserve history and retry with bounded backoff.

Guide answers bind each statement to exact evidence IDs. Citations retain composition, snapshot,
patch and source identity rather than collapsing by URL. DataTFT overview links open an in-app
viewer with the saved evidence text; the overview remains an explicitly labeled secondary link.
Verified web article citations may open their article directly. Source tiers are author ratings,
not match statistics. Questions and publication are serialized within the desktop process so a
single answer cannot mix snapshots. The sidebar shows server status, data version and refresh state.

Verified pages are stored in `web_knowledge_revisions` with URL, exact patch, publication/update
timestamps, retrieval/check times, content hash and immutable revision identity. Unchanged content
only updates its check time. Per-question checks live in `web_knowledge_queries`; failed checks
back off for 15 minutes and preserve good revisions. Same-instance concurrent requests share the
same lookup lock/cache, including the desktop's separate event loops. Source revisions from
different URLs remain distinct so disagreements are visible. Unknown publication dates and old
content cannot establish current strength. A single source may support conditional advice, with
an explicit evidence limitation; numerical statistics are not extracted from arbitrary prose.

No new entity aliases are automatically promoted. Existing exact facts, scoped state, conditional
evidence and unsupported-statistic checks remain in effect. Web content is untrusted quoted data,
not instructions for the model.

Offline regressions and a separate, billable real-provider intent evaluation:

```bash
uv run pytest tests/test_live_knowledge.py tests/test_guide_knowledge_flow.py
uv run tft-eval-guide-understanding --limit 10
```

The human-authored cases are in `evals/guide_understanding_cases.json`. Provider artifacts retain
the plan, contract checks and human-review target, not hidden reasoning. Contract pass rates are
not reported as semantic or strategy accuracy. Offline fixture responses are not model benchmarks.

See [the daily-update validation record](data-update-validation-2026-09-22.md) for
the repaired DataTFT adapter, actual snapshot updates, evidence-link checks and the remaining
official-server-version verification limitation. The earlier guide-only validation remains in
[its historical record](guide-validation-2026-09-22.md).

The parser, editor, and optional vision layer produce factual `GameState`. Missing optional fields
are recorded as context gaps and do not block conditional guidance. Retrieval is patch-filtered and
uses normalized DataTFT, static entity, shop-odds, and knowledge-document tables. Existing embedding
code is retained for legacy experiments, but the main retrieval path does not instantiate it or
depend on a vector database.

The desktop separates “阵容推荐” (current-version browsing and one-page comp summaries) from
“对局指导” (decisions based on the player's current state). The first is factual; deciding what to
play from items, augments, or units belongs to match guidance and therefore uses the one-call path.

## V0.4 Rich Decision Context

`GameState` accepts explicit board and bench units (including star level and held items),
classified component and completed-item inventory, augments, and a small win/loss streak fact.
The deterministic text parser only fills these fields from labeled input; it does not infer board
strength or actions such as whether to roll. Nested facts are individually registered and aliased
for final-answer provenance.

Rich state does not add a tool. Production remains limited to `get_champion_info`,
`get_shop_odds`, and `search_game_knowledge`; listing several owned units must not cause one
champion lookup per unit. The V0.4 provider-backed board evaluation is opt-in:

```bash
uv run tft-eval-board-v04
uv run tft-eval-board-v04 --case no_explosion_001
```

## V0.5 Vision Observation Layer

V0.5 adds an observation-only image boundary. `DeepSeekVisionClient` reads a user-provided local
PNG/JPEG, sends it as an in-memory base64 data URL to the independently configured
`deepseek-v4-flash-vision-exp` model, validates a strict `VisionObservation`, and then applies an
explicit-user-first merge into `GameState`. Vision never produces a strategy decision.

Unseen fields remain null and are listed in `not_visible_fields`; ambiguous fields are listed in
`uncertain_fields`. User/vision disagreements become `StateConflict` records rather than silent
overwrites. Vision evidence has separate `vision.<observation_id>.*` paths and can ground factual
claims or tool arguments, but it cannot by itself authorize a current-meta strategy assertion.
Trace records hashes and dimensions, never image base64.

```bash
uv run tft-vision-extract screenshot.png
uv run tft-eval-vision-v05 --detail high
uv run tft-eval-vision-e2e-v05
```

The checked-in V0.5 image set is a controlled synthetic baseline, not a claimed real-game accuracy
benchmark. Real screenshots require separate human-authored ground truth before reporting real-game
visual accuracy.

## Desktop Application and DataTFT Snapshots

Install dependencies and start the Chinese desktop chat UI:

```bash
uv sync --all-extras
uv run tft-agent-desktop
```

On Windows, after the first `uv sync --all-extras`, double-click
`启动云顶指导Agent.cmd` in the project folder; no PowerShell command is required.

The UI supports persistent local chat, screenshot selection, cancellation, and an explicitly gated
experimental monitor. Chat history is stored in `data/chat_history.json`; screenshots are not kept.

Runtime never scrapes DataTFT. Import a reviewed canonical JSON export (recommended):

```bash
uv run tft-sync-datatft --input data_sources/datatft_snapshot.example.json
uv run tft-sync-datatft --input snapshot.json --patch 18.2
```

Without `--input`, the synchronizer checks `robots.txt`, then uses the public frontend composition
list and strategy requests with a configurable delay between details. Each version, list, and
per-composition detail response is stored with its HTTP provenance before normalization. It imports
the real current
tiers, core units, item IDs, and early/mid/final operation text. The current endpoints do not expose
placement/top-four/win/play rates, so those values remain null. It does not bypass authentication,
CAPTCHAs, rate limits, or access restrictions. A canonical snapshot contains `game_patch`, optional
`sample_scope`, and a `comps` array with named statistics and guide fields. Every record retains its
source URL, retrieval time, content hash, server, and exact patch. A candidate patch becomes active
only after its complete snapshot and patch-scoped canonical catalog pass validation; activation is
atomic and a failed run leaves the previous active patch unchanged. Historical patches and immutable
snapshots remain queryable by an explicit exact-patch override. Settings provide the empty-database
bootstrap scope; once initialized, the SQLite patch registry is runtime authority.

## Real LLM Agent Loop

```text
User natural language
  -> DeepSeek Chat Completions
  -> native function tool_call
  -> project AgentDecision
  -> AgentRuntime
  -> stdio MCP client/server
  -> get_champion_info / get_shop_odds / search_game_knowledge
  -> ToolResult
  -> native assistant tool-call + tool message history
  -> DeepSeek
  -> FinalResponse
```

Tool calls use the provider's native `tools` and `tool_calls` protocol. Function arguments
are parsed as JSON and validated against the discovered MCP input schema before execution.
Tool results return through a native `role="tool"` message correlated by `tool_call_id`.
Each tool message identifies its authoritative fields and provenance; version-sensitive facts
outside that boundary must be reported as unknown rather than inferred from identifiers or model
memory. Hidden model reasoning is neither stored nor logged.

At task start, the system prompt receives a short capability context generated from the live MCP
tool names and descriptions. The native function schemas remain in the provider `tools` parameter,
so there is no second hand-maintained capability list or duplicated JSON Schema. Capability answers
must distinguish general reasoning over user-provided facts from reliable external data access.

Ordinary final answers are concise natural language rather than a mandatory large JSON contract.
`FinalResponse` supports a recommendation-only fact answer; `used_tools` and `sources` are overwritten
by `AgentRuntime` from actual tool history and provenance, never trusted from model output.

## LLM Provider Boundary

`AgentRuntime` depends only on the project `LLMClient` protocol. `DeepSeekLLMClient` owns the
OpenAI-compatible SDK, DeepSeek request parameters, bounded transient retries, native message
conversion, response normalization, and provider error classification. SDK response objects do
not cross this boundary. MCP `list_tools()` remains the only source for provider function schemas.

Parallel tool execution is intentionally unsupported. If a response contains multiple tool calls,
the adapter accepts only the first in provider order; the next bounded step reassesses after its
real observation. This permits sequential multi-tool decisions without a planner or concurrency.

## DeepSeek Configuration

The centralized settings module reads these environment variables from the local `.env` file:

- `DEEPSEEK_API_KEY`
- `TFT_AGENT_MODEL`
- `DEEPSEEK_BASE_URL`
- `TFT_AGENT_THINKING_MODE`

The local `.env` is ignored by Git. Never place credentials in this README or tracked files.

## Data Workflow

```text
Official / trusted data -> Fetch -> Parse -> Normalize -> Structured Store
Documents -> Clean -> Chunk -> Embed -> Vector DB
```

Ingestion is independent of runtime retrieval. It can be rerun, tested, and replaced
without teaching the agent loop how sources are downloaded or stored.

## Structured Data Pipeline

```text
Riot Data Dragon versions endpoint
  -> RiotDataDragonSource
  -> validated official payload models
  -> normalized ChampionRecord
  -> atomic SQLite version replacement
  -> GameDataRepository
  -> get_champion_info MCP tool
```

`RiotDataDragonSource` is the only layer that performs Riot HTTP requests. Runtime queries
never call Riot directly. The source currently observes document fields `type`, `version`,
and `data`; champion fields `id`, `name`, `tier`, `cost`, and `image`; and image fields
`full`, `sprite`, `group`, `x`, `y`, `w`, and `h`. `tier` is normalized as the champion
cost because current payloads contain `cost: 0`; no traits, ability, or combat stats are
invented. Unknown future source fields are ignored at this adapter boundary.

Shop odds use a separate, replaceable source boundary because Riot Data Dragon does not expose a
complete shop-odds table. The checked-in reviewed snapshot records the current LoLCHESS.GG reroll
reference, patch `16.17`, Set 18 source version, retrieval timestamp, and URL. It is explicitly
stored as `community_reference`, never `riot_official`. `FileShopOddsSource` normalizes that
versioned snapshot, and `tft-sync-shop-odds` atomically replaces the corresponding SQLite rows.
Runtime lookup remains offline and never scrapes a page during a tool call.

## Version Model

TFT `game_patch` and Data Dragon `data_version` are independent identifiers:

```text
game_patch != data_version
```

Ingestion resolves `data_version` dynamically from Riot's official versions endpoint.
It does not infer a TFT game patch from that value. A future verified mapping may relate
the two, but string equality is not a valid mapping policy.

## Current Vertical Slice

Three tools are registered on the production MCP server. `get_champion_info` has an input schema
generated by MCP SDK v2 from the typed function signature; no hand-written duplicate JSON
Schema exists. It queries SQLite through `GameDataRepository` and returns normalized identity,
tier-derived cost, image metadata, and provenance. No match returns `CHAMPION_NOT_FOUND`;
multiple same-name candidates return `AMBIGUOUS_CHAMPION` with candidate IDs.

`get_shop_odds` returns only per-shop-slot probabilities for cost tiers 1–5 at a player level,
plus patch/source provenance. It does not recommend rolling or leveling and cannot calculate an
exact champion hit chance, pool depletion, opponent holdings, or meta strategy. The item, trait,
and exact-hit contracts remain outside the production tool set.

`search_game_knowledge` retrieves version-filtered chunks from a small controlled SQLite corpus.
It covers Riot official patch/set material and limited attributed community strategy guides.
Strategy comp/entity/topic metadata gates explicit named-comp queries before vector similarity;
retrieval never replaces structured champion cost or shop-odds evidence.

## Module Responsibilities

- `app`: composition root and future CLI entry point only.
- `agent`: parsing, validation, execution state, loop control, and response formatting.
- `llm`: model-neutral client contract, prompts, and MCP-to-function schema adaptation.
- `mcp_client`: model-neutral contract plus an official SDK v2 stdio adapter.
- `tft_mcp_server`: injected MCP SDK server composition and repository-backed handlers.
- `tft_mcp_server.repositories`: storage-independent data access contracts.
- `ingestion`: separate structured-data and knowledge ingestion workflows.
- `core`: genuinely shared schemas, configuration, and exceptions.

## Design Principles

- State is not memory: `AgentState` is bounded execution state, not long-term memory.
- `GameState` contains facts, never recommendations or inferred board quality.
- Structured retrieval is not RAG; exact game records and semantic evidence have distinct contracts.
- Function calling is not MCP; adapters translate between the two boundaries.
- RAG retrieves evidence; the LLM reasons over evidence and current facts.
- Runtime never directly accesses storage; it invokes tools through an MCP client.
- `game_patch` is not necessarily the underlying `data_version`.
- Provider SDKs and storage engines remain behind replaceable interfaces.

## Development

```bash
uv sync --all-extras --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The project targets the Python version in `.python-version` and uses only `pyproject.toml`
plus `uv.lock`; there is no parallel requirements file.

Synchronize official champion data:

```bash
uv run tft-sync-game-data --database data/tft.sqlite3
```

Load the reviewed shop-odds reference snapshot into the same database:

```bash
uv run tft-sync-shop-odds --database data/tft.sqlite3
```

Start the local stdio server (normally the client launches this process):

```bash
uv run tft-mcp-server --database data/tft.sqlite3
```

The server writes protocol messages only to stdout; application logging uses stderr. The
stdio client is an async context manager so leaving its block closes the connection and child
process.

Run one real task after synchronizing data and configuring DeepSeek:

```bash
uv run tft-agent
```

## V0.3 Release Baseline

V0.3 adds the retrieval and evidence-backed strategy vertical slice while retaining the V0.2
grounding guarantees. The final stable provider-backed baselines are:

- Strategy Decision Set: 14/14 in one complete formal run.
- Grounded V0.2 Set: 14/14, with zero unsupported numeric claims and zero unauthorized arguments.
- Three-tool selection: 23/26 (accuracy 0.885, precision 0.905, recall 0.950); the single Shop FN
  passed an immediate targeted confirmation.
- Strategy relevance: 10/10, with zero entity/topic false positives and perfect hard-negative
  precision in the deterministic relevance set.

Finalization uses current-task evidence aliases, mutually exclusive FINAL/REQUEST_CONTEXT actions,
one bounded orchestration repair path, and deterministic safe fallback. Canonical evidence paths,
source authority, patch filtering, numeric validation, and tool-argument validation remain strict.

Known non-blocking provider variation remains in tool selection: global-best and unsupported Wisp
positioning questions may call RAG before receiving `knowledge_not_found` and stopping honestly.
Occasional structured-tool false negatives have passed immediate confirmation and have not bypassed
grounding or safety checks.

## Current Limitations

- Four production tools; DataTFT support currently requires a compatible public payload or reviewed
  canonical JSON export because DataTFT does not publish a stable API contract.
- The controlled strategy corpus is deliberately small and does not cover the entire meta.
- Live web supplementation needs a configured search key and readable, exact-patch sources;
  it cannot guarantee comprehensive coverage. Main local retrieval has no BM25 or reranker.
- No long-term memory or planning layer.
- No parallel tool execution.
- Real-screen accuracy still requires a human-labelled CN-client screenshot benchmark.

## Agent Evaluation

Tool selection is measured offline against the versioned Golden Set in
`evals/tool_selection_cases.json`. The workflow calls the real configured provider and reads the
existing agent trace; it does not add evaluation branches to `AgentRuntime` and never falls back to
a fake model while reporting real benchmark results.

The dataset policy marks unsupported champion fields as `no_call`: the current champion tool should
only be selected when identity, cost, image metadata, or Data Dragon version directly contributes to
the answer. In particular, the `champion_entity_but_tool_irrelevant` slice measures false positives
where a champion name is present but the actual request concerns items, positioning, composition,
meta strength, role, or trait strategy.

Metrics include accuracy, tool-call precision and recall, false-positive/negative rates, per-slice
analysis, latency, agent/LLM/tool call counts, and provider-reported token usage. Precision or recall
with a zero denominator is reported as `N/A`, not zero. Results contain visible final text for manual
review but no hidden reasoning or secrets.

The full evaluation is explicitly billable and opt-in:

```bash
uv run tft-eval-tool-selection
uv run tft-eval-tool-selection --slice champion_entity_but_tool_irrelevant
uv run tft-eval-tool-selection --case direct_001
uv run tft-eval-tool-selection --limit 3
```

Timestamped JSON artifacts are written under `eval_results/` and ignored by Git. Each artifact
captures the model, thinking mode, dataset version, system-prompt version/hash, tool-description
version/snapshot, overall and per-slice metrics, and per-case TP/TN/FP/FN classification.

Tool Selection Evaluation and Outcome Safety Evaluation measure different boundaries. Selection
asks whether the model chose an appropriate tool. Outcome safety checks deterministic behavior after
a structured tool failure, including whether ambiguity stopped the loop before an unauthorized
follow-up call. Run the small outcome regression set explicitly with:

```bash
uv run tft-eval-outcome-safety
uv run tft-eval-outcome-safety --case ambiguity_001
```

## Semantic Decisions vs Runtime Invariants

The LLM owns semantic choices: whether available evidence is relevant, which tool can help, concise
reasoning, and visible natural-language answers. `AgentRuntime` owns deterministic invariants:
bounded steps, tool-schema validation, structured error resolution, and the ambiguity stop policy.

Every `ToolError` declares one provider-neutral resolution: `retry`, `request_context`, or
`terminal`. A `request_context` failure is recorded as evidence and immediately transitions the task
to `waiting_for_context`; the LLM cannot select one of the candidates in the same autonomous loop.
A `terminal` failure ends the tool path with a minimal honest response. Retry is part of the contract,
but automatic tool retry orchestration is intentionally not implemented yet. The runtime never
branches on TFT-specific error codes or champion names.

## V0.2 Decision Evaluation

The versioned 12-case set in `evals/decision_v0_2_cases.json` checks direct shop-odds lookup,
grounded roll decisions, missing-context requests, wrong-tool temptation, and useful two-tool
chains. Assertions are deterministic: expected tool order, status, tool success, and trace data.
Final strategy text is retained for manual review; no judge LLM assigns a strategy score.

```bash
uv run tft-eval-decision-v02
uv run tft-eval-decision-v02 --case decision_001
uv run tft-eval-decision-v02 --slice two_tool_useful_chain
```

## Grounded Decision Policy

Structured evidence constrains factual precision; the LLM remains responsible for qualitative
decision reasoning. The provider receives a compact, source-separated evidence context containing
explicit `GameState` facts and authoritative ToolResults. Trace records the corresponding evidence
field paths without storing hidden reasoning.

Specific numerical claims about game state, costs, shop odds, probabilities, thresholds, rounds,
pool sizes, patches, or other version-sensitive mechanics must come from user facts, ToolResults,
or transparent deterministic arithmetic over those values. A comparison may query another level
when that evidence materially helps; otherwise the answer must remain qualitative. Unsupported
economy cutoffs, exact survival estimates, and exact champion hit probabilities are prohibited.

The separate opt-in regression set keeps its numerical and regex detectors strictly in the eval
layer; production output is not post-processed or censored:

```bash
uv run tft-eval-grounded-decision
uv run tft-eval-grounded-decision --case comparison_001
uv run tft-eval-grounded-decision --slice unsupported_economy_threshold
```

## Tool Argument Grounding

An LLM may propose a native ToolCall, but its arguments are not trusted merely because they satisfy
the MCP JSON Schema. Each argument receives an internal `ToolArgumentClaim` referencing current-task
evidence or a small deterministic derivation. A provider-neutral validator runs before MCP execution;
unsupported, mismatched, or invalid references block the call and move the task to
`WAITING_FOR_CONTEXT`. This provenance is an Agent control contract and is not added to MCP schemas.

## Numeric Claim Provenance

Prompting guides model behaviour; deterministic validation enforces factual precision. A final
provider response uses a small segmented envelope containing supported fact sentences, their
`numeric_claims`, and a quantity-free qualitative recommendation. Each user-visible number declares
a meaning, unit, and either an exact current-task evidence reference or a small arithmetic
derivation over supported values. The user sees only the deterministically assembled answer, not
this provenance metadata. Unicode numeric-property checking covers number symbols across languages
without a language-specific production blacklist.

The task-local evidence registry indexes scalar `GameState` and ToolResult fields. Validation
distinguishes supported, unsupported, mismatched, and invalid-reference claims, including percent
normalization such as `10% == 0.10`. Unsupported claims trigger at most one no-tools repair call.
If repair is still invalid, a deterministic fallback preserves validated facts using normalized
numeric rendering and replaces unsafe precision. Answers without quantities pass directly and incur
no repair call.

Factual precision is controlled by structured evidence, while qualitative strategy remains
model-generated.
