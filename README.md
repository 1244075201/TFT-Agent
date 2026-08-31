# TFT Decision Agent

## Project Goal

TFT Decision Agent is a maintainable, version-aware foundation for turning a user's
Teamfight Tactics game state into an evidence-backed decision. The current implementation
includes two narrow production structured-data tools while keeping RAG concerns out.

## Architecture

```text
User
  -> Parser
  -> GameState
  -> Agent Runtime
  -> LLM
  -> Function Call
  -> MCP Client
  -> TFT MCP Server
  -> Tool
  -> Repository
  -> SQLite / future Vector DB
  -> ToolResult
  -> Agent Runtime
  -> LLM
  -> Final Response
```

Function calling is the model-facing representation of an action. MCP is the transport
and discovery boundary used to execute that action; they are related, but not identical.

## Runtime Workflow

The parser produces factual `GameState`. `AgentRuntime` owns the bounded loop and asks an
LLM client for one typed `AgentDecision` per step. A decision may call one tool, return a
final response, or request missing context. Tool calls pass only through `MCPClient`, and
their normalized results and timings are recorded in `AgentState`. The runtime stops on a
terminal decision, an unrecoverable client failure, or the configured step limit.

## Real LLM Agent Loop

```text
User natural language
  -> DeepSeek Chat Completions
  -> native function tool_call
  -> project AgentDecision
  -> AgentRuntime
  -> stdio MCP client/server
  -> get_champion_info / get_shop_odds
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

Two tools are registered on the production MCP server. `get_champion_info` has an input schema
generated by MCP SDK v2 from the typed function signature; no hand-written duplicate JSON
Schema exists. It queries SQLite through `GameDataRepository` and returns normalized identity,
tier-derived cost, image metadata, and provenance. No match returns `CHAMPION_NOT_FOUND`;
multiple same-name candidates return `AMBIGUOUS_CHAMPION` with candidate IDs.

`get_shop_odds` returns only per-shop-slot probabilities for cost tiers 1–5 at a player level,
plus patch/source provenance. It does not recommend rolling or leveling and cannot calculate an
exact champion hit chance, pool depletion, opponent holdings, or meta strategy. The item, trait,
and knowledge contracts remain future boundaries and are not registered as production tools.

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

## Current Limitations

- Two production tools: `get_champion_info` and `get_shop_odds`.
- No RAG, vector database, or patch-note retrieval.
- No reliable active-set mapping.
- No long-term memory or planning layer.
- No vision or screenshot input.
- No parallel tool execution.
- One-shot CLI only; no persistent chat session or streaming UI.

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
