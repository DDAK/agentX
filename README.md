<p align="center">
  <img src="https://img.shields.io/badge/built_with-Rust-orange?style=flat-square&logo=rust" alt="Built with Rust" />
  <img src="https://img.shields.io/badge/runtime-Tokio-blue?style=flat-square" alt="Tokio" />
  <img src="https://img.shields.io/badge/LLM-Multi--Provider-purple?style=flat-square" alt="Multi-Provider LLM" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License" />
  <img src="https://img.shields.io/badge/docker-compose-blue?style=flat-square&logo=docker" alt="Docker Compose" />
</p>

<h1 align="center">AgentX</h1>

<p align="center">
  <strong>A blazing-fast, open-source code-editing AI agent built in Rust.</strong><br/>
  LLM + agentic loop + tools — self-hosted, multi-provider, production-ready.
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#api-reference">API</a> &bull;
  <a href="#extending">Extending</a> &bull;
  <a href="#contributing">Contributing</a>
</p>

---

## Why AgentX?

Most AI coding agents are Python wrappers around a single LLM. AgentX is different:

- **Rust-native** — async, zero-GC, single binary. Handles hundreds of concurrent sessions without breaking a sweat.
- **Provider-agnostic** — swap between Anthropic, OpenAI, and Gemini without changing a line of code. Load-balance across providers with automatic failover.
- **Sandboxed code execution** — the agent can write *and run* Python scripts in a secure [Monty](https://github.com/pydantic/monty) sandbox. No Docker-in-Docker, no shell escapes.
- **Production-grade** — WebSocket streaming, session persistence (filesystem or Postgres), structured hooks for observability, human-in-the-loop confirmation gates.
- **Self-hosted** — your keys, your data, your infrastructure. No SaaS dependency.

---

## Features

| Category | Details |
|----------|---------|
| **Agentic Loop** | Autonomous tool-calling loop with configurable iteration limits (default: 32 iterations/turn) |
| **6 Built-in Tools** | `read_file`, `write_file`, `edit_file`, `list_files`, `run_command`, `run_code` |
| **Sandboxed Python** | Execute agent-authored Python with host-bridged tool access (512 MiB heap, 30s timeout) |
| **Multi-Provider LLM** | Anthropic Claude, OpenAI GPT-4o, Google Gemini — via LiteLLM gateway with least-busy routing |
| **Real-time Streaming** | WebSocket + SSE for live token-by-token output |
| **Session Persistence** | Filesystem (zero-config) or Postgres (scalable) — resume conversations anytime |
| **Hook System** | Intercept any lifecycle event: logging, metrics, confirmation gates, post-processing |
| **Dual Mode** | HTTP API server (production) or interactive CLI REPL (development) |
| **Web UI** | Dark-mode chat interface with session management, markdown rendering, tool call visibility |
| **Docker Compose** | One command to run the full stack: LiteLLM + Postgres + Agent + Frontend |

---

## Architecture

```
┌───────────┐    WebSocket / SSE (token-streamed)    ┌──────────────────────┐
│ Frontend  │ ◄────────────────────────────────────► │  AgentX API Server   │
│ (Vite)    │                 :3000                  │  (Rust / Axum) :8080 │
└───────────┘                                        └──────────┬───────────┘
                                                                │
        ┌───────────────────────────┬───────────────────────────┼───────────────────────────┐
        │                           │                           │                           │
        ▼                           ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────────┐       ┌──────────────────┐        ┌─────────────────┐
│  Storage      │         │  LiteLLM Gateway  │       │  Tool Registry   │        │ Dynamic         │
│               │         │  (shared:         │       │                  │        │ Extensions      │
│ • Filesystem  │         │   atom-litellm)   │       │ • 5 built-ins    │        │ (AGENTX_HOME,   │
│ • Postgres    │         │  stream: true     │       │ • run_code       │        │  loaded @ start)│
└───────────────┘         └─────────┬─────────┘       │ • MCP tools      │        │                 │
                                    │                 │ • Rhai commands  │        │ • config.yaml   │
                          ┌─────────┼─────────┐       └────────┬─────────┘        │ • skills/*.md   │
                          ▼         ▼         ▼                │                  │ • commands/*.rhai│
                    ┌─────────┐ ┌────────┐ ┌────────┐          │                  └────────┬────────┘
                    │ Claude  │ │ OpenAI │ │ Gemini │          │                           │
                    │ (Haiku/ │ │ GPT-4o │ │ Flash  │          ▼                  (feeds prompt + tools)
                    │  Opus)  │ │        │ │        │   ┌───────────────────┐
                    └─────────┘ └────────┘ └────────┘   │  MCP Servers      │
                                                        │ stdio / HTTP / SSE│
                                                        │ e.g. postgres-mcp │
                                                        └───────────────────┘
```

### Component Overview

| Component | Responsibility |
|-----------|---------------|
| **`agent.rs`** | Core agentic loop — channel-driven, headless. Receives user messages, calls LLM, dispatches tools, emits typed events. |
| **`api.rs`** | Axum HTTP server — REST endpoints, WebSocket upgrades, token-streamed SSE. |
| **`llm.rs`** | LiteLLM gateway client — OpenAI-compatible chat completions with `stream: true`, parsed token-by-token. No SDK bloat. |
| **`tools.rs`** | Tool trait + registry. Built-ins + MCP tools + Rhai commands, all behind one trait. |
| **`mcp.rs`** | Model Context Protocol client — connects external tool servers over stdio, HTTP, or SSE; each remote tool becomes a native `Tool`. |
| **`scripting.rs`** | Dynamic extensions — loads `config.yaml`, `skills/*.md`, and Rhai `commands/*.rhai` from disk at startup. |
| **`code_tool.rs`** | Sandboxed Python execution via Monty — lets the agent write programs that call tools (incl. MCP via `call_tool`). |
| **`hooks.rs`** | Hook chain for lifecycle events. Built-in: logging, tool announcer, command confirmation. |
| **`storage.rs`** | Storage trait + FilesystemStorage + PostgresStorage with session CRUD. |
| **`config.rs`** | Environment-based configuration with sensible defaults. |

### Data Flow

```
User Message
    │
    ▼
┌─────────────────────────────────────────────┐
│  Agent Loop (agent.rs)                      │
│                                             │
│  1. Append user message to conversation     │
│  2. Emit AgentEvent::Thinking               │
│  3. Stream LLM completion (conversation +   │
│     tool defs); emit each token as Text     │
│  4. If no tool_calls → done                 │
│  5. If tool_calls:                          │
│     a. Fire BeforeToolExecution hook         │
│     b. Execute tool via ToolRegistry         │
│        (built-in / MCP / Rhai — same path)   │
│     c. Append result to conversation         │
│     d. Fire AfterToolExecution hook          │
│     e. Loop back to step 3                  │
│  6. Emit TurnDone                           │
│  7. Persist session to storage              │
└─────────────────────────────────────────────┘
```

### Recent Architectural Changes — What & Why

The diagram above reflects four changes layered onto the original design. Each was added without disturbing the core agent loop, because the loop only ever talks to a `Tool` trait and a streaming LLM client — extension points, not hard-coded lists.

| Change | What it adds | Why it matters |
|--------|-------------|----------------|
| **Token streaming** (`llm.rs`, `api.rs`) | `stream: true` to the gateway, parsed token-by-token and pushed out over SSE as `Text` events as they arrive. | Users see output immediately instead of waiting for the full completion — the difference between a tool that feels live and one that feels hung on long answers. |
| **MCP tool servers** (`mcp.rs`) | External tool servers over stdio, HTTP, or SSE. Each remote tool is wrapped as a native `Tool` (`mcp__<server>__<tool>`). | Capabilities can be added by pointing at a server — no code change. The Postgres MCP server, for example, gives the agent 9 database tools for free. |
| **Shared LiteLLM gateway** (`docker-compose.yml`) | The bundled local gateway was dropped; the agent now uses a shared external gateway (e.g. `atom-litellm`) on the LLM network. | One gateway, one set of keys and budgets across services — no duplicated config, no second container to keep in sync. (The old service is preserved as a comment for rollback.) |
| **Dynamic extensions** (`scripting.rs`) | `config.yaml` (behaviour), `skills/*.md` (prompt fragments), and Rhai `commands/*.rhai` (scripted tools), all loaded from `AGENTX_HOME` at startup. | Behaviour, knowledge, and new commands change by editing files and restarting — **no recompile, no rebuild**. The binary stays generic; deployments specialise it on disk. |

The unifying principle: **everything dynamic lives behind a trait or on disk.** New tools (MCP, Rhai), new behaviour (YAML/MD), and incremental output (streaming) all plug into seams the agent loop already exposes, so the hot path never had to change.

---

## Quickstart

### Prerequisites

- **Docker + Docker Compose v2** (recommended path)
- At least one LLM API key: Anthropic, OpenAI, or Google Gemini

For local development without Docker:
- Rust 1.82+ (`rustup update stable`)
- Node.js 22+ (frontend only)

### Option 1: Docker Compose (recommended)

```bash
# Clone the repo
git clone https://github.com/DDAK/agentX.git
cd agentX

# Configure environment
cp .env.example .env
# Edit .env — set at least one API key:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...
#   GEMINI_API_KEY=...

# Start the full stack
docker compose up --build
```

This starts four services:

| Service | Port | Description |
|---------|------|-------------|
| `litellm` | 4000 | LLM gateway with load-balancing and retry |
| `postgres` | 5432 | Session persistence (profile: optional) |
| `agent` | 8080 | Rust API server |
| `frontend` | 3030 | Web chat UI |

Open **http://localhost:3030** — the UI auto-creates a session and connects.

### Option 2: Local Development

```bash
# Clone and configure
git clone https://github.com/DDAK/agentX.git
cd agentX
cp .env.example .env
# Edit .env with your API key(s)

# Start infrastructure only
docker compose up litellm -d

# Run the agent (server mode)
cargo run

# Or run in interactive CLI mode
cargo run -- --cli
```

For the frontend:

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000 (proxies API to :8080)
```

---

## Environment Variables

### Agent / Server

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTX_MODE` | `serve` | `serve` = HTTP API, `cli` = terminal REPL |
| `BIND_ADDR` | `0.0.0.0:8080` | Server listen address |
| `STORAGE_BACKEND` | `filesystem` | `filesystem` or `postgres` |
| `WORKSPACE_DIR` | `.` | Root for file tool operations |
| `CONFIRM_COMMANDS` | `false` | Require y/N before `run_command` (CLI only) |
| `AGENTX_HOME` | `agentx.d` | Dynamic extensions dir (see [Dynamic Extensions](#dynamic-extensions)) |
| `RUST_LOG` | `agentx=info,warn` | Tracing filter |

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://localhost:4000` | Gateway URL |
| `LITELLM_API_KEY` | `sk-agentx-dev` | Must match `LITELLM_MASTER_KEY` |
| `AGENT_MODEL` | `claude-3-7-sonnet-20250219` | Model name from `litellm_config.yaml` |
| `AGENT_MAX_TOKENS` | `8192` | Max tokens per response |
| `AGENT_TEMPERATURE` | `0.0` | Sampling temperature |
| `MCP_CONFIG` | `.mcp.json` | Path to the MCP server config (see [MCP Tool Servers](#mcp-tool-servers)) |

### Provider Keys

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Google Gemini API key |

### Storage (Postgres)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | `postgres://user:pass@host:5432/db` |
| `POSTGRES_PASSWORD` | `agentx_dev` | Used by Docker postgres service |

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/api/sessions` | List all sessions (most-recent first) |
| `POST` | `/api/sessions` | Create session: `{ "label": "optional" }` |
| `GET` | `/api/sessions/:id` | Get session with full message history |

### WebSocket — `GET /api/sessions/:id/ws`

Bidirectional real-time communication.

**Send:**
```json
{ "text": "refactor the auth module to use JWT" }
```

**Receive** — a stream of typed `AgentEvent` objects:
```jsonc
{ "type": "thinking" }
{ "type": "text",          "text": "I'll start by reading the current auth code…" }
{ "type": "tool_call",     "name": "read_file", "input": { "path": "src/auth.rs" } }
{ "type": "tool_result",   "name": "read_file", "result": "use jsonwebtoken::…" }
{ "type": "text",          "text": "Here's the refactored implementation…" }
{ "type": "turn_done" }
```

### SSE — `GET /api/sessions/:id/sse`

Same events as WebSocket, delivered as Server-Sent Events. Send messages via:

```bash
curl -X POST http://localhost:8080/api/sessions/<id>/message \
     -H 'Content-Type: application/json' \
     -d '{"text": "list the files in this project"}'
```

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read full contents of a file |
| `write_file` | Create or overwrite a file (auto-creates directories) |
| `edit_file` | Replace an exact unique string — safer than full rewrites |
| `list_files` | List directory entries (dirs get trailing `/`) |
| `run_command` | Execute shell command; returns exit code + stdout + stderr |
| `run_code` | Execute Python in sandboxed Monty interpreter with access to all tools above |

### The `run_code` Sandbox

The `run_code` tool is what makes AgentX uniquely powerful. Instead of issuing one tool call per action, the agent writes Python scripts that orchestrate multiple operations:

```python
# Agent can write code like this:
files = await list_files("src")
for f in files:
    if f.endswith(".rs"):
        content = await read_file(f"src/{f}")
        if "TODO" in content:
            print(f"Found TODO in {f}")
```

**Security constraints:**
- 512 MiB heap limit
- 200 stack frames max
- 30-second wall-clock timeout
- No direct OS/filesystem access — must use host functions
- Hook chain applies to all tool calls from within the sandbox

### MCP Tool Servers

AgentX can pull in tools from external [Model Context Protocol](https://modelcontextprotocol.io) servers over two transports: **stdio** (a `command` launches a local subprocess) and **HTTP** (a `url` for a remote/hosted server). Drop a `.mcp.json` in the working directory (or point `MCP_CONFIG` at one):

```json
{
  "mcpServers": {
    "everything": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-everything"],
      "env": {}
    },
    "remote": {
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

On startup AgentX connects each server, performs the MCP handshake, and registers its tools alongside the built-ins. Discovered tools are namespaced `mcp__<server>__<tool>` so two servers can expose the same tool name without colliding. A server that fails to start is logged and skipped — it never takes down the agent. MCP is fully opt-in: with no config file, nothing is spawned. The hook chain applies to MCP tool calls just like built-ins.

Requests are dispatched **concurrently** — each carries a unique JSON-RPC id, and stdio responses are routed back to the awaiting caller by id, so a slow call never blocks a fast one.

MCP tools are also reachable from inside the `run_code` sandbox via the generic `call_tool` host function:

```python
results = await call_tool("mcp__everything__search", query="rust")
```

---

## Dynamic Extensions

Configuration, skills, and commands live on disk under `AGENTX_HOME` (default `./agentx.d`) and are loaded at startup — add, change, or remove them and **restart**; no recompile, no rebuild.

```text
agentx.d/
  config.yaml        agent behaviour (system prompt, iteration cap)
  skills/*.md        prompt fragments appended to the system prompt
  commands/*.rhai    scripted tools the model can call
```

Everything is optional: a missing dir or file just falls back to built-in defaults.

### Config — `config.yaml`

```yaml
system_prompt: |
  You are AgentX, an expert software engineering assistant.
max_iterations: 32
```

Deployment settings (gateway URL, keys, storage) stay as environment variables — only agent *behaviour* lives here.

### Skills — `skills/*.md`

Each Markdown file is appended to the system prompt (sorted by filename) under a `# Skills` heading. Use them to teach the agent project conventions, workflows, or domain knowledge — just drop in a `.md` file.

### Commands — `commands/*.rhai`

Each [Rhai](https://rhai.rs) script becomes a tool named after the file. This is the part that genuinely needs a scripting engine: dynamic *behaviour* that would otherwise require a recompile.

```rust
// commands/word_count.rhai → tool "word_count"
fn meta() {
    #{ description: "Count the words in a workspace file.",
       parameters: #{ type: "object",
                      properties: #{ path: #{ type: "string" } },
                      required: ["path"] } }
}
fn run(input) {
    let text = read_file(input.path);
    "" + text.split(' ').len + " words in " + input.path
}
```

- `run(input)` — entry point; `input` is a map of the model's arguments, the return value (stringified) is the tool result. **Required.**
- `meta()` — returns `#{ description, parameters }` describing the tool to the model. Optional.
- Host functions, all scoped to the workspace dir: `read_file(path)`, `write_file(path, content)`, `list_files(path)`, `sh(cmd)`.

A script that fails to compile is logged and skipped — one broken command never takes the agent down.

---

## Extending AgentX

### Adding a Custom Tool

```rust
use agentx::tools::Tool;
use async_trait::async_trait;
use serde_json::Value;

pub struct WebSearchTool { /* ... */ }

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str { "web_search" }
    fn description(&self) -> &str { "Search the web for current information." }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"]
        })
    }
    async fn execute(&self, input: Value) -> agentx::errors::Result<String> {
        let query = input["query"].as_str().unwrap_or("");
        // Your implementation here
        Ok(format!("Results for: {query}"))
    }
}

// Register it:
registry.register(WebSearchTool { /* ... */ });
```

### Adding a Custom Hook

```rust
use agentx::hooks::{Hook, HookEvent, HookResult};
use async_trait::async_trait;

pub struct MetricsHook;

#[async_trait]
impl Hook for MetricsHook {
    async fn on_event(&self, event: &HookEvent) -> HookResult {
        match event {
            HookEvent::AfterInference { content, tool_call_count } => {
                // Emit metrics to Datadog, Prometheus, etc.
            }
            _ => {}
        }
        HookResult::Continue
    }
}
```

### Swapping the Storage Backend

Implement `agentx::storage::Storage` and pass it to both the tool registry and the agent:

```rust
let my_storage: Arc<dyn Storage> = Arc::new(MyRedisStorage::new());
let registry = default_registry(Arc::clone(&my_storage));
let agent = Agent::new(llm, registry, hooks, my_storage, config);
```

---

## Running Tests

```bash
cargo test                   # all 14 integration tests (no network needed)
cargo test -- --nocapture    # with stdout output
```

Tests use `tempfile` for ephemeral storage — no LLM or Postgres required.

---

## Project Structure

```
agentx/
├── src/
│   ├── main.rs          Entry point — server or CLI mode
│   ├── lib.rs           Public crate API for integration tests
│   ├── agent.rs         Core agentic loop (channel-driven, headless)
│   ├── api.rs           Axum HTTP server (WebSocket + SSE + REST)
│   ├── llm.rs           LiteLLM gateway client
│   ├── tools.rs         Tool trait + registry + 5 built-in tools
│   ├── mcp.rs           MCP (Model Context Protocol) stdio client
│   ├── scripting.rs     Dynamic extensions — YAML config, MD skills, Rhai commands
│   ├── code_tool.rs     Sandboxed Python execution (Monty)
│   ├── hooks.rs         Hook system (logging, confirmation, extensible)
│   ├── storage.rs       Storage trait + Filesystem + Postgres backends
│   ├── config.rs        Environment-based configuration
│   └── errors.rs        Unified error types
├── tests/
│   ├── integration_test.rs   Core agent + tool tests
│   ├── api_test.rs           HTTP API tests
│   ├── code_tool_test.rs     Sandbox execution tests
│   ├── postgres_test.rs      Postgres backend tests
│   └── hooks_test.rs         Hook system tests
├── frontend/
│   ├── src/main.js      WebSocket client + chat UI
│   ├── src/style.css    Dark-mode styles
│   ├── index.html       SPA shell
│   ├── vite.config.js   Dev server with API proxy
│   ├── Dockerfile       Multi-stage: node build → nginx serve
│   └── nginx.conf       Reverse proxy to agent API
├── Cargo.toml           Rust dependencies
├── Dockerfile           Multi-stage Rust build
├── docker-compose.yml   Full stack orchestration
├── litellm_config.yaml  Model routing configuration
└── .env.example         Environment template
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Rust over Python** | Zero-cost abstractions, fearless concurrency, single static binary. No GIL bottleneck for concurrent sessions. |
| **LiteLLM gateway** | Unified OpenAI-compatible interface to all providers. Retry, load-balance, and swap models without agent code changes. |
| **Channel-driven agent** | The agentic loop is decoupled from I/O. Same loop serves CLI, WebSocket, and SSE — no duplication. |
| **Trait-based extensibility** | `Tool`, `Hook`, and `Storage` are all traits. Swap implementations without touching the core. |
| **Monty sandbox** | Lets the agent compose multi-step operations as code, not sequential tool calls. Safer than spawning subprocesses. |
| **No SDK dependency** | LLM communication is plain HTTP + JSON. Keeps the dependency tree small and debuggable. |

---

## Roadmap

- [x] Streaming token output (SSE chunked)
- [x] MCP (Model Context Protocol) tool server support (stdio + HTTP + SSE transports)
- [x] Dynamic extensions — YAML config, Markdown skills, Rhai scripted commands (no recompile)
- [ ] Multi-agent orchestration (supervisor + worker pattern)
- [ ] Git integration tool (diff, commit, branch)
- [ ] VS Code extension
- [ ] OpenTelemetry tracing export
- [ ] Rate limiting + usage tracking per session
- [x] Plugin system — Rhai-scripted commands loaded from disk (see [Dynamic Extensions](#dynamic-extensions))

---

## Inspired By

- [How to Build an Agent](https://ampcode.com/notes/how-to-build-an-agent) — Anthropic's guide to agentic systems
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — Anthropic's CLI agent
- [Aider](https://github.com/paul-gauthier/aider) — AI pair programming in your terminal
- [SWE-agent](https://github.com/princeton-nlp/SWE-agent) — Princeton's autonomous software engineering agent

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>If AgentX helps your workflow, consider giving it a star!</strong>
</p>
