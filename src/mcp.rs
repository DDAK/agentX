/// Model Context Protocol (MCP) client.
///
/// MCP servers expose tools over JSON-RPC 2.0. AgentX speaks two transports:
///
/// - **stdio** — launch the server as a child process and exchange
///   newline-delimited JSON-RPC over its stdin/stdout. The transport for local
///   servers (the kind Claude Desktop / Claude Code spawn).
/// - **HTTP** — the "Streamable HTTP" transport: POST each JSON-RPC message to
///   a URL; the server answers with `application/json` or a `text/event-stream`
///   SSE frame. Used for remote/hosted servers.
///
/// We implement only the slice the agent needs — `initialize`, `tools/list`,
/// and `tools/call` — directly, rather than pulling in the full `rmcp` SDK.
/// Each discovered MCP tool is wrapped in [`McpTool`], which implements the
/// existing [`Tool`] trait, so the agent loop treats it like any built-in.
///
/// Tools are namespaced `mcp__<server>__<tool>` (the Claude Code convention) so
/// two servers exposing a `search` tool don't collide.
///
/// Requests are dispatched **concurrently**: each carries a unique id, and
/// responses are routed back to the awaiting caller by id (see
/// [`StdioTransport`]'s background reader). HTTP requests are independent POSTs
/// and are naturally concurrent.
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::errors::{AgentError, Result};
use crate::tools::Tool;

/// MCP JSON-RPC requests are answered within this window or treated as failed.
/// ponytail: fixed per-request timeout; make it per-server config if a slow
/// server legitimately needs longer.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// MCP protocol version we advertise. Servers negotiate down if needed.
const PROTOCOL_VERSION: &str = "2024-11-05";

// ── configuration ──────────────────────────────────────────────────────────────

/// One MCP server entry, matching the de-facto `.mcp.json` shape. The transport
/// is chosen by which field is set (checked in this order):
///
/// - `sse_url`  → legacy HTTP+SSE transport (GET stream + POST endpoint)
/// - `url`      → Streamable HTTP transport (POST per message)
/// - `command`  → stdio transport (spawn a subprocess)
///
/// ```json
/// { "mcpServers": {
///     "everything": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-everything"] },
///     "remote":     { "url": "https://example.com/mcp", "headers": { "Authorization": "Bearer ..." } },
///     "postgres":   { "sse_url": "http://postgres-mcp:8000/sse" }
/// }}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct McpServerConfig {
    /// stdio transport: the executable to launch.
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Streamable-HTTP transport: the server endpoint URL.
    #[serde(default)]
    pub url: Option<String>,
    /// Legacy HTTP+SSE transport: the `/sse` GET-stream URL.
    #[serde(default)]
    pub sse_url: Option<String>,
    /// HTTP/SSE transport: extra headers (e.g. auth) sent with every request.
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct McpConfigFile {
    #[serde(default, rename = "mcpServers")]
    mcp_servers: HashMap<String, McpServerConfig>,
}

/// Load server definitions from `MCP_CONFIG` (a path) or a `.mcp.json` in the
/// working directory. Returns an empty map when neither exists — MCP is
/// strictly opt-in and costs nothing when unconfigured.
pub fn load_server_configs() -> HashMap<String, McpServerConfig> {
    let path = std::env::var("MCP_CONFIG").unwrap_or_else(|_| ".mcp.json".into());
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };
    match serde_json::from_str::<McpConfigFile>(&raw) {
        Ok(cfg) => cfg.mcp_servers,
        Err(e) => {
            warn!(path = %path, error = %e, "failed to parse MCP config; ignoring");
            HashMap::new()
        }
    }
}

// ── transport ──────────────────────────────────────────────────────────────────

/// A JSON-RPC transport to one MCP server. `&self` so calls run concurrently.
#[async_trait]
trait Transport: Send + Sync {
    async fn request(&self, method: &str, params: Value) -> Result<Value>;
    async fn notify(&self, method: &str, params: Value) -> Result<()>;
}

/// Extract the `result` from a parsed JSON-RPC response, or surface its error.
fn jsonrpc_result(msg: &Value, server: &str) -> Result<Value> {
    if let Some(err) = msg.get("error") {
        return Err(AgentError::Mcp(format!("'{server}': {err}")));
    }
    Ok(msg.get("result").cloned().unwrap_or(Value::Null))
}

// ── stdio transport ──────────────────────────────────────────────────────────

/// Pending requests keyed by JSON-RPC id, awaiting their response.
type Pending = Arc<Mutex<HashMap<i64, oneshot::Sender<Result<Value>>>>>;

/// Newline-delimited JSON-RPC over a child process's stdin/stdout.
///
/// A background task owns stdout and routes each response to the matching
/// caller by id, so many requests can be in flight at once. Writes share a
/// mutex (they're tiny and serialization keeps frames intact).
struct StdioTransport {
    name: String,
    stdin: Mutex<ChildStdin>,
    pending: Pending,
    next_id: AtomicI64,
    // Held so the child is killed (kill_on_drop) when the transport drops.
    _child: Child,
    reader: JoinHandle<()>,
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl StdioTransport {
    fn spawn(name: &str, cfg: &McpServerConfig) -> Result<Self> {
        let command = cfg
            .command
            .as_ref()
            .ok_or_else(|| AgentError::Mcp("stdio server needs 'command'".into()))?;

        let mut child = Command::new(command)
            .args(&cfg.args)
            .envs(&cfg.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| AgentError::Mcp(format!("failed to spawn '{command}': {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AgentError::Mcp("child stdin unavailable".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::Mcp("child stdout unavailable".into()))?;

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let reader = tokio::spawn(reader_loop(
            BufReader::new(stdout),
            Arc::clone(&pending),
            name.to_owned(),
        ));

        Ok(Self {
            name: name.to_owned(),
            stdin: Mutex::new(stdin),
            pending,
            next_id: AtomicI64::new(1),
            _child: child,
            reader,
        })
    }
}

/// Read responses off the server's stdout and hand each to its waiting caller.
/// On EOF (server exited), fail every pending request so callers don't hang.
async fn reader_loop(
    mut stdout: BufReader<tokio::process::ChildStdout>,
    pending: Pending,
    name: String,
) {
    let mut line = String::new();
    loop {
        line.clear();
        match stdout.read_line(&mut line).await {
            Ok(0) | Err(_) => break, // EOF or read error
            Ok(_) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Non-JSON log lines on stdout, and notifications (no id), are ignored.
        let Ok(msg) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        let Some(id) = msg.get("id").and_then(|v| v.as_i64()) else {
            continue;
        };
        if let Some(tx) = pending.lock().await.remove(&id) {
            let _ = tx.send(jsonrpc_result(&msg, &name));
        }
    }

    // Server closed: unblock everyone still waiting.
    let mut p = pending.lock().await;
    for (_, tx) in p.drain() {
        let _ = tx.send(Err(AgentError::Mcp(format!(
            "'{name}' closed the connection"
        ))));
    }
}

#[async_trait]
impl Transport for StdioTransport {
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let payload = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if let Err(e) = self.write(&payload).await {
            self.pending.lock().await.remove(&id);
            return Err(e);
        }

        match tokio::time::timeout(REQUEST_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(AgentError::Mcp(format!("'{}' dropped the response", self.name))),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(AgentError::Mcp(format!("'{}' timed out on {method}", self.name)))
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let payload = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        self.write(&payload).await
    }
}

impl StdioTransport {
    async fn write(&self, payload: &Value) -> Result<()> {
        let mut line = serde_json::to_string(payload)?;
        line.push('\n');
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| AgentError::Mcp(format!("'{}' write failed: {e}", self.name)))?;
        stdin
            .flush()
            .await
            .map_err(|e| AgentError::Mcp(format!("'{}' flush failed: {e}", self.name)))?;
        Ok(())
    }
}

// ── HTTP transport ─────────────────────────────────────────────────────────────

/// Streamable HTTP transport: each JSON-RPC message is an independent POST,
/// so concurrency comes for free. The server may answer with a single JSON
/// body or an SSE stream; both are handled. A `Mcp-Session-Id` returned on
/// `initialize` is echoed on every later request.
struct HttpTransport {
    name: String,
    http: reqwest::Client,
    url: String,
    headers: HashMap<String, String>,
    next_id: AtomicI64,
    session_id: Mutex<Option<String>>,
}

impl HttpTransport {
    fn new(name: &str, cfg: &McpServerConfig) -> Result<Self> {
        let url = cfg
            .url
            .clone()
            .ok_or_else(|| AgentError::Mcp("http server needs 'url'".into()))?;
        Ok(Self {
            name: name.to_owned(),
            http: reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .map_err(|e| AgentError::Mcp(format!("http client build failed: {e}")))?,
            url,
            headers: cfg.headers.clone(),
            next_id: AtomicI64::new(1),
            session_id: Mutex::new(None),
        })
    }

    /// Build a POST request with auth headers, MCP `Accept`, and session id.
    async fn post(&self, payload: &Value) -> Result<reqwest::RequestBuilder> {
        let mut req = self
            .http
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .json(payload);
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }
        if let Some(sid) = self.session_id.lock().await.as_ref() {
            req = req.header("Mcp-Session-Id", sid);
        }
        Ok(req)
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let payload = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });

        let resp = self
            .post(&payload)
            .await?
            .send()
            .await
            .map_err(|e| AgentError::Mcp(format!("'{}' POST {method} failed: {e}", self.name)))?;

        if !resp.status().is_success() {
            return Err(AgentError::Mcp(format!(
                "'{}' {method} returned HTTP {}",
                self.name,
                resp.status()
            )));
        }

        // Capture the session id handed back on initialize.
        if let Some(sid) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
            *self.session_id.lock().await = Some(sid.to_owned());
        }

        let is_sse = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.contains("text/event-stream"))
            .unwrap_or(false);

        let body = resp
            .text()
            .await
            .map_err(|e| AgentError::Mcp(format!("'{}' read body failed: {e}", self.name)))?;

        self.extract(&body, is_sse, id)
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let payload = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let resp = self
            .post(&payload)
            .await?
            .send()
            .await
            .map_err(|e| AgentError::Mcp(format!("'{}' notify {method} failed: {e}", self.name)))?;
        if !resp.status().is_success() {
            return Err(AgentError::Mcp(format!(
                "'{}' notify {method} returned HTTP {}",
                self.name,
                resp.status()
            )));
        }
        Ok(())
    }
}

impl HttpTransport {
    /// Pull the JSON-RPC response for `id` out of a JSON or SSE body.
    fn extract(&self, body: &str, is_sse: bool, id: i64) -> Result<Value> {
        if is_sse {
            for line in body.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let Ok(msg) = serde_json::from_str::<Value>(data.trim()) else {
                    continue;
                };
                if msg.get("id").and_then(|v| v.as_i64()) == Some(id) {
                    return jsonrpc_result(&msg, &self.name);
                }
            }
            Err(AgentError::Mcp(format!(
                "'{}' returned no matching response in SSE stream",
                self.name
            )))
        } else {
            let msg: Value = serde_json::from_str(body)
                .map_err(|e| AgentError::Mcp(format!("'{}' bad JSON response: {e}", self.name)))?;
            jsonrpc_result(&msg, &self.name)
        }
    }
}

// ── SSE transport (legacy HTTP+SSE) ──────────────────────────────────────────

/// The original MCP HTTP transport, which postgres-mcp and many existing
/// servers speak. Two endpoints:
///
/// 1. `GET <sse_url>` — a long-lived SSE stream. Its first frame is an
///    `endpoint` event carring a session-scoped POST path; every later frame
///    is a `message` event holding a JSON-RPC response.
/// 2. `POST <endpoint>` — where the client sends JSON-RPC requests (the body
///    is accepted with `202`; the actual response comes back on the stream).
///
/// A background task owns the GET stream and routes responses to callers by id
/// via the shared [`Pending`] table — identical concurrency model to stdio.
struct SseTransport {
    name: String,
    http: reqwest::Client,
    /// Absolute URL to POST requests to (origin of `sse_url` + endpoint path).
    post_url: String,
    headers: HashMap<String, String>,
    pending: Pending,
    next_id: AtomicI64,
    reader: JoinHandle<()>,
}

impl Drop for SseTransport {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl SseTransport {
    async fn connect(name: &str, cfg: &McpServerConfig) -> Result<Self> {
        let sse_url = cfg
            .sse_url
            .clone()
            .ok_or_else(|| AgentError::Mcp("sse server needs 'sse_url'".into()))?;
        let http = reqwest::Client::builder()
            .build()
            .map_err(|e| AgentError::Mcp(format!("http client build failed: {e}")))?;

        // Open the GET stream.
        let mut req = http.get(&sse_url).header("Accept", "text/event-stream");
        for (k, v) in &cfg.headers {
            req = req.header(k, v);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| AgentError::Mcp(format!("'{name}' GET {sse_url} failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(AgentError::Mcp(format!(
                "'{name}' SSE stream returned HTTP {}",
                resp.status()
            )));
        }

        // The endpoint path is relative; resolve it against the sse_url origin.
        let origin = url_origin(&sse_url);
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let (endpoint_tx, endpoint_rx) = oneshot::channel::<String>();
        let reader = tokio::spawn(sse_reader_loop(
            resp,
            Arc::clone(&pending),
            name.to_owned(),
            endpoint_tx,
        ));

        // Wait for the endpoint event (the server sends it immediately).
        let endpoint_path = tokio::time::timeout(REQUEST_TIMEOUT, endpoint_rx)
            .await
            .map_err(|_| AgentError::Mcp(format!("'{name}' sent no endpoint event")))?
            .map_err(|_| AgentError::Mcp(format!("'{name}' stream closed before endpoint")))?;

        Ok(Self {
            name: name.to_owned(),
            http,
            post_url: format!("{origin}{endpoint_path}"),
            headers: cfg.headers.clone(),
            pending,
            next_id: AtomicI64::new(1),
            reader,
        })
    }

    async fn post(&self, payload: &Value) -> Result<()> {
        let mut req = self.http.post(&self.post_url).json(payload);
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| AgentError::Mcp(format!("'{}' POST failed: {e}", self.name)))?;
        if !resp.status().is_success() {
            return Err(AgentError::Mcp(format!(
                "'{}' POST returned HTTP {}",
                self.name,
                resp.status()
            )));
        }
        Ok(())
    }
}

/// Read the SSE GET stream: deliver the `endpoint` event once, then route each
/// `message` event's JSON-RPC response to its waiting caller by id.
async fn sse_reader_loop(
    resp: reqwest::Response,
    pending: Pending,
    name: String,
    endpoint_tx: oneshot::Sender<String>,
) {
    use futures_util::StreamExt;

    let mut endpoint_tx = Some(endpoint_tx);
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else { break };
        // Normalize CRLF → LF so frame/line splitting is uniform (the SSE spec
        // allows \r\n, \n, or \r line endings; postgres-mcp uses \r\n).
        buf.push_str(&String::from_utf8_lossy(&chunk).replace("\r\n", "\n").replace('\r', "\n"));

        // Process complete SSE events (separated by a blank line).
        while let Some(pos) = buf.find("\n\n") {
            let event: String = buf.drain(..pos + 2).collect();
            let mut ev_type = "message";
            let mut data = String::new();
            for line in event.lines() {
                if let Some(rest) = line.strip_prefix("event:") {
                    ev_type = rest.trim();
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data = rest.trim().to_owned();
                }
            }
            if data.is_empty() {
                continue;
            }
            if ev_type == "endpoint" {
                if let Some(tx) = endpoint_tx.take() {
                    let _ = tx.send(data);
                }
                continue;
            }
            // message event → a JSON-RPC response routed by id.
            let Ok(msg) = serde_json::from_str::<Value>(&data) else {
                continue;
            };
            let Some(id) = msg.get("id").and_then(|v| v.as_i64()) else {
                continue;
            };
            if let Some(tx) = pending.lock().await.remove(&id) {
                let _ = tx.send(jsonrpc_result(&msg, &name));
            }
        }
    }

    // Stream closed: unblock anyone still waiting.
    let mut p = pending.lock().await;
    for (_, tx) in p.drain() {
        let _ = tx.send(Err(AgentError::Mcp(format!("'{name}' closed the SSE stream"))));
    }
}

/// `scheme://host:port` of a URL, for resolving the relative endpoint path.
/// ponytail: string slice rather than a url crate — we only need the origin.
fn url_origin(url: &str) -> String {
    // Find the end of "scheme://", then the next '/' starting the path.
    let after_scheme = url.find("://").map(|i| i + 3).unwrap_or(0);
    match url[after_scheme..].find('/') {
        Some(rel) => url[..after_scheme + rel].to_owned(),
        None => url.to_owned(),
    }
}

#[async_trait]
impl Transport for SseTransport {
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let payload = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if let Err(e) = self.post(&payload).await {
            self.pending.lock().await.remove(&id);
            return Err(e);
        }

        match tokio::time::timeout(REQUEST_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(AgentError::Mcp(format!("'{}' dropped the response", self.name))),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(AgentError::Mcp(format!("'{}' timed out on {method}", self.name)))
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let payload = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        self.post(&payload).await
    }
}

// ── client ───────────────────────────────────────────────────────────────────

/// A live connection to one MCP server over some [`Transport`].
pub struct McpClient {
    name: String,
    transport: Box<dyn Transport>,
}

impl McpClient {
    /// Connect over the configured transport, then perform the MCP handshake.
    /// Transport is chosen from the config: `sse_url` → SSE, `url` → HTTP,
    /// otherwise `command` → stdio.
    pub async fn connect(name: &str, cfg: &McpServerConfig) -> Result<Arc<Self>> {
        let transport: Box<dyn Transport> = if cfg.sse_url.is_some() {
            Box::new(SseTransport::connect(name, cfg).await?)
        } else if cfg.url.is_some() {
            Box::new(HttpTransport::new(name, cfg)?)
        } else if cfg.command.is_some() {
            Box::new(StdioTransport::spawn(name, cfg)?)
        } else {
            return Err(AgentError::Mcp(format!(
                "server '{name}' needs 'command' (stdio), 'url' (http), or 'sse_url' (sse)"
            )));
        };

        let client = Arc::new(Self {
            name: name.to_owned(),
            transport,
        });
        client.handshake().await?;
        Ok(client)
    }

    /// `initialize` request followed by the `initialized` notification.
    async fn handshake(&self) -> Result<()> {
        self.transport
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": { "name": "agentx", "version": env!("CARGO_PKG_VERSION") }
                }),
            )
            .await?;
        self.transport
            .notify("notifications/initialized", json!({}))
            .await
    }

    /// Discover the server's tools and wrap each as an agent [`Tool`].
    pub async fn list_tools(self: &Arc<Self>) -> Result<Vec<Arc<dyn Tool>>> {
        let result = self.transport.request("tools/list", json!({})).await?;
        let Some(raw) = result.get("tools").and_then(|t| t.as_array()) else {
            return Ok(vec![]);
        };

        let tools = raw
            .iter()
            .filter_map(|t| {
                let name = t.get("name")?.as_str()?.to_owned();
                let description = t
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_owned();
                // MCP uses camelCase `inputSchema`; default to a permissive object.
                let schema = t
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object" }));
                Some(Arc::new(McpTool {
                    client: Arc::clone(self),
                    remote_name: name.clone(),
                    qualified_name: format!("mcp__{}__{}", self.name, name),
                    description,
                    schema,
                }) as Arc<dyn Tool>)
            })
            .collect();
        Ok(tools)
    }

    /// Invoke a tool on the server and return its text content.
    async fn call_tool(&self, remote_name: &str, arguments: Value) -> Result<String> {
        let result = self
            .transport
            .request(
                "tools/call",
                json!({ "name": remote_name, "arguments": arguments }),
            )
            .await?;

        // Result content is an array of typed parts; concatenate the text ones.
        let text = result
            .get("content")
            .and_then(|c| c.as_array())
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();

        // Servers signal tool-level failure with `isError: true`.
        if result.get("isError").and_then(|e| e.as_bool()) == Some(true) {
            return Err(AgentError::ToolExecution {
                tool: remote_name.into(),
                reason: if text.is_empty() { "tool reported an error".into() } else { text },
            });
        }
        Ok(text)
    }
}

// ── tool wrapper ───────────────────────────────────────────────────────────────

/// An MCP server tool exposed to the agent as a native [`Tool`].
struct McpTool {
    client: Arc<McpClient>,
    /// The unqualified name the server expects in `tools/call`.
    remote_name: String,
    /// The namespaced name the model sees (`mcp__<server>__<tool>`).
    qualified_name: String,
    description: String,
    schema: Value,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.qualified_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        self.client.call_tool(&self.remote_name, input).await
    }
}

// ── bootstrap helper ─────────────────────────────────────────────────────────

/// Connect every configured MCP server and return all their tools.
///
/// A server that fails to start is logged and skipped — one broken server
/// must not take down the agent.
pub async fn connect_all() -> Vec<Arc<dyn Tool>> {
    let configs = load_server_configs();
    if configs.is_empty() {
        return vec![];
    }

    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    for (name, cfg) in &configs {
        match McpClient::connect(name, cfg).await {
            Ok(client) => match client.list_tools().await {
                Ok(mut t) => {
                    info!(server = %name, tools = t.len(), "MCP server connected");
                    tools.append(&mut t);
                }
                Err(e) => warn!(server = %name, error = %e, "MCP tools/list failed; skipping"),
            },
            Err(e) => warn!(server = %name, error = %e, "MCP connect failed; skipping"),
        }
    }
    debug!(total = tools.len(), "MCP tools registered");
    tools
}
