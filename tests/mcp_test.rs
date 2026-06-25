//! MCP client integration test.
//!
//! Spawns a tiny mock MCP server (a Python script speaking JSON-RPC over
//! stdio) and verifies the full handshake → discover → call flow, including
//! the `mcp__<server>__<tool>` namespacing and `isError` handling.

use std::collections::HashMap;
use std::io::Write;

use agentx::mcp::{McpClient, McpServerConfig};
use tempfile::NamedTempFile;

/// Minimal MCP server: answers `initialize`, `tools/list`, `tools/call`.
/// `echo` returns its `text` arg; `boom` always reports `isError`.
const MOCK_SERVER: &str = r#"
import sys, json, threading, time
_lock = threading.Lock()
def send(obj):
    with _lock:
        sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
def handle(line):
    msg = json.loads(line)
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params", {})
    if method == "initialize":
        send({"jsonrpc":"2.0","id":mid,"result":{"protocolVersion":"2024-11-05","capabilities":{}}})
    elif method == "notifications/initialized":
        pass  # notification, no reply
    elif method == "tools/list":
        send({"jsonrpc":"2.0","id":mid,"result":{"tools":[
            {"name":"echo","description":"echoes text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}},
            {"name":"slow","description":"slow echo","inputSchema":{"type":"object","properties":{"text":{"type":"string"},"delay":{"type":"number"}}}},
            {"name":"boom","description":"always fails","inputSchema":{"type":"object"}}
        ]}})
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "echo":
            send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":args.get("text","")}]}})
        elif name == "slow":
            time.sleep(float(args.get("delay", 0)))
            send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":args.get("text","")}]}})
        elif name == "boom":
            send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"kaboom"}],"isError":True}})
        else:
            send({"jsonrpc":"2.0","id":mid,"error":{"code":-32601,"message":"unknown tool"}})
    else:
        send({"jsonrpc":"2.0","id":mid,"error":{"code":-32601,"message":"unknown method"}})
# Each request handled on its own thread so responses may return out of order.
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    threading.Thread(target=handle, args=(line,), daemon=True).start()
"#;

fn mock_config() -> (NamedTempFile, McpServerConfig) {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(MOCK_SERVER.as_bytes()).unwrap();
    f.flush().unwrap();
    let cfg = McpServerConfig {
        command: Some("python3".into()),
        args: vec![f.path().to_str().unwrap().to_owned()],
        env: HashMap::new(),
        sse_url: None,
        url: None,
        headers: HashMap::new(),
    };
    (f, cfg)
}

#[tokio::test]
async fn mcp_handshake_discover_and_call() {
    let (_keep, cfg) = mock_config();

    let client = McpClient::connect("mock", &cfg)
        .await
        .expect("connect + handshake");

    let tools = client.list_tools().await.expect("tools/list");
    assert_eq!(tools.len(), 3);

    // Tools are namespaced.
    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert!(names.contains(&"mcp__mock__echo"));
    assert!(names.contains(&"mcp__mock__boom"));

    // A successful call returns the server's text content.
    let echo = tools.iter().find(|t| t.name() == "mcp__mock__echo").unwrap();
    let out = echo
        .execute(serde_json::json!({ "text": "hello mcp" }))
        .await
        .expect("echo call");
    assert_eq!(out, "hello mcp");

    // An isError result surfaces as an Err with the server's message.
    let boom = tools.iter().find(|t| t.name() == "mcp__mock__boom").unwrap();
    let err = boom.execute(serde_json::json!({})).await.unwrap_err();
    assert!(err.to_string().contains("kaboom"), "got: {err}");
}

#[tokio::test]
async fn mcp_connect_failure_is_an_error() {
    let cfg = McpServerConfig {
        command: Some("definitely-not-a-real-binary-xyz".into()),
        args: vec![],
        env: HashMap::new(),
        sse_url: None,
        url: None,
        headers: HashMap::new(),
    };
    assert!(McpClient::connect("ghost", &cfg).await.is_err());
}

#[tokio::test]
async fn mcp_concurrent_calls_resolve_by_id() {
    // A slow call and a fast call issued together: the fast one must come back
    // first, proving responses are routed by id rather than head-of-line
    // blocking on the slow request.
    let (_keep, cfg) = mock_config();
    let client = McpClient::connect("mock", &cfg).await.expect("connect");
    let tools = client.list_tools().await.expect("list");

    let slow = tools.iter().find(|t| t.name() == "mcp__mock__slow").unwrap().clone();
    let fast = tools.iter().find(|t| t.name() == "mcp__mock__echo").unwrap().clone();

    let order = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<&'static str>::new()));
    let o1 = order.clone();
    let o2 = order.clone();

    let slow_task = tokio::spawn(async move {
        let r = slow.execute(serde_json::json!({ "text": "slow", "delay": 0.5 })).await.unwrap();
        o1.lock().await.push("slow");
        r
    });
    let fast_task = tokio::spawn(async move {
        let r = fast.execute(serde_json::json!({ "text": "fast" })).await.unwrap();
        o2.lock().await.push("fast");
        r
    });

    assert_eq!(slow_task.await.unwrap(), "slow");
    assert_eq!(fast_task.await.unwrap(), "fast");
    // Fast completed before slow despite being issued second.
    assert_eq!(*order.lock().await, vec!["fast", "slow"]);
}

// ── HTTP transport ──────────────────────────────────────────────────────────

/// Mock Streamable-HTTP MCP server: one axum handler answering the three
/// methods with a single JSON body.
async fn spawn_http_mock() -> String {
    use axum::{routing::post, Json, Router};
    use serde_json::{json, Value};

    async fn handle(Json(req): Json<Value>) -> Json<Value> {
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let result = match method {
            "initialize" => json!({ "protocolVersion": "2024-11-05", "capabilities": {} }),
            "tools/list" => json!({ "tools": [
                { "name": "ping", "description": "pong", "inputSchema": { "type": "object" } }
            ] }),
            "tools/call" => json!({ "content": [ { "type": "text", "text": "pong" } ] }),
            _ => json!({}),
        };
        Json(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
    }

    let app = Router::new().route("/mcp", post(handle));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}/mcp")
}

#[tokio::test]
async fn mcp_http_transport_handshake_and_call() {
    let url = spawn_http_mock().await;
    let cfg = McpServerConfig {
        command: None,
        args: vec![],
        env: HashMap::new(),
        sse_url: None,
        url: Some(url),
        headers: HashMap::new(),
    };

    let client = McpClient::connect("remote", &cfg).await.expect("http connect");
    let tools = client.list_tools().await.expect("http list");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "mcp__remote__ping");

    let out = tools[0].execute(serde_json::json!({})).await.expect("http call");
    assert_eq!(out, "pong");
}
