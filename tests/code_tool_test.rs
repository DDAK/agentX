/// Integration tests for `RunCodeTool` — the Monty-sandboxed Python executor.
///
/// These tests exercise the full sandbox dispatch loop without network or LLM
/// calls. A real `FilesystemStorage` (ephemeral `TempDir`) backs the host
/// tool functions so file operations are genuine.
///
/// Run with:  cargo test --test code_tool_test
use std::path::Path;
use std::sync::Arc;

use agentx::code_tool::RunCodeTool;
use agentx::errors::Result;
use agentx::storage::{FilesystemStorage, Storage};
use agentx::tools::{default_registry, default_registry_with_mcp, Tool, ToolRegistry};
use async_trait::async_trait;
use serde_json::{json, Value};
use tempfile::TempDir;

// ── helpers ───────────────────────────────────────────────────────────────────

async fn setup() -> (TempDir, Arc<dyn Storage>, Arc<ToolRegistry>) {
    let dir = TempDir::new().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(
        FilesystemStorage::new(dir.path()).await.unwrap(),
    );
    let registry = default_registry(Arc::clone(&storage));
    (dir, storage, registry)
}

/// Run a Python script through `RunCodeTool` and return the output string.
async fn run(registry: Arc<ToolRegistry>, code: &str) -> String {
    let tool = RunCodeTool::new(registry);
    tool.execute(json!({ "code": code })).await.expect("tool execution failed")
}

/// Run a Python script and expect an error.
async fn run_err(registry: Arc<ToolRegistry>, code: &str) -> String {
    let tool = RunCodeTool::new(registry);
    tool.execute(json!({ "code": code })).await.unwrap_err().to_string()
}

// ── basic execution ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_simple_expression() {
    let (_dir, _storage, registry) = setup().await;
    let output = run(registry, "1 + 1").await;
    assert_eq!(output, "2");
}

#[tokio::test]
async fn test_print_captured() {
    let (_dir, _storage, registry) = setup().await;
    let output = run(registry, r#"print("hello from sandbox")"#).await;
    assert_eq!(output, "hello from sandbox");
}

#[tokio::test]
async fn test_print_and_return_value() {
    let (_dir, _storage, registry) = setup().await;
    let output = run(registry, r#"
print("line one")
42
"#).await;
    assert!(output.contains("line one"));
    assert!(output.contains("return value: 42"));
}

#[tokio::test]
async fn test_no_output_returns_placeholder() {
    let (_dir, _storage, registry) = setup().await;
    let output = run(registry, "x = 1").await;
    assert_eq!(output, "(no output)");
}

#[tokio::test]
async fn test_syntax_error_returns_error() {
    let (_dir, _storage, registry) = setup().await;
    let err = run_err(registry, "def (broken syntax").await;
    assert!(err.contains("syntax error") || err.contains("SyntaxError"),
        "expected syntax error, got: {err}");
}

// ── host function: read_file ──────────────────────────────────────────────────

#[tokio::test]
async fn test_sandbox_read_file() {
    let (_dir, storage, registry) = setup().await;
    storage.write_file(Path::new("hello.txt"), "hello from disk").await.unwrap();

    let output = run(registry, r#"
content = await read_file("hello.txt")
content
"#).await;
    assert_eq!(output, "hello from disk");
}

// ── host function: write_file ─────────────────────────────────────────────────

#[tokio::test]
async fn test_sandbox_write_file() {
    let (_dir, storage, registry) = setup().await;

    run(registry, r#"await write_file("out.txt", "written by sandbox")"#).await;

    let content = storage.read_file(Path::new("out.txt")).await.unwrap();
    assert_eq!(content, "written by sandbox");
}

// ── host function: edit_file ──────────────────────────────────────────────────

#[tokio::test]
async fn test_sandbox_edit_file() {
    let (_dir, storage, registry) = setup().await;
    storage.write_file(Path::new("edit_me.txt"), "foo bar baz").await.unwrap();

    let output = run(registry, r#"
result = await edit_file("edit_me.txt", "bar", "qux")
result
"#).await;
    assert_eq!(output, "OK");

    let content = storage.read_file(Path::new("edit_me.txt")).await.unwrap();
    assert_eq!(content, "foo qux baz");
}

// ── host function: list_files ─────────────────────────────────────────────────

#[tokio::test]
async fn test_sandbox_list_files() {
    let (_dir, storage, registry) = setup().await;
    storage.write_file(Path::new("a.rs"), "fn main(){}").await.unwrap();
    storage.write_file(Path::new("b.rs"), "fn foo(){}").await.unwrap();

    let output = run(registry, r#"
files = await list_files(".")
files
"#).await;
    // list returns a JSON array string from the tool; the sandbox gets it as str
    assert!(output.contains("a.rs"));
    assert!(output.contains("b.rs"));
}

// ── host function: run_command ────────────────────────────────────────────────

#[tokio::test]
async fn test_sandbox_run_command() {
    let (_dir, _storage, registry) = setup().await;

    let output = run(registry, r#"
result = await run_command("echo hello_from_cmd")
result
"#).await;
    assert!(output.contains("hello_from_cmd"), "got: {output}");
}

// ── multi-step orchestration ──────────────────────────────────────────────────

#[tokio::test]
async fn test_multi_step_orchestration() {
    let (_dir, storage, registry) = setup().await;
    storage.write_file(Path::new("counter.txt"), "0").await.unwrap();

    run(registry, r#"
# Read the current value, increment it, write back.
raw = await read_file("counter.txt")
count = int(raw.strip())
count += 1
await write_file("counter.txt", str(count))
"#).await;

    let content = storage.read_file(Path::new("counter.txt")).await.unwrap();
    assert_eq!(content, "1");
}

#[tokio::test]
async fn test_loop_over_files() {
    let (_dir, storage, registry) = setup().await;
    storage.write_file(Path::new("f1.txt"), "TODO: fix this").await.unwrap();
    storage.write_file(Path::new("f2.txt"), "all good").await.unwrap();
    storage.write_file(Path::new("f3.txt"), "TODO: another").await.unwrap();

    let output = run(registry, r#"
import json
raw = await list_files(".")
files = json.loads(raw)
todos = []
for f in files:
    if not f.endswith("/"):
        content = await read_file(f)
        if "TODO" in content:
            todos.append(f)
print(f"found {len(todos)} TODO files")
todos
"#).await;
    assert!(output.contains("found 2 TODO files"), "got: {output}");
    assert!(output.contains("f1.txt"), "got: {output}");
    assert!(output.contains("f3.txt"), "got: {output}");
}

// ── tool name & schema ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_tool_metadata() {
    let (_dir, _storage, registry) = setup().await;
    let tool = RunCodeTool::new(registry);

    assert_eq!(tool.name(), "run_code");
    assert!(tool.description().contains("sandbox"));
    assert!(tool.description().contains("read_file"));

    let schema = tool.parameters_schema();
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["code"].is_object());
    assert_eq!(schema["required"][0], "code");
}

// ── error handling ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_runtime_error_propagates() {
    let (_dir, _storage, registry) = setup().await;
    // Division by zero raises ZeroDivisionError inside the sandbox.
    let err = run_err(registry, "1 / 0").await;
    assert!(err.contains("ZeroDivisionError") || err.contains("division") || err.contains("runtime"),
        "expected division error, got: {err}");
}

#[tokio::test]
async fn test_missing_code_field_returns_error() {
    let (_dir, _storage, registry) = setup().await;
    let tool = RunCodeTool::new(registry);
    let err = tool.execute(json!({})).await.unwrap_err();
    assert!(err.to_string().contains("code") || err.to_string().contains("missing"),
        "got: {err}");
}

// ── MCP tools reachable from the sandbox via call_tool ────────────────────────

/// A stub tool standing in for an MCP-provided tool. Echoes its `query` arg
/// back so the test can assert the kwargs reached it.
struct StubMcpTool;

#[async_trait]
impl Tool for StubMcpTool {
    fn name(&self) -> &str {
        "mcp__demo__echo"
    }
    fn description(&self) -> &str {
        "echoes its query argument"
    }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": { "query": { "type": "string" } } })
    }
    async fn execute(&self, input: Value) -> Result<String> {
        Ok(format!("echo: {}", input.get("query").and_then(|q| q.as_str()).unwrap_or("")))
    }
}

#[tokio::test]
async fn test_sandbox_can_call_mcp_tool() {
    let dir = TempDir::new().unwrap();
    let storage: Arc<dyn Storage> =
        Arc::new(FilesystemStorage::new(dir.path()).await.unwrap());
    let extra: Vec<Arc<dyn Tool>> = vec![Arc::new(StubMcpTool)];
    let registry = default_registry_with_mcp(Arc::clone(&storage), &extra);

    // The script reaches the MCP tool through the generic call_tool host fn.
    let output = run(
        registry,
        r#"
result = await call_tool("mcp__demo__echo", query="hello from python")
result
"#,
    )
    .await;
    assert_eq!(output, "echo: hello from python");
}

#[tokio::test]
async fn test_sandbox_call_tool_unknown_tool_errors() {
    let (_dir, _storage, registry) = setup().await;
    let err = run_err(registry, r#"await call_tool("mcp__nope__missing", x=1)"#).await;
    assert!(err.contains("unknown tool") || err.contains("nope"), "got: {err}");
}
