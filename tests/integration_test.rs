/// Integration tests for AgentX.
///
/// These tests use `tempfile` for ephemeral filesystem storage and
/// `mockall` to stub the LLM client — no real network calls are made.
///
/// Run with:   cargo test
///             cargo test -- --nocapture   (to see stdout)
use std::path::Path;
use std::sync::Arc;

use agentx::storage::{FilesystemStorage, Storage};
use agentx::tools::{
    default_registry, EditFileTool, ListFilesTool, ReadFileTool, Tool, WriteFileTool,
};
use serde_json::json;
use tempfile::TempDir;

// ── helpers ───────────────────────────────────────────────────────────────────

async fn temp_storage() -> (TempDir, Arc<dyn Storage>) {
    let dir = TempDir::new().expect("failed to create temp dir");
    let storage = FilesystemStorage::new(dir.path())
        .await
        .expect("failed to create storage");
    (dir, Arc::new(storage))
}

// ── storage: filesystem ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_filesystem_write_and_read() {
    let (_dir, storage) = temp_storage().await;

    storage
        .write_file(Path::new("hello.txt"), "Hello, world!")
        .await
        .expect("write failed");

    let content = storage
        .read_file(Path::new("hello.txt"))
        .await
        .expect("read failed");

    assert_eq!(content, "Hello, world!");
}

#[tokio::test]
async fn test_filesystem_write_creates_parent_dirs() {
    let (_dir, storage) = temp_storage().await;

    storage
        .write_file(Path::new("a/b/c/deep.txt"), "deep content")
        .await
        .expect("write failed");

    let content = storage
        .read_file(Path::new("a/b/c/deep.txt"))
        .await
        .expect("read failed");

    assert_eq!(content, "deep content");
}

#[tokio::test]
async fn test_filesystem_list_files() {
    let (_dir, storage) = temp_storage().await;

    storage.write_file(Path::new("a.txt"), "a").await.unwrap();
    storage.write_file(Path::new("b.txt"), "b").await.unwrap();
    // create a sub-directory by writing a nested file
    storage
        .write_file(Path::new("subdir/c.txt"), "c")
        .await
        .unwrap();

    let entries = storage.list_files(Path::new(".")).await.unwrap();

    assert!(entries.contains(&"a.txt".to_owned()));
    assert!(entries.contains(&"b.txt".to_owned()));
    assert!(entries.contains(&"subdir/".to_owned()));
}

#[tokio::test]
async fn test_filesystem_sessions_roundtrip() {
    let (_dir, storage) = temp_storage().await;

    let mut session = storage
        .create_session(Some("test session".into()))
        .await
        .unwrap();

    session.messages.push(agentx::llm::Message::user("Hello!"));
    storage.save_session(&session).await.unwrap();

    let loaded = storage
        .load_session(session.id)
        .await
        .unwrap()
        .expect("session not found");

    assert_eq!(loaded.id, session.id);
    assert_eq!(loaded.label, Some("test session".into()));
    assert_eq!(loaded.messages.len(), 1);
}

#[tokio::test]
async fn test_load_nonexistent_session_returns_none() {
    let (_dir, storage) = temp_storage().await;
    let missing_id = uuid::Uuid::new_v4();
    let result = storage.load_session(missing_id).await.unwrap();
    assert!(result.is_none());
}

// ── tools ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_read_file_tool() {
    let (_dir, storage) = temp_storage().await;
    storage
        .write_file(Path::new("test.txt"), "tool content")
        .await
        .unwrap();

    let tool = ReadFileTool::new(Arc::clone(&storage));
    let result = tool
        .execute(json!({ "path": "test.txt" }))
        .await
        .unwrap();

    assert_eq!(result, "tool content");
}

#[tokio::test]
async fn test_write_file_tool() {
    let (_dir, storage) = temp_storage().await;

    let tool = WriteFileTool::new(Arc::clone(&storage));
    let result = tool
        .execute(json!({ "path": "out.txt", "content": "written by tool" }))
        .await
        .unwrap();

    assert!(result.contains("out.txt"));

    let content = storage.read_file(Path::new("out.txt")).await.unwrap();
    assert_eq!(content, "written by tool");
}

#[tokio::test]
async fn test_edit_file_tool_replaces_string() {
    let (_dir, storage) = temp_storage().await;
    storage
        .write_file(Path::new("src.txt"), "foo bar baz")
        .await
        .unwrap();

    let tool = EditFileTool::new(Arc::clone(&storage));
    let result = tool
        .execute(json!({
            "path": "src.txt",
            "old_str": "bar",
            "new_str": "qux"
        }))
        .await
        .unwrap();

    assert_eq!(result, "OK");

    let content = storage.read_file(Path::new("src.txt")).await.unwrap();
    assert_eq!(content, "foo qux baz");
}

#[tokio::test]
async fn test_edit_file_tool_creates_file_when_old_str_empty() {
    let (_dir, storage) = temp_storage().await;

    let tool = EditFileTool::new(Arc::clone(&storage));
    let result = tool
        .execute(json!({
            "path": "new_file.txt",
            "old_str": "",
            "new_str": "brand new content"
        }))
        .await
        .unwrap();

    assert!(result.contains("new_file.txt"));

    let content = storage.read_file(Path::new("new_file.txt")).await.unwrap();
    assert_eq!(content, "brand new content");
}

#[tokio::test]
async fn test_edit_file_tool_errors_on_missing_old_str() {
    let (_dir, storage) = temp_storage().await;
    storage
        .write_file(Path::new("f.txt"), "hello world")
        .await
        .unwrap();

    let tool = EditFileTool::new(Arc::clone(&storage));
    let err = tool
        .execute(json!({
            "path": "f.txt",
            "old_str": "not present",
            "new_str": "x"
        }))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("not found"));
}

#[tokio::test]
async fn test_edit_file_tool_errors_when_old_equals_new() {
    let (_dir, storage) = temp_storage().await;

    let tool = EditFileTool::new(Arc::clone(&storage));
    let err = tool
        .execute(json!({
            "path": "f.txt",
            "old_str": "same",
            "new_str": "same"
        }))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("must be different"));
}

#[tokio::test]
async fn test_list_files_tool() {
    let (_dir, storage) = temp_storage().await;
    storage.write_file(Path::new("x.rs"), "fn main(){}").await.unwrap();
    storage.write_file(Path::new("y.rs"), "fn foo(){}").await.unwrap();

    let tool = ListFilesTool::new(Arc::clone(&storage));
    let result = tool.execute(json!({})).await.unwrap();

    let files: Vec<String> = serde_json::from_str(&result).unwrap();
    assert!(files.contains(&"x.rs".to_owned()));
    assert!(files.contains(&"y.rs".to_owned()));
}

#[tokio::test]
async fn test_registry_dispatches_correctly() {
    let (_dir, storage) = temp_storage().await;
    storage
        .write_file(Path::new("dispatch.txt"), "dispatch content")
        .await
        .unwrap();

    let registry = default_registry(Arc::clone(&storage));

    let result = registry
        .execute("read_file", json!({ "path": "dispatch.txt" }))
        .await
        .unwrap();

    assert_eq!(result, "dispatch content");
}

#[tokio::test]
async fn test_registry_unknown_tool_returns_error() {
    let (_dir, storage) = temp_storage().await;
    let registry = default_registry(storage);

    let err = registry
        .execute("nonexistent_tool", json!({}))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("nonexistent_tool"));
}
