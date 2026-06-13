/// Postgres storage integration tests.
///
/// These tests require a running Postgres instance.  They are gated behind
/// the `POSTGRES_TEST_URL` environment variable — if it is not set, every
/// test is skipped gracefully so `cargo test` keeps passing in CI / local
/// environments without Postgres.
///
/// Start a local instance before running:
///
///   docker run --rm -p 5433:5432 \
///     -e POSTGRES_USER=agentx \
///     -e POSTGRES_PASSWORD=agentx_dev \
///     -e POSTGRES_DB=agentx \
///     postgres:16-alpine
///
///   POSTGRES_TEST_URL=postgres://agentx:agentx_dev@localhost:5433/agentx \
///     cargo test --test postgres_test
///
/// The tests use port 5433 to avoid colliding with any existing local
/// Postgres on 5432.
use std::path::Path;
use std::sync::Arc;

use agentx::storage::{PostgresStorage, Storage};
use tempfile::TempDir;
use tokio::sync::Mutex;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Global mutex so parallel tests don't race on the DATABASE_URL env var.
static ENV_LOCK: Mutex<()> = Mutex::const_new(());

/// Returns the test DB URL, or `None` if the var is unset (skip the test).
fn test_db_url() -> Option<String> {
    std::env::var("POSTGRES_TEST_URL").ok()
}

/// Build a `PostgresStorage` pointed at `POSTGRES_TEST_URL`.
/// Returns `None` if the env var is absent (caller should skip the test).
async fn pg_storage() -> Option<(TempDir, Arc<dyn Storage>)> {
    let url = test_db_url()?;
    let dir = TempDir::new().unwrap();
    // Serialize env mutation across parallel tests.
    let _guard = ENV_LOCK.lock().await;
    std::env::set_var("DATABASE_URL", &url);
    let storage = PostgresStorage::from_env(dir.path())
        .await
        .expect("failed to connect to test Postgres — is POSTGRES_TEST_URL reachable?");
    Some((dir, Arc::new(storage) as Arc<dyn Storage>))
}

/// Generate a unique table-safe prefix so parallel test runs don't clash.
fn unique_label(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4())
}

// ── session CRUD ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn pg_test_create_and_load_session() {
    let Some((_dir, storage)) = pg_storage().await else { return };

    let label = unique_label("pg-create");
    let session = storage
        .create_session(Some(label.clone()))
        .await
        .expect("create_session failed");

    assert!(!session.id.is_nil());
    assert_eq!(session.label, Some(label.clone()));
    assert!(session.messages.is_empty());

    let loaded = storage
        .load_session(session.id)
        .await
        .expect("load_session failed")
        .expect("session not found");

    assert_eq!(loaded.id, session.id);
    assert_eq!(loaded.label, Some(label));
}

#[tokio::test]
async fn pg_test_load_nonexistent_returns_none() {
    let Some((_dir, storage)) = pg_storage().await else { return };
    let missing = uuid::Uuid::new_v4();
    let result = storage.load_session(missing).await.expect("query failed");
    assert!(result.is_none());
}

#[tokio::test]
async fn pg_test_save_and_reload_messages() {
    let Some((_dir, storage)) = pg_storage().await else { return };

    let mut session = storage
        .create_session(Some(unique_label("pg-messages")))
        .await
        .unwrap();

    session.messages.push(agentx::llm::Message::user("hello postgres"));
    session.messages.push(agentx::llm::Message::assistant("hello back"));
    storage.save_session(&session).await.expect("save failed");

    let loaded = storage
        .load_session(session.id)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(loaded.messages.len(), 2);
    // Verify roles survived the JSON round-trip.
    use agentx::llm::MessageRole;
    assert!(matches!(loaded.messages[0].role, MessageRole::User));
    assert!(matches!(loaded.messages[1].role, MessageRole::Assistant));
}

#[tokio::test]
async fn pg_test_list_sessions_includes_created() {
    let Some((_dir, storage)) = pg_storage().await else { return };

    let label_a = unique_label("pg-list-a");
    let label_b = unique_label("pg-list-b");
    let a = storage.create_session(Some(label_a.clone())).await.unwrap();
    let b = storage.create_session(Some(label_b.clone())).await.unwrap();

    let sessions = storage.list_sessions().await.expect("list failed");
    let ids: Vec<_> = sessions.iter().map(|s| s.id).collect();

    assert!(ids.contains(&a.id), "session A not in list");
    assert!(ids.contains(&b.id), "session B not in list");
}

#[tokio::test]
async fn pg_test_overwrite_session_messages() {
    let Some((_dir, storage)) = pg_storage().await else { return };

    let mut session = storage
        .create_session(Some(unique_label("pg-overwrite")))
        .await
        .unwrap();

    // First save
    session.messages.push(agentx::llm::Message::user("first"));
    storage.save_session(&session).await.unwrap();

    // Second save — replace messages
    session.messages.clear();
    session.messages.push(agentx::llm::Message::user("second"));
    session.messages.push(agentx::llm::Message::user("third"));
    storage.save_session(&session).await.unwrap();

    let loaded = storage.load_session(session.id).await.unwrap().unwrap();
    assert_eq!(loaded.messages.len(), 2);
}

// ── file ops ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn pg_test_write_and_read_file() {
    let Some((_dir, storage)) = pg_storage().await else { return };

    storage
        .write_file(Path::new("pg_test.txt"), "postgres file content")
        .await
        .expect("write failed");

    let content = storage
        .read_file(Path::new("pg_test.txt"))
        .await
        .expect("read failed");

    assert_eq!(content, "postgres file content");
}

#[tokio::test]
async fn pg_test_write_nested_file() {
    let Some((_dir, storage)) = pg_storage().await else { return };

    storage
        .write_file(Path::new("pg/nested/dir/file.txt"), "nested")
        .await
        .expect("write failed");

    let content = storage
        .read_file(Path::new("pg/nested/dir/file.txt"))
        .await
        .unwrap();

    assert_eq!(content, "nested");
}

#[tokio::test]
async fn pg_test_list_files() {
    let Some((_dir, storage)) = pg_storage().await else { return };

    storage.write_file(Path::new("pg_a.txt"), "a").await.unwrap();
    storage.write_file(Path::new("pg_b.txt"), "b").await.unwrap();

    let entries = storage.list_files(Path::new(".")).await.unwrap();
    assert!(entries.contains(&"pg_a.txt".to_owned()));
    assert!(entries.contains(&"pg_b.txt".to_owned()));
}
