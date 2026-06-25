/// API integration tests using axum's built-in test helpers.
///
/// These tests spin up the full Axum router in-process — no network port is
/// bound.  A real `FilesystemStorage` is used (ephemeral `TempDir`), but the
/// `LiteLlmClient` is pointed at a non-reachable URL so no actual LLM calls
/// are made.  We verify routing, status codes, request/response shapes, and
/// session lifecycle.
///
/// Run with:  cargo test --test api_test
use std::sync::Arc;

use agentx::agent::AgentConfig;
use agentx::api::build_router;
use agentx::config::AppConfig;
use agentx::llm::{LiteLlmClient, LiteLlmConfig};
use agentx::storage::{FilesystemStorage, Storage};
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt as _; // for `oneshot`

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a router + ephemeral storage for testing.
/// LiteLLM is pointed at 127.0.0.1:1 — guaranteed unreachable.
async fn test_app() -> (axum::Router, TempDir) {
    let dir = TempDir::new().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(
        FilesystemStorage::new(dir.path()).await.unwrap(),
    );
    let (router, _) = router_with_storage(storage, &dir).await;
    (router, dir)
}

async fn router_with_storage(
    storage: Arc<dyn Storage>,
    dir: &TempDir,
) -> (axum::Router, Arc<AppConfig>) {
    let llm_cfg = LiteLlmConfig {
        base_url:      "http://127.0.0.1:1".into(),
        api_key:       "".into(),
        default_model: "test-model".into(),
        max_tokens:    1024,
        temperature:   0.0,
    };
    let llm = Arc::new(LiteLlmClient::new(llm_cfg));
    let app_cfg = Arc::new(AppConfig {
        storage_backend:  agentx::config::StorageBackend::Filesystem,
        workspace_dir:    dir.path().to_path_buf(),
        session_label:    None,
        resume_session:   None,
        confirm_commands: false,
    });
    let router = build_router(storage, llm, Arc::clone(&app_cfg), vec![], AgentConfig::default()).await;
    (router, app_cfg)
}

/// Deserialise a response body as JSON.
async fn json_body(resp: axum::response::Response) -> Value {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

fn json_req(method: Method, uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn empty_req(method: Method, uri: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

// ── health ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_health_returns_ok() {
    let (app, _dir) = test_app().await;
    let resp = app.oneshot(empty_req(Method::GET, "/health")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["status"], "ok");
    assert!(body["version"].is_string());
}

// ── list sessions ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_list_sessions_empty() {
    let (app, _dir) = test_app().await;
    let resp = app
        .oneshot(empty_req(Method::GET, "/api/sessions"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body.as_array().unwrap().len(), 0);
}

// ── create session ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_create_session_returns_201() {
    let (app, _dir) = test_app().await;
    let resp = app
        .oneshot(json_req(
            Method::POST,
            "/api/sessions",
            serde_json::json!({ "label": "my session" }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = json_body(resp).await;
    assert!(body["id"].is_string());
    assert_eq!(body["label"], "my session");
    assert_eq!(body["message_count"], 0);
}

#[tokio::test]
async fn test_create_session_null_label() {
    let (app, _dir) = test_app().await;
    let resp = app
        .oneshot(json_req(
            Method::POST,
            "/api/sessions",
            serde_json::json!({ "label": null }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = json_body(resp).await;
    assert!(body["label"].is_null());
}

// ── get session ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_get_session_after_create() {
    let dir = TempDir::new().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(
        FilesystemStorage::new(dir.path()).await.unwrap(),
    );

    let session = storage.create_session(Some("direct".into())).await.unwrap();
    let id = session.id;

    let (app, _cfg) = router_with_storage(Arc::clone(&storage), &dir).await;

    let resp = app
        .oneshot(empty_req(Method::GET, &format!("/api/sessions/{id}")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["id"], id.to_string());
    assert_eq!(body["label"], "direct");
}

#[tokio::test]
async fn test_get_session_not_found_returns_404() {
    let (app, _dir) = test_app().await;
    let fake_id = uuid::Uuid::new_v4();
    let resp = app
        .oneshot(empty_req(Method::GET, &format!("/api/sessions/{fake_id}")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = json_body(resp).await;
    assert!(body["error"].as_str().unwrap().contains("not found"));
}

// ── list sessions after create ────────────────────────────────────────────────

#[tokio::test]
async fn test_list_sessions_after_create() {
    let dir = TempDir::new().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(
        FilesystemStorage::new(dir.path()).await.unwrap(),
    );
    storage.create_session(Some("a".into())).await.unwrap();
    storage.create_session(Some("b".into())).await.unwrap();

    let (app, _cfg) = router_with_storage(Arc::clone(&storage), &dir).await;

    let resp = app
        .oneshot(empty_req(Method::GET, "/api/sessions"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body.as_array().unwrap().len(), 2);
}

// ── post message (no active SSE) ──────────────────────────────────────────────

#[tokio::test]
async fn test_post_message_without_sse_returns_404() {
    let (app, _dir) = test_app().await;
    let fake_id = uuid::Uuid::new_v4();
    let resp = app
        .oneshot(json_req(
            Method::POST,
            &format!("/api/sessions/{fake_id}/message"),
            serde_json::json!({ "text": "hello" }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_post_message_empty_text_returns_400() {
    let (app, _dir) = test_app().await;
    let fake_id = uuid::Uuid::new_v4();
    let resp = app
        .oneshot(json_req(
            Method::POST,
            &format!("/api/sessions/{fake_id}/message"),
            serde_json::json!({ "text": "   " }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = json_body(resp).await;
    assert!(body["error"].as_str().unwrap().contains("text field"));
}

// ── unknown route ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_unknown_route_returns_404() {
    let (app, _dir) = test_app().await;
    let resp = app
        .oneshot(empty_req(Method::GET, "/nonexistent"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
