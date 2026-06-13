/// HTTP API server — WebSocket + SSE + REST session management.
///
/// Routes
/// ──────
///   GET  /health                     liveness probe
///   GET  /api/sessions               list all sessions
///   POST /api/sessions               create a new session
///   GET  /api/sessions/:id           get a session (with full message history)
///   GET  /api/sessions/:id/ws        WebSocket  — bidirectional chat
///   GET  /api/sessions/:id/sse       SSE        — receive agent events (stream)
///   POST /api/sessions/:id/message   send a message (companion to SSE)
///
/// WebSocket protocol
/// ──────────────────
///   Client → Server:  `{ "text": "your message" }`
///   Server → Client:  `AgentEvent` serialised as JSON  (see agent.rs)
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, Mutex, OnceCell};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as TokioStreamExt; // for .map() on tokio streams
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::agent::{Agent, AgentConfig, AgentEvent};
use crate::config::AppConfig;
use crate::errors::AgentError;
use crate::hooks::{HookChain, LoggingHook, ToolAnnouncerHook};
use crate::llm::LiteLlmClient;
use crate::storage::{Session, Storage};
use crate::tools::default_registry;

// ── shared application state ──────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub storage: Arc<dyn Storage>,
    pub llm:     Arc<LiteLlmClient>,
    #[allow(dead_code)] // reserved for per-request config (rate limits, etc.)
    pub app_cfg: Arc<AppConfig>,
}

// ── SSE sender registry (process-global) ─────────────────────────────────────
// Maps session UUID → the mpsc::Sender that feeds messages into the agent loop.

static SSE_SENDERS: OnceCell<Mutex<HashMap<Uuid, mpsc::Sender<String>>>> =
    OnceCell::const_new();

async fn sse_senders() -> &'static Mutex<HashMap<Uuid, mpsc::Sender<String>>> {
    SSE_SENDERS
        .get_or_init(|| async { Mutex::new(HashMap::new()) })
        .await
}

// ── router ────────────────────────────────────────────────────────────────────

pub async fn build_router(
    storage: Arc<dyn Storage>,
    llm:     Arc<LiteLlmClient>,
    app_cfg: Arc<AppConfig>,
) -> Router {
    let state = AppState { storage, llm, app_cfg };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    Router::new()
        .route("/health",                    get(health))
        .route("/api/sessions",              get(list_sessions).post(create_session))
        .route("/api/sessions/{id}",          get(get_session))
        .route("/api/sessions/{id}/ws",       get(ws_handler))
        .route("/api/sessions/{id}/sse",      get(sse_handler))
        .route("/api/sessions/{id}/message",  post(post_message))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// ── health ────────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

// ── session REST ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SessionSummary {
    id:            Uuid,
    label:         Option<String>,
    created_at:    chrono::DateTime<chrono::Utc>,
    updated_at:    chrono::DateTime<chrono::Utc>,
    message_count: usize,
}

impl From<&Session> for SessionSummary {
    fn from(s: &Session) -> Self {
        Self {
            id:            s.id,
            label:         s.label.clone(),
            created_at:    s.created_at,
            updated_at:    s.updated_at,
            message_count: s.messages.len(),
        }
    }
}

async fn list_sessions(State(state): State<AppState>) -> impl IntoResponse {
    match state.storage.list_sessions().await {
        Ok(sessions) => Json(
            sessions.iter().map(SessionSummary::from).collect::<Vec<_>>()
        ).into_response(),
        Err(e) => api_error(e),
    }
}

#[derive(Deserialize)]
struct CreateSessionBody {
    label: Option<String>,
}

async fn create_session(
    State(state): State<AppState>,
    Json(body): Json<CreateSessionBody>,
) -> impl IntoResponse {
    match state.storage.create_session(body.label).await {
        Ok(s) => (StatusCode::CREATED, Json(SessionSummary::from(&s))).into_response(),
        Err(e) => api_error(e),
    }
}

async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.storage.load_session(id).await {
        Ok(Some(s)) => Json(s).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "session not found" })),
        )
            .into_response(),
        Err(e) => api_error(e),
    }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

async fn ws_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state, id))
}

async fn handle_ws(mut socket: WebSocket, state: AppState, session_id: Uuid) {
    info!(session = %session_id, "WebSocket connected");

    let mut session = match load_or_create(&state.storage, session_id).await {
        Ok(s) => s,
        Err(e) => {
            error!("session error: {e}");
            return;
        }
    };

    // user text → agent
    let (tx_user, mut rx_user) = mpsc::channel::<String>(32);
    // agent events → broadcast
    let (tx_events, _) = broadcast::channel::<AgentEvent>(256);
    // JSON strings to send over the socket
    let (tx_json, mut rx_json) = mpsc::channel::<String>(256);

    // Relay broadcast → json channel
    let tx_events_clone = tx_events.clone();
    let mut rx_events   = tx_events.subscribe();
    let relay = tokio::spawn(async move {
        while let Ok(event) = rx_events.recv().await {
            let json = serde_json::to_string(&event).unwrap_or_default();
            if tx_json.send(json).await.is_err() {
                break;
            }
        }
    });

    // Agent loop
    let agent = make_agent(&state);
    let agent_task = tokio::spawn(async move {
        agent.run(&mut session, &mut rx_user, &tx_events_clone).await;
    });

    // Drive the socket: multiplex outgoing JSON and incoming user messages.
    loop {
        tokio::select! {
            Some(json) = rx_json.recv() => {
                if socket.send(WsMessage::Text(json.into())).await.is_err() {
                    break;
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(WsMessage::Text(text))) => {
                        #[derive(Deserialize)]
                        struct ClientMsg { text: String }
                        if let Ok(m) = serde_json::from_str::<ClientMsg>(&text) {
                            let _ = tx_user.send(m.text).await;
                        }
                    }
                    Some(Ok(WsMessage::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }

    debug!(session = %session_id, "WebSocket disconnected");
    drop(tx_user);
    relay.abort();
    let _ = agent_task.await;
}

// ── SSE ───────────────────────────────────────────────────────────────────────

async fn sse_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Response {
    let mut session = match load_or_create(&state.storage, id).await {
        Ok(s) => s,
        Err(e) => return api_error(e),
    };

    let (tx_user, mut rx_user) = mpsc::channel::<String>(32);
    let (tx_events, _)         = broadcast::channel::<AgentEvent>(256);
    let rx_events              = tx_events.subscribe();

    // Register tx_user so POST /message can reach this session.
    sse_senders().await.lock().await.insert(id, tx_user);

    // Spawn agent loop.
    let agent = make_agent(&state);
    tokio::spawn(async move {
        agent.run(&mut session, &mut rx_user, &tx_events).await;
        // Deregister when done.
        sse_senders().await.lock().await.remove(&id);
    });

    // BroadcastStream → SSE Event stream.
    // tokio_stream::StreamExt::map works on Streams (not futures::StreamExt).
    let stream = BroadcastStream::new(rx_events).map(|res| {
        let data = res
            .ok()
            .and_then(|e| serde_json::to_string(&e).ok())
            .unwrap_or_default();
        Ok::<Event, std::convert::Infallible>(Event::default().data(data))
    });

    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("ping"),
        )
        .into_response()
}

async fn post_message(
    State(_state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_owned();

    if text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "text field is required" })),
        )
            .into_response();
    }

    let map = sse_senders().await.lock().await;
    match map.get(&id) {
        Some(tx) => {
            let _ = tx.send(text).await;
            Json(serde_json::json!({ "ok": true })).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "no active SSE session — connect first via GET /api/sessions/:id/sse"
            })),
        )
            .into_response(),
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

async fn load_or_create(storage: &Arc<dyn Storage>, id: Uuid) -> crate::errors::Result<Session> {
    match storage.load_session(id).await? {
        Some(s) => Ok(s),
        None    => storage.create_session(None).await,
    }
}

fn make_agent(state: &AppState) -> Agent {
    let mut hooks = HookChain::new();
    hooks.add(LoggingHook);
    hooks.add(ToolAnnouncerHook);
    Agent::new(
        Arc::clone(&state.llm),
        default_registry(Arc::clone(&state.storage)),
        hooks,
        Arc::clone(&state.storage),
        AgentConfig::default(),
    )
}

fn api_error(e: AgentError) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": e.to_string() })),
    )
        .into_response()
}
