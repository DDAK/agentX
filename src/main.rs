/// AgentX — entry point.
///
/// Modes
/// ─────
///   (default / --serve)   Start the HTTP API server (WebSocket + SSE + REST)
///   --cli                 Interactive terminal REPL (stdin/stdout)
///
/// The mode is selected via the `AGENTX_MODE` env var or the first CLI arg.
mod agent;
mod api;
mod code_tool;
mod config;
mod errors;
mod hooks;
mod llm;
mod storage;
mod tools;

use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{broadcast, mpsc};
use tracing::info;
use uuid::Uuid;

use crate::agent::{Agent, AgentConfig, AgentEvent};
use crate::api::build_router;
use crate::config::{AppConfig, StorageBackend};
use crate::errors::Result;
use crate::hooks::{ConfirmCommandHook, HookChain, LoggingHook, ToolAnnouncerHook};
use crate::llm::{LiteLlmClient, LiteLlmConfig};
use crate::storage::{FilesystemStorage, PostgresStorage, Storage};
use crate::tools::default_registry;

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "agentx=info,warn".into()),
        )
        .init();

    let mode = std::env::args().nth(1).unwrap_or_default();
    let cli_mode = mode == "--cli"
        || std::env::var("AGENTX_MODE")
            .map(|v| v == "cli")
            .unwrap_or(false);

    if let Err(e) = if cli_mode { run_cli().await } else { run_server().await } {
        eprintln!("\x1b[31mFatal: {e}\x1b[0m");
        std::process::exit(1);
    }
}

// ── shared bootstrap ──────────────────────────────────────────────────────────

async fn bootstrap() -> Result<(Arc<dyn Storage>, Arc<LiteLlmClient>, Arc<AppConfig>)> {
    let app_cfg = Arc::new(AppConfig::from_env()?);
    let llm_cfg = LiteLlmConfig::from_env()?;

    info!(
        model  = %llm_cfg.default_model,
        mode   = ?app_cfg.storage_backend,
        workspace = %app_cfg.workspace_dir.display(),
        "AgentX starting"
    );

    let storage: Arc<dyn Storage> = match app_cfg.storage_backend {
        StorageBackend::Filesystem => {
            Arc::new(FilesystemStorage::new(&app_cfg.workspace_dir).await?)
        }
        StorageBackend::Postgres => {
            Arc::new(PostgresStorage::from_env(&app_cfg.workspace_dir).await?)
        }
    };

    let llm = Arc::new(LiteLlmClient::new(llm_cfg));
    Ok((storage, llm, app_cfg))
}

// ── server mode ───────────────────────────────────────────────────────────────

async fn run_server() -> Result<()> {
    let (storage, llm, app_cfg) = bootstrap().await?;

    let bind = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let router = build_router(storage, llm, app_cfg).await;

    info!(addr = %bind, "HTTP server listening");

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .map_err(|e| crate::errors::AgentError::Config(format!("bind {bind}: {e}")))?;

    axum::serve(listener, router)
        .await
        .map_err(|e| crate::errors::AgentError::Config(e.to_string()))
}

// ── CLI mode ──────────────────────────────────────────────────────────────────

async fn run_cli() -> Result<()> {
    let (storage, llm, app_cfg) = bootstrap().await?;

    // Build tool registry + hooks.
    let registry = default_registry(Arc::clone(&storage));
    let mut hooks = HookChain::new();
    hooks.add(LoggingHook);
    hooks.add(ToolAnnouncerHook);
    if app_cfg.confirm_commands {
        hooks.add(ConfirmCommandHook::stdin());
    }

    // Load or create session.
    let mut session = if let Some(id_str) = &app_cfg.resume_session {
        let id = id_str
            .parse::<Uuid>()
            .map_err(|_| crate::errors::AgentError::Session("invalid UUID".into()))?;
        storage
            .load_session(id)
            .await?
            .ok_or_else(|| crate::errors::AgentError::Session(format!("session {id} not found")))?
    } else {
        storage.create_session(app_cfg.session_label.clone()).await?
    };

    info!(session_id = %session.id, "session ready");

    // Channel pair connecting stdin → agent and agent → stdout.
    let (tx_in, mut rx_in)   = mpsc::channel::<String>(32);
    let (tx_out, _)          = broadcast::channel::<AgentEvent>(256);
    let tx_out_clone         = tx_out.clone();
    let mut rx_events        = tx_out.subscribe();

    // Task: print AgentEvents to stdout.
    let print_task = tokio::spawn(async move {
        while let Ok(event) = rx_events.recv().await {
            match event {
                AgentEvent::Thinking => {
                    print!("\x1b[90m…thinking…\x1b[0m ");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
                AgentEvent::Text { text } => {
                    println!("\x1b[93mAgentX\x1b[0m: {text}");
                }
                AgentEvent::ToolCall { name, input } => {
                    let compact = serde_json::to_string(&input).unwrap_or_default();
                    println!("\x1b[90m  → {name}({compact})\x1b[0m");
                }
                AgentEvent::ToolResult { name, result } => {
                    // Truncate long results for the terminal.
                    let preview = if result.len() > 200 {
                        format!("{}…", &result[..200])
                    } else {
                        result.clone()
                    };
                    println!("\x1b[90m  ← {name}: {preview}\x1b[0m");
                }
                AgentEvent::IterationLimitReached => {
                    println!("\x1b[31m[iteration limit reached]\x1b[0m");
                }
                AgentEvent::Error { message } => {
                    println!("\x1b[31mError: {message}\x1b[0m");
                }
                AgentEvent::TurnDone => {
                    println!(); // blank line between turns
                }
            }
        }
    });

    // Task: run agent loop.
    let agent = Agent::new(llm, registry, hooks, Arc::clone(&storage), AgentConfig::default());
    let agent_task = tokio::spawn(async move {
        agent.run(&mut session, &mut rx_in, &tx_out_clone).await;
    });

    // Main: read stdin lines → send to agent.
    println!("\x1b[1mAgentX CLI\x1b[0m — Ctrl-D to quit");
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();

    loop {
        print!("\x1b[94mYou\x1b[0m: ");
        use std::io::Write;
        let _ = std::io::stdout().flush();

        match reader.next_line().await {
            Ok(Some(line)) => {
                if !line.trim().is_empty() {
                    let _ = tx_in.send(line).await;
                    // Wait for TurnDone before prompting again.
                    // We just sleep briefly — a real TUI would track state.
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
            _ => break, // EOF or error
        }
    }

    println!("\nBye!");
    drop(tx_in);
    print_task.abort();
    let _ = agent_task.await;
    Ok(())
}
