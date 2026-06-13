/// Centralized error types for the entire agent system.
///
/// We use `thiserror` for the derive macros, which gives us clean `Display`
/// implementations while keeping the source chain intact for `anyhow`.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("LLM inference failed: {0}")]
    Inference(String),

    #[error("Tool execution failed for '{tool}': {reason}")]
    ToolExecution { tool: String, reason: String },

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Invalid tool input for '{tool}': {reason}")]
    InvalidToolInput { tool: String, reason: String },

    #[error("Memory storage error: {0}")]
    Storage(String),

    #[error("Session error: {0}")]
    Session(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Convenience alias — most functions return this.
pub type Result<T> = std::result::Result<T, AgentError>;
