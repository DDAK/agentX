/// Application configuration loaded from environment variables / `.env`.
use crate::errors::Result;

#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Which storage backend to use: "filesystem" | "postgres"
    pub storage_backend: StorageBackend,
    /// Root directory for filesystem operations.
    pub workspace_dir: std::path::PathBuf,
    /// Session label for new sessions (optional).
    pub session_label: Option<String>,
    /// Session ID to resume (optional UUID string).
    pub resume_session: Option<String>,
    /// Whether to require confirmation before running shell commands.
    pub confirm_commands: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageBackend {
    Filesystem,
    Postgres,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        let storage_backend = match std::env::var("STORAGE_BACKEND")
            .unwrap_or_else(|_| "filesystem".into())
            .to_lowercase()
            .as_str()
        {
            "postgres" => StorageBackend::Postgres,
            _ => StorageBackend::Filesystem,
        };

        let workspace_dir = std::env::var("WORKSPACE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| ".".into()));

        let confirm_commands = std::env::var("CONFIRM_COMMANDS")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        Ok(Self {
            storage_backend,
            workspace_dir,
            session_label: std::env::var("SESSION_LABEL").ok(),
            resume_session: std::env::var("RESUME_SESSION").ok(),
            confirm_commands,
        })
    }
}
