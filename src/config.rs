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

        // Treat empty env vars as absent — a blank `RESUME_SESSION=` in .env
        // should mean "no session", not an empty (invalid) UUID.
        let non_empty = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());

        Ok(Self {
            storage_backend,
            workspace_dir,
            session_label: non_empty("SESSION_LABEL"),
            resume_session: non_empty("RESUME_SESSION"),
            confirm_commands,
        })
    }
}
