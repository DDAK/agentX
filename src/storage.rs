/// Storage abstraction layer.
///
/// The `Storage` trait is the single dependency-injection point for all
/// persistence concerns.  Two concrete backends are provided:
///
/// - `FilesystemStorage` — reads / writes files on the local filesystem;
///   suitable for tools that need to touch source files.
/// - `PostgresStorage`  — stores agent sessions & conversation history in a
///   Postgres database; suitable for long-lived, resumable sessions.
///
/// Consumers receive a `Arc<dyn Storage + Send + Sync>` so they never depend
/// on a concrete type.
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, instrument};
use uuid::Uuid;

use crate::errors::{AgentError, Result};
use crate::llm::Message;

// ── session model ─────────────────────────────────────────────────────────────

/// Persistent representation of a single agent session (one user conversation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Human-readable label, e.g. "refactor auth module".
    pub label: Option<String>,
    /// Full conversation history for this session.
    pub messages: Vec<Message>,
}

impl Session {
    pub fn new(label: Option<String>) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Uuid::new_v4(),
            created_at: now,
            updated_at: now,
            label,
            messages: Vec::new(),
        }
    }
}

// ── the trait ─────────────────────────────────────────────────────────────────

/// Unified interface for all persistent storage operations.
///
/// Methods are grouped by concern:
/// - **Files** — low-level read/write used by agent tools.
/// - **Sessions** — CRUD for conversation sessions.
#[async_trait]
pub trait Storage: Send + Sync {
    // ── file operations ───────────────────────────────────────────────────

    /// Read the full contents of a file as UTF-8 text.
    async fn read_file(&self, path: &Path) -> Result<String>;

    /// Write (create or overwrite) a file with the given contents.
    async fn write_file(&self, path: &Path, content: &str) -> Result<()>;

    /// List all entries (files and directories) under `path`.
    /// Directories are returned with a trailing `/`.
    async fn list_files(&self, path: &Path) -> Result<Vec<String>>;

    // ── session operations ────────────────────────────────────────────────

    /// Create and persist a new session, returning it.
    async fn create_session(&self, label: Option<String>) -> Result<Session>;

    /// Load an existing session by ID.  Returns `None` if not found.
    async fn load_session(&self, id: Uuid) -> Result<Option<Session>>;

    /// Persist (upsert) a session.
    async fn save_session(&self, session: &Session) -> Result<()>;

    /// Return a list of all sessions, most-recent first.
    #[allow(dead_code)]
    async fn list_sessions(&self) -> Result<Vec<Session>>;
}

// ── FilesystemStorage ─────────────────────────────────────────────────────────

/// Filesystem backend — file ops are physical I/O; sessions are JSON files in
/// a `.agentx/sessions/` directory.
pub struct FilesystemStorage {
    /// Root directory for file operations (defaults to current working dir).
    root: PathBuf,
    /// Directory where session JSON files are stored.
    sessions_dir: PathBuf,
}

impl FilesystemStorage {
    /// Create a new `FilesystemStorage` rooted at `root`.
    /// The sessions directory is `<root>/.agentx/sessions/`.
    pub async fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let sessions_dir = root.join(".agentx").join("sessions");
        tokio::fs::create_dir_all(&sessions_dir).await?;
        Ok(Self { root, sessions_dir })
    }

    fn abs(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        }
    }
}

#[async_trait]
impl Storage for FilesystemStorage {
    #[instrument(skip(self), fields(path = %path.display()))]
    async fn read_file(&self, path: &Path) -> Result<String> {
        let full = self.abs(path);
        debug!("reading file");
        tokio::fs::read_to_string(&full)
            .await
            .map_err(|e| AgentError::Io(e))
    }

    #[instrument(skip(self, content), fields(path = %path.display()))]
    async fn write_file(&self, path: &Path, content: &str) -> Result<()> {
        let full = self.abs(path);
        debug!("writing file");
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&full, content)
            .await
            .map_err(AgentError::Io)
    }

    #[instrument(skip(self), fields(path = %path.display()))]
    async fn list_files(&self, path: &Path) -> Result<Vec<String>> {
        let full = self.abs(path);
        debug!("listing files");
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&full).await?;
        while let Some(entry) = dir.next_entry().await? {
            let file_name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = entry.file_type().await?.is_dir();
            if is_dir {
                entries.push(format!("{file_name}/"));
            } else {
                entries.push(file_name);
            }
        }
        entries.sort();
        Ok(entries)
    }

    async fn create_session(&self, label: Option<String>) -> Result<Session> {
        let session = Session::new(label);
        self.save_session(&session).await?;
        Ok(session)
    }

    async fn load_session(&self, id: Uuid) -> Result<Option<Session>> {
        let path = self.sessions_dir.join(format!("{id}.json"));
        match tokio::fs::read_to_string(&path).await {
            Ok(text) => {
                let s: Session = serde_json::from_str(&text)?;
                Ok(Some(s))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(AgentError::Io(e)),
        }
    }

    async fn save_session(&self, session: &Session) -> Result<()> {
        let path = self.sessions_dir.join(format!("{}.json", session.id));
        let text = serde_json::to_string_pretty(session)?;
        tokio::fs::write(&path, text).await.map_err(AgentError::Io)
    }

    async fn list_sessions(&self) -> Result<Vec<Session>> {
        let mut sessions = Vec::new();
        let mut dir = tokio::fs::read_dir(&self.sessions_dir).await?;
        while let Some(entry) = dir.next_entry().await? {
            if entry
                .path()
                .extension()
                .map(|e| e == "json")
                .unwrap_or(false)
            {
                if let Ok(text) = tokio::fs::read_to_string(entry.path()).await {
                    if let Ok(s) = serde_json::from_str::<Session>(&text) {
                        sessions.push(s);
                    }
                }
            }
        }
        // Most-recent first.
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }
}

// ── PostgresStorage ───────────────────────────────────────────────────────────

use deadpool_postgres::{Config as PgConfig, Pool, Runtime};
use tokio_postgres::NoTls;

/// Postgres backend — file ops hit the real filesystem (same as
/// `FilesystemStorage`); sessions are stored in a `sessions` table.
///
/// Schema (auto-applied on startup):
/// ```sql
/// CREATE TABLE IF NOT EXISTS sessions (
///   id          UUID PRIMARY KEY,
///   created_at  TIMESTAMPTZ NOT NULL,
///   updated_at  TIMESTAMPTZ NOT NULL,
///   label       TEXT,
///   messages    JSONB NOT NULL DEFAULT '[]'
/// );
/// ```
pub struct PostgresStorage {
    pool: Pool,
    /// Where file-system tool operations are rooted.
    fs_root: PathBuf,
}

impl PostgresStorage {
    /// Connect to Postgres using `DATABASE_URL` from the environment and apply
    /// the schema migration.
    pub async fn from_env(fs_root: impl Into<PathBuf>) -> Result<Self> {
        let database_url = std::env::var("DATABASE_URL").map_err(|_| {
            AgentError::Config("DATABASE_URL environment variable is not set".into())
        })?;

        let mut pg_cfg = PgConfig::new();
        pg_cfg.url = Some(database_url);

        let pool = pg_cfg
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        // Verify connectivity and run DDL.
        let client = pool
            .get()
            .await
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        client
            .batch_execute(
                "CREATE TABLE IF NOT EXISTS sessions (
                    id          UUID        PRIMARY KEY,
                    created_at  TIMESTAMPTZ NOT NULL,
                    updated_at  TIMESTAMPTZ NOT NULL,
                    label       TEXT,
                    messages    JSONB       NOT NULL DEFAULT '[]'
                );",
            )
            .await
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        Ok(Self {
            pool,
            fs_root: fs_root.into(),
        })
    }

    fn abs(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.fs_root.join(path)
        }
    }
}

#[async_trait]
impl Storage for PostgresStorage {
    // ── file ops are identical to filesystem backend ───────────────────────

    async fn read_file(&self, path: &Path) -> Result<String> {
        let full = self.abs(path);
        tokio::fs::read_to_string(&full)
            .await
            .map_err(AgentError::Io)
    }

    async fn write_file(&self, path: &Path, content: &str) -> Result<()> {
        let full = self.abs(path);
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&full, content)
            .await
            .map_err(AgentError::Io)
    }

    async fn list_files(&self, path: &Path) -> Result<Vec<String>> {
        let full = self.abs(path);
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&full).await?;
        while let Some(entry) = dir.next_entry().await? {
            let file_name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = entry.file_type().await?.is_dir();
            if is_dir {
                entries.push(format!("{file_name}/"));
            } else {
                entries.push(file_name);
            }
        }
        entries.sort();
        Ok(entries)
    }

    // ── session ops hit Postgres ───────────────────────────────────────────

    async fn create_session(&self, label: Option<String>) -> Result<Session> {
        let session = Session::new(label);
        self.save_session(&session).await?;
        Ok(session)
    }

    async fn load_session(&self, id: Uuid) -> Result<Option<Session>> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        let row = client
            .query_opt(
                "SELECT id, created_at, updated_at, label, messages
                 FROM sessions WHERE id = $1",
                &[&id],
            )
            .await
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        match row {
            None => Ok(None),
            Some(r) => {
                let messages_val: serde_json::Value = r.get("messages");
                let messages: Vec<Message> = serde_json::from_value(messages_val)?;
                Ok(Some(Session {
                    id: r.get("id"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                    label: r.get("label"),
                    messages,
                }))
            }
        }
    }

    async fn save_session(&self, session: &Session) -> Result<()> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        let messages_json = serde_json::to_value(&session.messages)?;

        client
            .execute(
                "INSERT INTO sessions (id, created_at, updated_at, label, messages)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (id) DO UPDATE
                   SET updated_at = EXCLUDED.updated_at,
                       label      = EXCLUDED.label,
                       messages   = EXCLUDED.messages",
                &[
                    &session.id,
                    &session.created_at,
                    &session.updated_at,
                    &session.label,
                    &messages_json,
                ],
            )
            .await
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<Session>> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        let rows = client
            .query(
                "SELECT id, created_at, updated_at, label, messages
                 FROM sessions ORDER BY updated_at DESC",
                &[],
            )
            .await
            .map_err(|e| AgentError::Storage(e.to_string()))?;

        let mut sessions = Vec::with_capacity(rows.len());
        for r in rows {
            let messages_val: serde_json::Value = r.get("messages");
            let messages: Vec<Message> = serde_json::from_value(messages_val)?;
            sessions.push(Session {
                id: r.get("id"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                label: r.get("label"),
                messages,
            });
        }
        Ok(sessions)
    }
}
