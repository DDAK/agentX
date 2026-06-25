/// Tool registry and built-in tools.
///
/// A `Tool` is anything that implements the `Tool` trait — a name, a JSON
/// Schema description, and an `execute` method that receives raw JSON input and
/// returns a string result.
///
/// The `ToolRegistry` holds all registered tools and is the single place the
/// agent calls into when the model requests a tool invocation.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tracing::{debug, instrument};

use crate::errors::{AgentError, Result};
use crate::llm::ToolDefinitionParam;
use crate::storage::Storage;

// ── trait ─────────────────────────────────────────────────────────────────────

/// Every tool the agent can call must implement this trait.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Machine-readable name; matches what the model will request.
    fn name(&self) -> &str;

    /// Human/model-readable description.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input parameters.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the raw JSON `input` the model provided.
    async fn execute(&self, input: Value) -> Result<String>;

    /// Convenience: convert this tool into the wire param the LLM expects.
    fn to_definition_param(&self) -> ToolDefinitionParam {
        ToolDefinitionParam {
            kind: "function".into(),
            function: crate::llm::FunctionDefinition {
                name: self.name().into(),
                description: self.description().into(),
                parameters: self.parameters_schema(),
            },
        }
    }
}

// ── registry ──────────────────────────────────────────────────────────────────

/// Holds all tools and dispatches calls by name.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.  The tool's `name()` is the lookup key.
    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.insert(tool.name().to_owned(), Arc::new(tool));
    }

    /// Register an already-boxed tool (e.g. MCP tools shared behind an `Arc`).
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_owned(), tool);
    }

    /// Whether a tool with this name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Execute a named tool with the given JSON input.
    #[instrument(skip(self, input), fields(tool = %name))]
    pub async fn execute(&self, name: &str, input: Value) -> Result<String> {
        let tool = self.tools.get(name).ok_or_else(|| AgentError::ToolNotFound(name.into()))?;
        debug!("executing tool");
        tool.execute(input).await
    }

    /// Return all tool definitions to send to the model.
    pub fn definitions(&self) -> Vec<ToolDefinitionParam> {
        self.tools.values().map(|t| t.to_definition_param()).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── read_file ─────────────────────────────────────────────────────────────────

pub struct ReadFileTool {
    storage: Arc<dyn Storage>,
}

impl ReadFileTool {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }
}

#[derive(Deserialize)]
struct ReadFileInput {
    path: String,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the full UTF-8 contents of a file at the given relative path. \
         Use this to inspect source files, configuration, or any text file."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to read."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let parsed: ReadFileInput =
            serde_json::from_value(input).map_err(|e| AgentError::InvalidToolInput {
                tool: self.name().into(),
                reason: e.to_string(),
            })?;

        self.storage.read_file(Path::new(&parsed.path)).await
    }
}

// ── write_file ────────────────────────────────────────────────────────────────

pub struct WriteFileTool {
    storage: Arc<dyn Storage>,
}

impl WriteFileTool {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }
}

#[derive(Deserialize)]
struct WriteFileInput {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Create or completely overwrite a file with the given content. \
         The parent directory is created automatically if needed. \
         Prefer edit_file for small targeted changes to existing files."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to create or overwrite."
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file."
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let parsed: WriteFileInput =
            serde_json::from_value(input).map_err(|e| AgentError::InvalidToolInput {
                tool: self.name().into(),
                reason: e.to_string(),
            })?;

        self.storage
            .write_file(Path::new(&parsed.path), &parsed.content)
            .await?;
        Ok(format!("Successfully wrote {}", parsed.path))
    }
}

// ── edit_file ─────────────────────────────────────────────────────────────────

pub struct EditFileTool {
    storage: Arc<dyn Storage>,
}

impl EditFileTool {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }
}

#[derive(Deserialize)]
struct EditFileInput {
    path: String,
    old_str: String,
    new_str: String,
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Make a targeted edit to an existing file by replacing an exact string \
         `old_str` with `new_str`. The match must be unique within the file. \
         If the file does not exist and `old_str` is empty, it will be created \
         with `new_str` as its content. `old_str` and `new_str` must differ."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to edit."
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact text to search for. Must appear exactly once."
                },
                "new_str": {
                    "type": "string",
                    "description": "Text to replace old_str with."
                }
            },
            "required": ["path", "old_str", "new_str"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let parsed: EditFileInput =
            serde_json::from_value(input).map_err(|e| AgentError::InvalidToolInput {
                tool: self.name().into(),
                reason: e.to_string(),
            })?;

        if parsed.old_str == parsed.new_str {
            return Err(AgentError::InvalidToolInput {
                tool: self.name().into(),
                reason: "old_str and new_str must be different".into(),
            });
        }

        // Try reading the file first.
        match self.storage.read_file(Path::new(&parsed.path)).await {
            Ok(content) => {
                // Verify the old string appears exactly once.
                let occurrences = content.matches(&parsed.old_str).count();
                match occurrences {
                    0 => Err(AgentError::ToolExecution {
                        tool: self.name().into(),
                        reason: format!(
                            "'old_str' not found in {}",
                            parsed.path
                        ),
                    }),
                    1 => {
                        let new_content = content.replacen(&parsed.old_str, &parsed.new_str, 1);
                        self.storage
                            .write_file(Path::new(&parsed.path), &new_content)
                            .await?;
                        Ok("OK".into())
                    }
                    n => Err(AgentError::ToolExecution {
                        tool: self.name().into(),
                        reason: format!(
                            "'old_str' matches {n} times in {}; it must match exactly once",
                            parsed.path
                        ),
                    }),
                }
            }
            Err(AgentError::Io(ref io_err))
                if io_err.kind() == std::io::ErrorKind::NotFound
                    && parsed.old_str.is_empty() =>
            {
                // File does not exist and old_str is empty → create it.
                self.storage
                    .write_file(Path::new(&parsed.path), &parsed.new_str)
                    .await?;
                Ok(format!("Created {}", parsed.path))
            }
            Err(e) => Err(e),
        }
    }
}

// ── list_files ────────────────────────────────────────────────────────────────

pub struct ListFilesTool {
    storage: Arc<dyn Storage>,
}

impl ListFilesTool {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }
}

#[derive(Deserialize)]
struct ListFilesInput {
    #[serde(default)]
    path: Option<String>,
}

#[async_trait]
impl Tool for ListFilesTool {
    fn name(&self) -> &str {
        "list_files"
    }

    fn description(&self) -> &str {
        "List files and directories at the given path. Directories are shown \
         with a trailing '/'. Omit `path` to list the current working directory."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional relative path to list. Defaults to '.' (current directory)."
                }
            }
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let parsed: ListFilesInput =
            serde_json::from_value(input).map_err(|e| AgentError::InvalidToolInput {
                tool: self.name().into(),
                reason: e.to_string(),
            })?;

        let dir = parsed.path.unwrap_or_else(|| ".".into());
        let entries = self.storage.list_files(Path::new(&dir)).await?;
        Ok(serde_json::to_string(&entries)?)
    }
}

// ── run_command ───────────────────────────────────────────────────────────────

/// Runs an arbitrary shell command and returns its combined stdout + stderr.
///
/// # Security
/// This is intentionally powerful and should be guarded by a confirmation
/// hook in production environments.
pub struct RunCommandTool;

#[derive(Deserialize)]
struct RunCommandInput {
    command: String,
    #[serde(default)]
    working_dir: Option<String>,
}

#[async_trait]
impl Tool for RunCommandTool {
    fn name(&self) -> &str {
        "run_command"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its stdout + stderr. \
         Use for running tests, build tools, linters, or inspecting command output. \
         Commands run relative to the project root unless `working_dir` is specified."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute, e.g. 'cargo test' or 'ls -la'."
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional directory to run the command in."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let parsed: RunCommandInput =
            serde_json::from_value(input).map_err(|e| AgentError::InvalidToolInput {
                tool: self.name().into(),
                reason: e.to_string(),
            })?;

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg(&parsed.command);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        if let Some(dir) = &parsed.working_dir {
            cmd.current_dir(dir);
        }

        let output = cmd.output().await.map_err(|e| AgentError::ToolExecution {
            tool: self.name().into(),
            reason: e.to_string(),
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let exit_code = output.status.code().unwrap_or(-1);

        let result = format!(
            "exit_code: {exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        Ok(result)
    }
}

// ── helper: build the default tool set ───────────────────────────────────────

/// Build a `ToolRegistry` with all built-in tools pre-registered (no MCP).
///
/// The returned registry is wrapped in an `Arc` so it can be shared with
/// `RunCodeTool`, which needs a reference back to the registry to dispatch
/// host-function calls from inside the sandbox. Used by tests and as the
/// convenience entry point when MCP is not configured.
#[allow(dead_code)] // public helper used by integration tests
pub fn default_registry(storage: Arc<dyn Storage>) -> Arc<ToolRegistry> {
    default_registry_with_mcp(storage, &[])
}

/// Like [`default_registry`], plus any externally-supplied tools (e.g. tools
/// discovered from MCP servers). MCP clients are connected once at startup and
/// shared here behind `Arc`s, so this stays cheap to call per request.
pub fn default_registry_with_mcp(
    storage: Arc<dyn Storage>,
    extra: &[Arc<dyn Tool>],
) -> Arc<ToolRegistry> {
    // Inner registry: the tools reachable from inside the run_code sandbox —
    // the built-ins plus every MCP tool, so scripts can orchestrate MCP calls
    // alongside file ops. (RunCodeTool itself is excluded to avoid recursion.)
    let mut registry = ToolRegistry::new();
    registry.register(ReadFileTool::new(Arc::clone(&storage)));
    registry.register(WriteFileTool::new(Arc::clone(&storage)));
    registry.register(EditFileTool::new(Arc::clone(&storage)));
    registry.register(ListFilesTool::new(Arc::clone(&storage)));
    registry.register(RunCommandTool);
    for tool in extra {
        registry.register_arc(Arc::clone(tool));
    }
    let registry = Arc::new(registry);
    // Outer registry: everything the model can call directly.
    let mut full = ToolRegistry::new();
    full.register(ReadFileTool::new(Arc::clone(&storage)));
    full.register(WriteFileTool::new(Arc::clone(&storage)));
    full.register(EditFileTool::new(Arc::clone(&storage)));
    full.register(ListFilesTool::new(Arc::clone(&storage)));
    full.register(RunCommandTool);
    full.register(crate::code_tool::RunCodeTool::new(Arc::clone(&registry)));
    for tool in extra {
        full.register_arc(Arc::clone(tool));
    }
    Arc::new(full)
}
