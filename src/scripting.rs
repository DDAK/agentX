/// Dynamic extensions — config, skills, and commands loaded from disk at
/// startup so they can be added, changed, or removed without recompiling.
///
/// Everything lives under one home directory (`AGENTX_HOME`, default
/// `./agentx.d`):
///
/// ```text
/// agentx.d/
///   config.yaml        agent behaviour (system prompt, iteration cap)
///   skills/*.md        prompt fragments appended to the system prompt
///   commands/*.rhai    scripted tools the model can call
/// ```
///
/// Design note: config is parsed with `serde_yaml` and skills are plain file
/// reads — running those *through* a scripting engine would be machinery for
/// nothing. Rhai earns its place for **commands**: dynamic behaviour that would
/// otherwise need a recompile. Each `.rhai` file becomes a [`Tool`].
///
/// Every part is optional: a missing home dir, or any missing subdirectory,
/// just yields defaults / no tools. Extensions are strictly opt-in.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use rhai::{Dynamic, Engine, Map, Scope, AST};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::errors::{AgentError, Result};
use crate::tools::Tool;

/// Resolve the extensions home directory (`AGENTX_HOME`, default `./agentx.d`).
pub fn home_dir() -> PathBuf {
    std::env::var("AGENTX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("agentx.d"))
}

// ── config (YAML) ──────────────────────────────────────────────────────────────

/// Agent behaviour loaded from `config.yaml`. Both fields are optional; the
/// caller supplies defaults for anything absent.
#[derive(Debug, Default, Deserialize)]
pub struct AgentYamlConfig {
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub max_iterations: Option<usize>,
}

/// Load `<home>/config.yaml`, or defaults if it's absent. A malformed file is
/// logged and treated as empty so a typo never takes the agent down.
pub fn load_config(home: &Path) -> AgentYamlConfig {
    let path = home.join("config.yaml");
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return AgentYamlConfig::default(),
    };
    match serde_yaml::from_str(&raw) {
        Ok(cfg) => {
            info!(path = %path.display(), "loaded agent config");
            cfg
        }
        Err(e) => {
            warn!(path = %path.display(), error = %e, "bad config.yaml; using defaults");
            AgentYamlConfig::default()
        }
    }
}

// ── skills (Markdown) ──────────────────────────────────────────────────────────

/// Concatenate every `<home>/skills/*.md` file into one prompt fragment, sorted
/// by filename for determinism. Returns an empty string when none exist.
pub fn load_skills(home: &Path) -> String {
    let dir = home.join("skills");
    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "md"))
            .collect(),
        Err(_) => return String::new(),
    };
    files.sort();

    let mut out = String::new();
    for path in &files {
        if let Ok(content) = std::fs::read_to_string(path) {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(content.trim_end());
        }
    }
    if !out.is_empty() {
        info!(skills = files.len(), "loaded skills");
    }
    out
}

// ── commands (Rhai) ──────────────────────────────────────────────────────────

/// A tool whose behaviour is a Rhai script.
///
/// Each `<home>/commands/<name>.rhai` becomes a tool named `<name>`. The script
/// may define two functions:
///
/// - `fn run(input)` — the entry point; `input` is a map of the model's
///   arguments. Its return value (stringified) is the tool result. Required.
/// - `fn meta()` — returns `#{ description: "...", parameters: #{...} }` to
///   describe the tool to the model. Optional; sensible defaults otherwise.
///
/// Host functions registered into the engine let scripts do real work:
/// `sh(cmd)`, `read_file(path)`, `write_file(path, content)`,
/// `list_files(path)` — all scoped to the workspace directory.
struct RhaiCommandTool {
    name: String,
    description: String,
    schema: Value,
    engine: Arc<Engine>,
    ast: Arc<AST>,
}

#[async_trait]
impl Tool for RhaiCommandTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let engine = Arc::clone(&self.engine);
        let ast = Arc::clone(&self.ast);
        let name = self.name.clone();

        // Rhai is synchronous; run it off the async worker pool.
        tokio::task::spawn_blocking(move || {
            let mut scope = Scope::new();
            let arg: Dynamic = json_to_dynamic(&input);
            engine
                .call_fn::<Dynamic>(&mut scope, &ast, "run", (arg,))
                .map(|v| dynamic_to_string(&v))
                .map_err(|e| AgentError::ToolExecution {
                    tool: name.clone(),
                    reason: e.to_string(),
                })
        })
        .await
        .map_err(|e| AgentError::ToolExecution {
            tool: self.name.clone(),
            reason: format!("command thread panicked: {e}"),
        })?
    }
}

/// Build a Rhai engine with the host functions scripts may call. `workspace`
/// scopes file operations; shell commands run with it as the working dir.
fn build_engine(workspace: PathBuf) -> Engine {
    let mut engine = Engine::new();
    // Keep runaway scripts in check.
    engine.set_max_operations(5_000_000);
    engine.set_max_call_levels(64);
    engine.set_max_string_size(10 * 1024 * 1024);

    let ws = workspace.clone();
    engine.register_fn("read_file", move |path: &str| -> String {
        std::fs::read_to_string(ws.join(path)).unwrap_or_default()
    });
    let ws = workspace.clone();
    engine.register_fn("write_file", move |path: &str, content: &str| {
        let full = ws.join(path);
        if let Some(parent) = full.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(full, content);
    });
    let ws = workspace.clone();
    engine.register_fn("list_files", move |path: &str| -> String {
        match std::fs::read_dir(ws.join(path)) {
            Ok(entries) => entries
                .flatten()
                .map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    if e.path().is_dir() { format!("{name}/") } else { name }
                })
                .collect::<Vec<_>>()
                .join("\n"),
            Err(e) => format!("ERROR: {e}"),
        }
    });
    let ws = workspace.clone();
    engine.register_fn("sh", move |cmd: &str| -> String {
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(&ws)
            .output();
        match out {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                let stderr = String::from_utf8_lossy(&o.stderr);
                format!("{stdout}{stderr}")
            }
            Err(e) => format!("ERROR: {e}"),
        }
    });
    engine
}

/// Compile one `.rhai` file into a tool. Reads `meta()` for its description and
/// parameter schema if present.
fn load_command(path: &Path, engine: Arc<Engine>) -> Result<Arc<dyn Tool>> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| AgentError::Config(format!("bad command filename: {}", path.display())))?
        .to_owned();

    let ast = engine
        .compile_file(path.to_path_buf())
        .map_err(|e| AgentError::Config(format!("compile {}: {e}", path.display())))?;
    let ast = Arc::new(ast);

    // Pull optional metadata by calling `meta()` if the script defines it.
    let (description, schema) = read_meta(&engine, &ast).unwrap_or_else(|| {
        (
            format!("Scripted command '{name}'."),
            json!({ "type": "object" }),
        )
    });

    Ok(Arc::new(RhaiCommandTool {
        name,
        description,
        schema,
        engine,
        ast,
    }))
}

/// Call `meta()` and translate its `#{ description, parameters }` map.
fn read_meta(engine: &Engine, ast: &AST) -> Option<(String, Value)> {
    let mut scope = Scope::new();
    let meta: Map = engine.call_fn(&mut scope, ast, "meta", ()).ok()?;
    let description = meta
        .get("description")
        .map(|d| d.to_string())
        .unwrap_or_default();
    let schema = meta
        .get("parameters")
        .map(dynamic_to_json)
        .unwrap_or_else(|| json!({ "type": "object" }));
    Some((description, schema))
}

/// Load every `<home>/commands/*.rhai` as a tool. A script that fails to
/// compile is logged and skipped — one broken command can't break the agent.
pub fn load_commands(home: &Path, workspace: PathBuf) -> Vec<Arc<dyn Tool>> {
    let dir = home.join("commands");
    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "rhai"))
            .collect(),
        Err(_) => return vec![],
    };
    files.sort();

    let engine = Arc::new(build_engine(workspace));
    let mut tools = Vec::new();
    for path in &files {
        match load_command(path, Arc::clone(&engine)) {
            Ok(tool) => {
                info!(command = %tool.name(), "loaded scripted command");
                tools.push(tool);
            }
            Err(e) => warn!(path = %path.display(), error = %e, "bad command; skipping"),
        }
    }
    debug!(total = tools.len(), "scripted commands registered");
    tools
}

// ── value conversion (JSON ⇄ Rhai Dynamic) ──────────────────────────────────

/// Convert the model's JSON arguments into a Rhai `Dynamic` for `run(input)`.
fn json_to_dynamic(v: &Value) -> Dynamic {
    match v {
        Value::Null => Dynamic::UNIT,
        Value::Bool(b) => (*b).into(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into()
            } else {
                n.as_f64().unwrap_or(0.0).into()
            }
        }
        Value::String(s) => s.clone().into(),
        Value::Array(items) => items.iter().map(json_to_dynamic).collect::<rhai::Array>().into(),
        Value::Object(map) => {
            let mut m = Map::new();
            for (k, val) in map {
                m.insert(k.clone().into(), json_to_dynamic(val));
            }
            Dynamic::from_map(m)
        }
    }
}

/// Convert a Rhai `Dynamic` (e.g. a `meta()` parameters map) back to JSON.
fn dynamic_to_json(d: &Dynamic) -> Value {
    if d.is_unit() {
        Value::Null
    } else if let Ok(b) = d.as_bool() {
        Value::Bool(b)
    } else if let Ok(i) = d.as_int() {
        json!(i)
    } else if let Ok(f) = d.as_float() {
        json!(f)
    } else if d.is_array() {
        let arr = d.clone().into_array().unwrap_or_default();
        Value::Array(arr.iter().map(dynamic_to_json).collect())
    } else if d.is_map() {
        let map = d.read_lock::<Map>().map(|m| m.clone()).unwrap_or_default();
        let mut obj = serde_json::Map::new();
        for (k, v) in map.iter() {
            obj.insert(k.to_string(), dynamic_to_json(v));
        }
        Value::Object(obj)
    } else {
        Value::String(d.to_string())
    }
}

/// Stringify a Rhai value for use as a tool result.
fn dynamic_to_string(d: &Dynamic) -> String {
    if d.is_unit() {
        String::new()
    } else if d.is_string() {
        d.clone().into_string().unwrap_or_default()
    } else {
        d.to_string()
    }
}

// ── bootstrap helper ─────────────────────────────────────────────────────────

/// Everything loaded from the extensions home dir.
pub struct Extensions {
    pub config: AgentYamlConfig,
    pub skills: String,
    pub commands: Vec<Arc<dyn Tool>>,
}

/// Load config, skills, and scripted commands from `AGENTX_HOME`.
pub fn load_all(workspace: PathBuf) -> Extensions {
    let home = home_dir();
    Extensions {
        config: load_config(&home),
        skills: load_skills(&home),
        commands: load_commands(&home, workspace),
    }
}

impl Extensions {
    /// Build the effective [`AgentConfig`](crate::agent::AgentConfig): YAML
    /// overrides the built-in defaults, and loaded skills are appended to the
    /// system prompt under a `# Skills` heading.
    pub fn agent_config(&self) -> crate::agent::AgentConfig {
        let mut prompt = self
            .config
            .system_prompt
            .clone()
            .unwrap_or_else(|| crate::agent::DEFAULT_SYSTEM_PROMPT.to_owned());
        if !self.skills.is_empty() {
            prompt.push_str("\n\n# Skills\n\n");
            prompt.push_str(&self.skills);
        }
        crate::agent::AgentConfig {
            max_iterations: self.config.max_iterations.unwrap_or(32),
            system_prompt: prompt,
        }
    }
}
