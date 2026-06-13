/// `RunCodeTool` — execute agent-authored Python in the Monty sandbox.
///
/// Inspired by Pydantic's [Monty](https://github.com/pydantic/monty) project and
/// Anthropic's "programmatic tool calling" pattern: instead of issuing one tool
/// call per action, the agent writes a short Python script that orchestrates
/// multiple operations in a single LLM turn.
///
/// The script runs inside the Monty sandboxed interpreter. All five built-in
/// AgentX tools are exposed as async Python functions the script can `await`.
/// Monty's iterative `RunProgress` loop pauses execution at each host-function
/// call, dispatches it to the real `ToolRegistry`, and resumes with the result.
///
/// ## Host functions available inside the sandbox
///
/// | Python name   | Signature                                                  |
/// |---------------|------------------------------------------------------------|
/// | `read_file`   | `async def read_file(path: str) -> str`                    |
/// | `write_file`  | `async def write_file(path: str, content: str) -> str`     |
/// | `edit_file`   | `async def edit_file(path: str, old_str: str, new_str: str) -> str` |
/// | `list_files`  | `async def list_files(path: str = ".") -> list`            |
/// | `run_command` | `async def run_command(command: str) -> str`               |
///
/// ## Resource limits
///
/// Each execution is capped at 512 MiB heap, 200 stack frames, and a 30-second
/// wall-clock timeout to prevent runaway scripts.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use monty::{
    ExcType, ExtFunctionResult, LimitedTracker, MontyException, MontyObject, MontyRun,
    NameLookupResult, PrintWriter, ResourceLimits, RunProgress,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{debug, instrument, warn};

use crate::errors::{AgentError, Result};
use crate::tools::{Tool, ToolRegistry};

// ── resource limits ───────────────────────────────────────────────────────────

/// Maximum heap memory (512 MiB).
const MAX_MEMORY_BYTES: usize = 512 * 1024 * 1024;
/// Maximum call-stack depth.
const MAX_STACK_DEPTH: usize = 200;
/// Wall-clock timeout for a single sandbox execution.
const EXECUTION_TIMEOUT: Duration = Duration::from_secs(30);

// ── tool ──────────────────────────────────────────────────────────────────────

/// Executes agent-authored Python code in the Monty sandbox.
///
/// The `registry` is used to dispatch the five host-function calls
/// (`read_file`, `write_file`, `edit_file`, `list_files`, `run_command`)
/// that the sandboxed script may issue.
pub struct RunCodeTool {
    registry: Arc<ToolRegistry>,
}

impl RunCodeTool {
    /// Create a new `RunCodeTool` backed by the given `ToolRegistry`.
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self { registry }
    }
}

// ── Tool trait ────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RunCodeInput {
    code: String,
}

#[async_trait]
impl Tool for RunCodeTool {
    fn name(&self) -> &str {
        "run_code"
    }

    fn description(&self) -> &str {
        "Execute a Python script in a secure sandbox. The script has access to \
         these async host functions:\n\
         - read_file(path)                    → str\n\
         - write_file(path, content)          → str\n\
         - edit_file(path, old_str, new_str)  → str\n\
         - list_files(path='.')               → list\n\
         - run_command(command)               → str\n\n\
         Use run_code when a task benefits from expressing logic as a program — \
         loops, conditionals, aggregation across many files — rather than issuing \
         sequential individual tool calls. \
         print() output and the final expression value are both returned."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python script. May use async/await. \
                                    The five host functions are available as \
                                    top-level async callables."
                }
            },
            "required": ["code"]
        })
    }

    #[instrument(skip(self, input), name = "run_code")]
    async fn execute(&self, input: Value) -> Result<String> {
        let parsed: RunCodeInput =
            serde_json::from_value(input).map_err(|e| AgentError::InvalidToolInput {
                tool: self.name().into(),
                reason: e.to_string(),
            })?;

        debug!(code_len = parsed.code.len(), "executing Python sandbox");

        let registry = Arc::clone(&self.registry);
        let code = parsed.code.clone();

        // Monty is synchronous; run it on a blocking thread so we don't stall
        // the Tokio thread pool.
        tokio::task::spawn_blocking(move || run_in_sandbox(&code, &registry))
            .await
            .map_err(|e| AgentError::ToolExecution {
                tool: "run_code".into(),
                reason: format!("sandbox thread panicked: {e}"),
            })?
    }
}

// ── sandbox execution ─────────────────────────────────────────────────────────

/// Run `code` in the Monty sandbox, driving the `RunProgress` loop to dispatch
/// host-function calls through `registry`. Called from `spawn_blocking`.
fn run_in_sandbox(code: &str, registry: &ToolRegistry) -> Result<String> {
    let runner = MontyRun::new(code.to_owned(), "agent_script.py", vec![])
        .map_err(|e| AgentError::ToolExecution {
            tool: "run_code".into(),
            reason: format!("syntax error: {e}"),
        })?;

    let limits = ResourceLimits {
        max_memory: Some(MAX_MEMORY_BYTES),
        max_recursion_depth: Some(MAX_STACK_DEPTH),
        max_duration: Some(EXECUTION_TIMEOUT),
        ..ResourceLimits::new()
    };
    let tracker = LimitedTracker::new(limits);

    let mut stdout = String::new();
    let mut writer = PrintWriter::CollectString(&mut stdout);

    let mut progress = runner
        .start(vec![], tracker, writer.reborrow())
        .map_err(|e| monty_err("startup", &e))?;

    // Inner single-threaded Tokio runtime for dispatching async tool calls.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| AgentError::ToolExecution {
            tool: "run_code".into(),
            reason: format!("inner runtime build failed: {e}"),
        })?;

    let final_obj = loop {
        progress = match progress {
            RunProgress::Complete(obj) => break obj,

            RunProgress::FunctionCall(call) => {
                let fn_name = call.function_name.clone();
                let call_id = call.call_id;
                debug!(function = %fn_name, call_id, "sandbox → host dispatch");

                let ext_result = match build_tool_input(&fn_name, &call.args, &call.kwargs) {
                    Ok(tool_input) => {
                        match rt.block_on(registry.execute(&fn_name, tool_input)) {
                            Ok(result) => ExtFunctionResult::Return(MontyObject::String(result)),
                            Err(e) => {
                                warn!(tool = %fn_name, error = %e, "host tool returned error");
                                ExtFunctionResult::Error(MontyException::new(
                                    ExcType::RuntimeError,
                                    Some(e.to_string()),
                                ))
                            }
                        }
                    }
                    Err(reason) => ExtFunctionResult::Error(MontyException::new(
                        ExcType::TypeError,
                        Some(reason),
                    )),
                };

                // For `await fn()` calls Monty uses a two-step async pattern:
                // 1. resume_pending → registers a Future(call_id), yields ResolveFutures
                // 2. ResolveFutures.resume with the actual result → execution continues
                let resolve = call
                    .resume_pending(writer.reborrow())
                    .map_err(|e| monty_err(&fn_name, &e))?;

                match resolve {
                    RunProgress::ResolveFutures(rf) => rf
                        .resume(vec![(call_id, ext_result)], writer.reborrow())
                        .map_err(|e| monty_err(&format!("resolve {fn_name}"), &e))?,
                    other => other,
                }
            }

            RunProgress::NameLookup(lookup) => {
                // Monty pauses here the first time it encounters an unknown
                // name. We return a Function object for known host names;
                // the actual dispatch happens when Monty issues FunctionCall.
                let name = lookup.name.clone();
                let result = if is_host_function(&name) {
                    NameLookupResult::Value(MontyObject::Function {
                        name: name.clone(),
                        docstring: None,
                    })
                } else {
                    NameLookupResult::Undefined
                };
                lookup
                    .resume(result, writer.reborrow())
                    .map_err(|e| monty_err(&format!("name lookup '{name}'"), &e))?
            }

            RunProgress::OsCall(os_call) => {
                // Direct OS/filesystem calls are blocked — the agent must use
                // the host-function wrappers instead, which go through the
                // ConfirmCommandHook and hook chain.
                let exc = os_call.function.on_no_handler(&os_call.args);
                warn!(?os_call.function, "sandbox blocked OS call");
                os_call
                    .resume(ExtFunctionResult::Error(exc), writer.reborrow())
                    .map_err(|e| monty_err("os_call", &e))?
            }

            RunProgress::ResolveFutures(resolve) => {
                // Should only be reached if a FunctionCall did not go through
                // the resume_pending path above (e.g. an unknown async name).
                // Resolve all pending with NameError.
                warn!("unexpected ResolveFutures — resolving pending with NameError");
                let results = resolve
                    .pending_call_ids()
                    .iter()
                    .map(|&id| (
                        id,
                        ExtFunctionResult::Error(MontyException::new(
                            ExcType::NameError,
                            Some("unresolved async call".into()),
                        )),
                    ))
                    .collect();
                resolve
                    .resume(results, writer.reborrow())
                    .map_err(|e| monty_err("resolve_futures", &e))?
            }
        };
    };

    let return_value = monty_to_string(&final_obj);
    Ok(format_output(&stdout, &return_value))
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Names of the host functions the sandbox may call.
const HOST_FUNCTIONS: &[&str] =
    &["read_file", "write_file", "edit_file", "list_files", "run_command"];

fn is_host_function(name: &str) -> bool {
    HOST_FUNCTIONS.contains(&name)
}

/// Convert positional + keyword arguments from a Monty `FunctionCall` into the
/// `serde_json::Value` that `ToolRegistry::execute` expects.
fn build_tool_input(
    fn_name: &str,
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
) -> std::result::Result<Value, String> {
    // Collect keyword arguments into a string-keyed map.
    let mut kw: HashMap<String, &MontyObject> = HashMap::new();
    for (k, v) in kwargs {
        if let MontyObject::String(key) = k {
            kw.insert(key.clone(), v);
        }
    }

    let as_str = |obj: &MontyObject| match obj {
        MontyObject::String(s) => Ok(s.clone()),
        other => Err(format!("expected str, got {}", type_name(other))),
    };

    match fn_name {
        "read_file" => {
            let path = positional_or_kw(args, 0, "path", &kw)
                .ok_or("read_file requires 'path'")?;
            Ok(json!({ "path": as_str(path)? }))
        }
        "write_file" => {
            let path = positional_or_kw(args, 0, "path", &kw)
                .ok_or("write_file requires 'path'")?;
            let content = positional_or_kw(args, 1, "content", &kw)
                .ok_or("write_file requires 'content'")?;
            Ok(json!({ "path": as_str(path)?, "content": as_str(content)? }))
        }
        "edit_file" => {
            let path = positional_or_kw(args, 0, "path", &kw)
                .ok_or("edit_file requires 'path'")?;
            let old = positional_or_kw(args, 1, "old_str", &kw)
                .ok_or("edit_file requires 'old_str'")?;
            let new = positional_or_kw(args, 2, "new_str", &kw)
                .ok_or("edit_file requires 'new_str'")?;
            Ok(json!({
                "path":    as_str(path)?,
                "old_str": as_str(old)?,
                "new_str": as_str(new)?,
            }))
        }
        "list_files" => {
            let path = positional_or_kw(args, 0, "path", &kw)
                .map(|obj| as_str(obj))
                .transpose()?
                .unwrap_or_else(|| ".".to_owned());
            Ok(json!({ "path": path }))
        }
        "run_command" => {
            let command = positional_or_kw(args, 0, "command", &kw)
                .ok_or("run_command requires 'command'")?;
            Ok(json!({ "command": as_str(command)? }))
        }
        other => Err(format!("unknown host function '{other}'")),
    }
}

/// Look up an argument by positional index first, then by keyword name.
fn positional_or_kw<'a>(
    args: &'a [MontyObject],
    index: usize,
    kw_name: &str,
    kw: &'a HashMap<String, &'a MontyObject>,
) -> Option<&'a MontyObject> {
    args.get(index).or_else(|| kw.get(kw_name).copied())
}

/// Render a `MontyObject` as a human-readable string for the tool result.
fn monty_to_string(obj: &MontyObject) -> String {
    match obj {
        MontyObject::None => String::new(),
        MontyObject::String(s) => s.clone(),
        MontyObject::Bool(b) => b.to_string(),
        MontyObject::Int(n) => n.to_string(),
        MontyObject::BigInt(n) => n.to_string(),
        MontyObject::Float(f) => f.to_string(),
        MontyObject::List(items) => {
            let parts: Vec<_> = items.iter().map(monty_to_string).collect();
            format!("[{}]", parts.join(", "))
        }
        MontyObject::Tuple(items) => {
            let parts: Vec<_> = items.iter().map(monty_to_string).collect();
            format!("({})", parts.join(", "))
        }
        MontyObject::Dict(pairs) => {
            let parts: Vec<_> = pairs
                .into_iter()
                .map(|(k, v)| format!("{}: {}", monty_to_string(k), monty_to_string(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        MontyObject::Bytes(b) => format!("<bytes len={}>", b.len()),
        _ => format!("{obj:?}"),
    }
}

/// Python type name for a `MontyObject` — used in error messages.
fn type_name(obj: &MontyObject) -> &'static str {
    match obj {
        MontyObject::None => "NoneType",
        MontyObject::Bool(_) => "bool",
        MontyObject::Int(_) | MontyObject::BigInt(_) => "int",
        MontyObject::Float(_) => "float",
        MontyObject::String(_) => "str",
        MontyObject::Bytes(_) => "bytes",
        MontyObject::List(_) => "list",
        MontyObject::Tuple(_) => "tuple",
        MontyObject::Dict(_) => "dict",
        MontyObject::Ellipsis => "ellipsis",
        _ => "object",
    }
}

/// Format the final tool result from captured stdout and the script's return value.
fn format_output(stdout: &str, return_value: &str) -> String {
    match (stdout.trim_end().is_empty(), return_value.is_empty()) {
        (true, true) => "(no output)".to_owned(),
        (false, true) => stdout.trim_end().to_owned(),
        (true, false) => return_value.to_owned(),
        (false, false) => {
            format!("{}\n\nreturn value: {return_value}", stdout.trim_end())
        }
    }
}

/// Wrap a `MontyException` into an `AgentError::ToolExecution`.
fn monty_err(context: &str, e: &MontyException) -> AgentError {
    AgentError::ToolExecution {
        tool: "run_code".into(),
        reason: format!("{context}: {e}"),
    }
}
