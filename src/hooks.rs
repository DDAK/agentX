/// Hook system for intercepting and observing agent lifecycle events.
///
/// Hooks are optional callbacks the agent fires at key points in its loop.
/// They can be used for:
/// - Logging / metrics
/// - Human-in-the-loop confirmation (e.g. before destructive commands)
/// - Post-processing of tool results
/// - Injecting additional context
///
/// All hooks receive an immutable reference to the event and return a
/// `HookResult` that can either continue execution or abort with an error.
use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value;


// ── event types ───────────────────────────────────────────────────────────────

/// Events fired by the agent.  Each variant carries relevant context.
#[derive(Debug, Clone)]
#[allow(dead_code)] // fields are consumed by user-implemented Hook impls
pub enum HookEvent {
    /// The agent is about to call the LLM.
    BeforeInference {
        /// Number of messages in the current conversation.
        message_count: usize,
    },

    /// The LLM produced a response.
    AfterInference {
        /// The text content, if any.
        content: Option<String>,
        /// Number of tool calls requested.
        tool_call_count: usize,
    },

    /// A tool is about to be executed.
    BeforeToolExecution {
        tool_name: String,
        input: Value,
    },

    /// A tool finished executing.
    AfterToolExecution {
        tool_name: String,
        result: String,
    },

    /// The agent loop is starting a new user turn.
    TurnStart { turn: usize },

    /// The agent loop has finished a complete user → model → tools cycle.
    TurnEnd { turn: usize },
}

/// What the hook tells the agent to do next.
#[derive(Debug)]
pub enum HookResult {
    /// Continue normal execution.
    Continue,
    /// Abort the current operation with an error message.
    Abort(String),
}

// ── trait ─────────────────────────────────────────────────────────────────────

/// A hook that can observe and influence agent execution.
#[async_trait]
pub trait Hook: Send + Sync {
    async fn on_event(&self, event: &HookEvent) -> HookResult;
}

// ── hook chain ────────────────────────────────────────────────────────────────

/// Runs a list of hooks in order, stopping at the first `Abort`.
pub struct HookChain {
    hooks: Vec<Box<dyn Hook>>,
}

impl HookChain {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    pub fn add(&mut self, hook: impl Hook + 'static) {
        self.hooks.push(Box::new(hook));
    }

    /// Fire an event through all hooks.
    pub async fn fire(&self, event: &HookEvent) -> HookResult {
        for hook in &self.hooks {
            match hook.on_event(event).await {
                HookResult::Continue => {}
                HookResult::Abort(msg) => return HookResult::Abort(msg),
            }
        }
        HookResult::Continue
    }
}

impl Default for HookChain {
    fn default() -> Self {
        Self::new()
    }
}

// ── built-in hooks ────────────────────────────────────────────────────────────

/// Logs every event to `tracing`.
pub struct LoggingHook;

#[async_trait]
impl Hook for LoggingHook {
    async fn on_event(&self, event: &HookEvent) -> HookResult {
        tracing::debug!(?event, "hook event");
        HookResult::Continue
    }
}

/// Prints tool calls to stdout so the user can see what the agent is doing.
pub struct ToolAnnouncerHook;

#[async_trait]
impl Hook for ToolAnnouncerHook {
    async fn on_event(&self, event: &HookEvent) -> HookResult {
        if let HookEvent::BeforeToolExecution { tool_name, input } = event {
            // Compact JSON for display.
            let compact = serde_json::to_string(input).unwrap_or_default();
            println!("\x1b[90m  → tool: {tool_name}({compact})\x1b[0m");
        }
        HookResult::Continue
    }
}

/// Interactively asks the user to confirm before running a `run_command` tool.
/// Useful when you want a human-in-the-loop safety gate.
///
/// The confirmation strategy is injected via a callback so that:
/// - CLI mode reads from stdin (`ConfirmCommandHook::stdin()`)
/// - Server/headless mode rejects automatically (`ConfirmCommandHook::auto_reject()`)
/// - Tests can inject any custom policy (`ConfirmCommandHook::custom(...)`)
pub struct ConfirmCommandHook {
    confirm_fn: Arc<dyn Fn(String) -> bool + Send + Sync>,
}

#[allow(dead_code)]
impl ConfirmCommandHook {
    /// Read y/N from stdin — suitable for CLI mode only.
    pub fn stdin() -> Self {
        Self {
            confirm_fn: Arc::new(|cmd| {
                use std::io::{BufRead, Write};
                print!("\x1b[33mAllow command `{cmd}`? [y/N] \x1b[0m");
                std::io::stdout().flush().ok();
                let mut line = String::new();
                std::io::stdin().lock().read_line(&mut line).ok();
                line.trim().eq_ignore_ascii_case("y")
            }),
        }
    }

    /// Always reject — safe default for server / headless mode.
    pub fn auto_reject() -> Self {
        Self {
            confirm_fn: Arc::new(|_cmd| false),
        }
    }

    /// Custom approval function — e.g. an allowlist or out-of-band channel.
    pub fn custom(f: impl Fn(String) -> bool + Send + Sync + 'static) -> Self {
        Self { confirm_fn: Arc::new(f) }
    }
}

#[async_trait]
impl Hook for ConfirmCommandHook {
    async fn on_event(&self, event: &HookEvent) -> HookResult {
        if let HookEvent::BeforeToolExecution { tool_name, input } = event {
            if tool_name == "run_command" {
                let cmd = input
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_owned();

                // Run the (potentially blocking) confirm fn off the async executor.
                let confirm = Arc::clone(&self.confirm_fn);
                let cmd_for_closure = cmd.clone();
                let approved = tokio::task::spawn_blocking(move || confirm(cmd_for_closure))
                    .await
                    .unwrap_or(false);

                if !approved {
                    return HookResult::Abort(format!("User rejected command: {cmd}"));
                }
            }
        }
        HookResult::Continue
    }
}
