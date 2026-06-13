/// The agent loop — headless, channel-driven.
///
/// The loop no longer reads stdin or writes stdout directly.  Instead:
///
/// - The caller sends a user message via `tx_in: mpsc::Sender<String>`.
/// - The agent emits typed `AgentEvent` values via `tx_out: broadcast::Sender<AgentEvent>`.
///
/// This design lets the same loop serve both the CLI (which wraps it in a
/// stdin/stdout adapter) and the HTTP/WebSocket API (which wires it to
/// connected clients).
///
/// ```text
///  Caller                   Agent loop
///  ──────                   ──────────
///  tx_in.send(msg) ──────►  recv user message
///                           append to conversation
///                    ┌────  emit AgentEvent::Thinking
///                    │      call LLM
///                    │      emit AgentEvent::Text(chunk)
///                    │      if tool_calls:
///                    │        emit AgentEvent::ToolCall { .. }
///                    │        execute tool
///                    │        emit AgentEvent::ToolResult { .. }
///                    │        loop
///                    └────  emit AgentEvent::TurnDone
/// ```
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, instrument, warn};

use crate::errors::{AgentError, Result};
use crate::hooks::{HookChain, HookEvent, HookResult};
use crate::llm::{LiteLlmClient, Message, MessageContent, MessageRole, ToolDefinitionParam};
use crate::storage::{Session, Storage};
use crate::tools::{ToolRegistry};
use std::sync::Arc as ToolArc;

// ── events emitted by the agent ───────────────────────────────────────────────

/// Every event the agent can emit during a turn.
///
/// Serialised as JSON and sent over WebSocket / SSE.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// The agent received the user message and is working.
    Thinking,
    /// A fragment of assistant text (may be partial during streaming).
    Text { text: String },
    /// The agent is about to call a tool.
    ToolCall { name: String, input: serde_json::Value },
    /// A tool finished; this is the result.
    ToolResult { name: String, result: String },
    /// The agent hit its iteration cap.
    IterationLimitReached,
    /// The full turn is complete.
    TurnDone,
    /// An error occurred.
    Error { message: String },
}

// ── configuration ─────────────────────────────────────────────────────────────

pub struct AgentConfig {
    /// Maximum tool-call iterations per user turn.
    pub max_iterations: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self { max_iterations: 32 }
    }
}

// ── system prompt ─────────────────────────────────────────────────────────────

const SYSTEM_PROMPT: &str = "\
You are AgentX, an expert software engineering assistant.
You have access to tools that let you read, write, and edit files on the local \
filesystem, list directory contents, and run shell commands.

Guidelines:
- Always read a file before editing it so you understand the existing content.
- Prefer targeted edits with edit_file over full rewrites.
- When asked about a project, start by listing files to get your bearings.
- Run tests after making changes to verify correctness.
- Explain briefly what you are doing and why.
- Be concise — do not repeat yourself.
";

// ── the agent ─────────────────────────────────────────────────────────────────

pub struct Agent {
    llm:     Arc<LiteLlmClient>,
    tools:   ToolArc<ToolRegistry>,
    hooks:   HookChain,
    storage: Arc<dyn Storage>,
    config:  AgentConfig,
}

impl Agent {
    pub fn new(
        llm:     Arc<LiteLlmClient>,
        tools:   ToolArc<ToolRegistry>,
        hooks:   HookChain,
        storage: Arc<dyn Storage>,
        config:  AgentConfig,
    ) -> Self {
        Self { llm, tools, hooks, storage, config }
    }

    /// Run the agent loop for one session.
    ///
    /// - `rx_in`  — receives user messages (one per turn).
    /// - `tx_out` — broadcasts `AgentEvent`s to all listeners.
    ///
    /// Returns when `rx_in` is closed (caller dropped the sender).
    pub async fn run(
        &self,
        session:  &mut Session,
        rx_in:    &mut mpsc::Receiver<String>,
        tx_out:   &broadcast::Sender<AgentEvent>,
    ) {
        // Inject system prompt on fresh sessions.
        if session.messages.is_empty() {
            session.messages.push(Message {
                role: MessageRole::System,
                content: MessageContent::Text(SYSTEM_PROMPT.into()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }

        let tool_defs = self.tools.definitions();
        let mut turn = 0usize;

        while let Some(user_msg) = rx_in.recv().await {
            let user_msg = user_msg.trim().to_owned();
            if user_msg.is_empty() {
                continue;
            }

            turn += 1;
            self.hooks.fire(&HookEvent::TurnStart { turn }).await;

            session.messages.push(Message::user(&user_msg));
            let _ = tx_out.send(AgentEvent::Thinking);

            let result = self
                .run_agentic_loop(session, &tool_defs, tx_out)
                .await;

            // Always persist, even on error.
            if let Err(e) = self.storage.save_session(session).await {
                warn!("failed to persist session: {e}");
            }

            if let Err(e) = result {
                let _ = tx_out.send(AgentEvent::Error { message: e.to_string() });
            }

            let _ = tx_out.send(AgentEvent::TurnDone);
            self.hooks.fire(&HookEvent::TurnEnd { turn }).await;
        }
    }

    /// Drive one user→model→tools cycle to completion.
    #[instrument(skip(self, session, tool_defs, tx_out))]
    async fn run_agentic_loop(
        &self,
        session:   &mut Session,
        tool_defs: &[ToolDefinitionParam],
        tx_out:    &broadcast::Sender<AgentEvent>,
    ) -> Result<()> {
        let mut iterations = 0;

        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                warn!(max = self.config.max_iterations, "iteration cap reached");
                let msg = "[AgentX hit the iteration limit and stopped.]";
                session.messages.push(Message::assistant(msg));
                let _ = tx_out.send(AgentEvent::IterationLimitReached);
                break;
            }

            self.hooks.fire(&HookEvent::BeforeInference {
                message_count: session.messages.len(),
            }).await;

            debug!(iteration = iterations, "calling LLM");

            let response = self.llm.chat(&session.messages, Some(tool_defs)).await?;

            self.hooks.fire(&HookEvent::AfterInference {
                content: response.content.clone(),
                tool_call_count: response.tool_calls.as_ref().map(|v| v.len()).unwrap_or(0),
            }).await;

            // Emit any text the model produced.
            if let Some(ref text) = response.content {
                if !text.is_empty() {
                    let _ = tx_out.send(AgentEvent::Text { text: text.clone() });
                }
            }

            // No tool calls → turn is done.
            let Some(tool_calls) = response.tool_calls else {
                session.messages.push(Message::assistant(
                    response.content.as_deref().unwrap_or(""),
                ));
                break;
            };
            if tool_calls.is_empty() {
                session.messages.push(Message::assistant(
                    response.content.as_deref().unwrap_or(""),
                ));
                break;
            }

            // Attach the assistant turn (with tool_calls) to history.
            session.messages.push(Message {
                role: MessageRole::Assistant,
                content: MessageContent::Text(
                    response.content.clone().unwrap_or_default(),
                ),
                tool_calls: Some(tool_calls.clone()),
                tool_call_id: None,
                name: None,
            });

            // Execute each tool call.
            for tc in &tool_calls {
                let input: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).map_err(|e| {
                        AgentError::InvalidToolInput {
                            tool: tc.function.name.clone(),
                            reason: e.to_string(),
                        }
                    })?;

                let _ = tx_out.send(AgentEvent::ToolCall {
                    name:  tc.function.name.clone(),
                    input: input.clone(),
                });

                // Before-tool hook (can reject the call).
                match self.hooks.fire(&HookEvent::BeforeToolExecution {
                    tool_name: tc.function.name.clone(),
                    input:     input.clone(),
                }).await {
                    HookResult::Continue => {}
                    HookResult::Abort(reason) => {
                        let msg = format!("Tool call rejected: {reason}");
                        session.messages.push(Message::tool_result(&tc.id, &msg));
                        let _ = tx_out.send(AgentEvent::ToolResult {
                            name:   tc.function.name.clone(),
                            result: msg,
                        });
                        continue;
                    }
                }

                let result = self
                    .tools
                    .execute(&tc.function.name, input)
                    .await
                    .unwrap_or_else(|e| format!("ERROR: {e}"));

                self.hooks.fire(&HookEvent::AfterToolExecution {
                    tool_name: tc.function.name.clone(),
                    result:    result.clone(),
                }).await;

                let _ = tx_out.send(AgentEvent::ToolResult {
                    name:   tc.function.name.clone(),
                    result: result.clone(),
                });

                session.messages.push(Message::tool_result(&tc.id, &result));
            }
            // Loop — let the model react to tool results.
        }

        Ok(())
    }
}
