/// LiteLLM gateway client.
///
/// LiteLLM exposes an OpenAI-compatible `/chat/completions` endpoint, so we
/// speak plain JSON rather than pulling in a heavy SDK crate.  This keeps the
/// dependency tree small and makes the serialisation logic trivial to follow.
use anyhow::Context;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::errors::{AgentError, Result};

// ── wire types ────────────────────────────────────────────────────────────────

/// A single message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: MessageContent,
    /// Present on tool-call messages produced by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Present on tool-result messages we send back.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Present on tool-result messages we send back.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Construct a plain user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Text(text.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Construct a plain assistant message (text only).
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(text.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Construct a tool-result message to feed back to the model.
    pub fn tool_result(tool_call_id: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: MessageContent::Text(result.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Message content — either plain text or a structured array of parts.
/// Most responses are plain text; we keep the array variant for completeness.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract inner text regardless of variant.
    #[allow(dead_code)]
    pub fn as_text(&self) -> &str {
        match self {
            Self::Text(t) => t,
            Self::Parts(parts) => parts
                .first()
                .and_then(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                })
                .unwrap_or(""),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
}

/// A tool-call request emitted by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // always "function"
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// Raw JSON string — we let each tool unmarshal its own input.
    pub arguments: String,
}

// ── tool definition (sent to the model) ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinitionParam {
    #[serde(rename = "type")]
    pub kind: String, // always "function"
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema object
}

// ── request / response ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDefinitionParam]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
    max_tokens: u32,
    temperature: f32,
    /// When true, the gateway streams the response as SSE `data:` chunks.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AssistantMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ── streaming chunk types ─────────────────────────────────────────────────────
//
// In stream mode the gateway emits SSE frames `data: {chunk}` where each chunk
// carries a partial `delta`.  Text arrives token-by-token in `delta.content`;
// tool calls arrive incrementally, keyed by `index`, with `function.arguments`
// fragments that must be concatenated.

#[derive(Debug, Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: Delta,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ── client ────────────────────────────────────────────────────────────────────

/// Configuration for the LiteLLM gateway.
#[derive(Debug, Clone)]
pub struct LiteLlmConfig {
    /// e.g. `http://litellm:4000`
    pub base_url: String,
    /// Master key set in `litellm_config.yaml`
    pub api_key: String,
    /// Default model to use, e.g. `claude-3-7-sonnet`
    pub default_model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl LiteLlmConfig {
    /// Load configuration from environment variables.
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            base_url: std::env::var("LITELLM_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:4000".into()),
            api_key: std::env::var("LITELLM_API_KEY")
                .unwrap_or_else(|_| "sk-agentx-dev".into()),
            default_model: std::env::var("AGENT_MODEL")
                .unwrap_or_else(|_| "claude-3-7-sonnet-20250219".into()),
            max_tokens: std::env::var("AGENT_MAX_TOKENS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8192),
            temperature: std::env::var("AGENT_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
        })
    }
}

/// Thin async wrapper around the LiteLLM `/chat/completions` endpoint.
#[derive(Debug, Clone)]
pub struct LiteLlmClient {
    http: Client,
    config: LiteLlmConfig,
}

impl LiteLlmClient {
    pub fn new(config: LiteLlmConfig) -> Self {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");
        Self { http, config }
    }

    /// Send a chat-completion request, streaming the response token-by-token.
    ///
    /// `on_text` is invoked with each text delta as it arrives.  The fully
    /// assembled [`AssistantMessage`] (content + any tool calls) is returned
    /// once the stream completes, so all downstream logic is identical to the
    /// non-streaming path.
    #[instrument(skip(self, messages, tools, on_text), fields(model = %self.config.default_model))]
    pub async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinitionParam]>,
        mut on_text: impl FnMut(&str),
    ) -> Result<AssistantMessage> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let body = ChatRequest {
            model: &self.config.default_model,
            messages,
            tools,
            tool_choice: tools.filter(|t| !t.is_empty()).map(|_| "auto"),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            stream: true,
        };

        debug!(messages = messages.len(), "sending streaming chat request");

        let req = self.http.post(&url);
        let req = if self.config.api_key.is_empty() {
            req
        } else {
            req.bearer_auth(&self.config.api_key)
        };
        let resp = req
            .json(&body)
            .send()
            .await
            .context("HTTP request to LiteLLM failed")
            .map_err(|e| AgentError::Inference(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AgentError::Inference(format!(
                "LiteLLM returned {status}: {text}"
            )));
        }

        let mut acc = StreamAccumulator::default();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk
                .context("error reading streaming response body")
                .map_err(|e| AgentError::Inference(e.to_string()))?;
            acc.feed(&String::from_utf8_lossy(&chunk), &mut on_text);
        }

        Ok(acc.finish())
    }
}

/// Reassembles an OpenAI-style streaming response from raw SSE byte chunks.
///
/// Network chunks don't align to SSE event boundaries, so we buffer and split
/// on `\n\n`.  Text deltas are concatenated; tool calls are accumulated by
/// their `index`, with `arguments` fragments joined into the full JSON string.
#[derive(Default)]
struct StreamAccumulator {
    buf: String,
    content: String,
    tool_acc: Vec<(String, String, String)>, // (id, name, args)
}

impl StreamAccumulator {
    /// Feed one network chunk; invoke `on_text` for each new text delta.
    fn feed(&mut self, bytes: &str, on_text: &mut impl FnMut(&str)) {
        self.buf.push_str(bytes);

        while let Some(pos) = self.buf.find("\n\n") {
            let event: String = self.buf.drain(..pos + 2).collect();
            for line in event.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    break;
                }
                let parsed: ChatChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(_) => continue, // skip keep-alives / non-JSON frames
                };
                let Some(choice) = parsed.choices.into_iter().next() else {
                    continue;
                };
                if let Some(text) = choice.delta.content {
                    if !text.is_empty() {
                        on_text(&text);
                        self.content.push_str(&text);
                    }
                }
                for tc in choice.delta.tool_calls.into_iter().flatten() {
                    if self.tool_acc.len() <= tc.index {
                        self.tool_acc.resize(tc.index + 1, Default::default());
                    }
                    let slot = &mut self.tool_acc[tc.index];
                    if let Some(id) = tc.id {
                        slot.0 = id;
                    }
                    if let Some(f) = tc.function {
                        if let Some(name) = f.name {
                            slot.1 = name;
                        }
                        if let Some(args) = f.arguments {
                            slot.2.push_str(&args);
                        }
                    }
                }
            }
        }
    }

    fn finish(self) -> AssistantMessage {
        let tool_calls: Vec<ToolCall> = self
            .tool_acc
            .into_iter()
            .filter(|(_, name, _)| !name.is_empty())
            .map(|(id, name, arguments)| ToolCall {
                id,
                kind: "function".into(),
                function: FunctionCall { name, arguments },
            })
            .collect();

        AssistantMessage {
            role: "assistant".into(),
            content: (!self.content.is_empty()).then_some(self.content),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_split_frames_and_tool_args() {
        let mut acc = StreamAccumulator::default();
        let mut text = String::new();
        let mut sink = |s: &str| text.push_str(s);

        // Stream split mid-frame across chunk boundaries, with a tool call whose
        // JSON arguments arrive in fragments and a trailing keep-alive + [DONE].
        let chunks = [
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel",
            "lo\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"p\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"x\\\"}\"}}]}}]}\n\n",
            ": keep-alive\n\ndata: [DONE]\n\n",
        ];
        for c in chunks {
            acc.feed(c, &mut sink);
        }

        assert_eq!(text, "Hello world");
        let msg = acc.finish();
        assert_eq!(msg.content.as_deref(), Some("Hello world"));
        let tcs = msg.tool_calls.expect("tool calls");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "call_1");
        assert_eq!(tcs[0].function.name, "read");
        assert_eq!(tcs[0].function.arguments, r#"{"p":"x"}"#);
    }
}
