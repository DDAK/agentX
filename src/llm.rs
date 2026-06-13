/// LiteLLM gateway client.
///
/// LiteLLM exposes an OpenAI-compatible `/chat/completions` endpoint, so we
/// speak plain JSON rather than pulling in a heavy SDK crate.  This keeps the
/// dependency tree small and makes the serialisation logic trivial to follow.
use anyhow::Context;
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
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct Choice {
    pub message: AssistantMessage,
    pub finish_reason: Option<String>,
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

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
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

    /// Send a chat-completion request and return the model's response.
    ///
    /// `tools` is `None` when we don't want tool-use in a particular call.
    #[instrument(skip(self, messages, tools), fields(model = %self.config.default_model))]
    pub async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinitionParam]>,
    ) -> Result<AssistantMessage> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let body = ChatRequest {
            model: &self.config.default_model,
            messages,
            tools,
            tool_choice: tools.filter(|t| !t.is_empty()).map(|_| "auto"),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
        };

        debug!(messages = messages.len(), "sending chat request");

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

        let chat_resp: ChatResponse = resp
            .json()
            .await
            .context("failed to deserialize LiteLLM response")
            .map_err(|e| AgentError::Inference(e.to_string()))?;

        if let Some(usage) = &chat_resp.usage {
            debug!(
                prompt = usage.prompt_tokens,
                completion = usage.completion_tokens,
                total = usage.total_tokens,
                "token usage"
            );
        }

        chat_resp
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| AgentError::Inference("empty choices in response".into()))
    }
}
