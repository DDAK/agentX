/// Tests for the hook system.
///
/// Covers:
/// - HookChain fires all hooks in order, stops at first Abort
/// - LoggingHook always continues
/// - ToolAnnouncerHook always continues
/// - ConfirmCommandHook::auto_reject() rejects run_command, passes others
/// - ConfirmCommandHook::custom() with allow/deny policies
/// - Non-run_command tools pass through ConfirmCommandHook
///
/// Run with:  cargo test --test hooks_test
use agentx::hooks::{
    ConfirmCommandHook, Hook, HookChain, HookEvent, HookResult, LoggingHook, ToolAnnouncerHook,
};
use serde_json::json;

// ── helpers ───────────────────────────────────────────────────────────────────

fn tool_event(tool_name: &str) -> HookEvent {
    HookEvent::BeforeToolExecution {
        tool_name: tool_name.to_owned(),
        input: json!({ "command": tool_name }),
    }
}

fn run_command_event(cmd: &str) -> HookEvent {
    HookEvent::BeforeToolExecution {
        tool_name: "run_command".to_owned(),
        input: json!({ "command": cmd }),
    }
}

fn is_continue(r: &HookResult) -> bool {
    matches!(r, HookResult::Continue)
}

fn is_abort(r: &HookResult) -> bool {
    matches!(r, HookResult::Abort(_))
}

// ── LoggingHook ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_logging_hook_always_continues() {
    let hook = LoggingHook;
    let events = vec![
        HookEvent::TurnStart { turn: 1 },
        HookEvent::TurnEnd { turn: 1 },
        HookEvent::BeforeInference { message_count: 3 },
        HookEvent::AfterInference { content: Some("hi".into()), tool_call_count: 0 },
        tool_event("read_file"),
        HookEvent::AfterToolExecution { tool_name: "read_file".into(), result: "ok".into() },
    ];
    for event in &events {
        assert!(is_continue(&hook.on_event(event).await), "LoggingHook should always Continue");
    }
}

// ── ToolAnnouncerHook ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_tool_announcer_always_continues() {
    let hook = ToolAnnouncerHook;
    assert!(is_continue(&hook.on_event(&tool_event("write_file")).await));
    assert!(is_continue(&hook.on_event(&tool_event("run_command")).await));
    assert!(is_continue(&hook.on_event(&HookEvent::TurnStart { turn: 1 }).await));
}

// ── ConfirmCommandHook: auto_reject ───────────────────────────────────────────

#[tokio::test]
async fn test_confirm_hook_auto_reject_blocks_run_command() {
    let hook = ConfirmCommandHook::auto_reject();
    let result = hook.on_event(&run_command_event("rm -rf /")).await;
    assert!(is_abort(&result));
    if let HookResult::Abort(msg) = result {
        assert!(msg.contains("rm -rf /"));
    }
}

#[tokio::test]
async fn test_confirm_hook_auto_reject_passes_other_tools() {
    let hook = ConfirmCommandHook::auto_reject();
    // Non-run_command tools should not be blocked.
    for tool in &["read_file", "write_file", "edit_file", "list_files"] {
        let result = hook.on_event(&tool_event(tool)).await;
        assert!(is_continue(&result), "{tool} should not be blocked by ConfirmCommandHook");
    }
}

#[tokio::test]
async fn test_confirm_hook_auto_reject_passes_non_tool_events() {
    let hook = ConfirmCommandHook::auto_reject();
    assert!(is_continue(&hook.on_event(&HookEvent::TurnStart { turn: 1 }).await));
    assert!(is_continue(&hook.on_event(&HookEvent::BeforeInference { message_count: 1 }).await));
}

// ── ConfirmCommandHook: custom ────────────────────────────────────────────────

#[tokio::test]
async fn test_confirm_hook_custom_allowlist() {
    // Only allow "echo hello"
    let hook = ConfirmCommandHook::custom(|cmd| cmd == "echo hello");

    let allowed = hook.on_event(&run_command_event("echo hello")).await;
    assert!(is_continue(&allowed));

    let denied = hook.on_event(&run_command_event("rm -rf /")).await;
    assert!(is_abort(&denied));
}

#[tokio::test]
async fn test_confirm_hook_custom_always_allow() {
    let hook = ConfirmCommandHook::custom(|_| true);
    assert!(is_continue(&hook.on_event(&run_command_event("anything")).await));
}

// ── HookChain ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_hook_chain_empty_continues() {
    let chain = HookChain::new();
    assert!(is_continue(&chain.fire(&HookEvent::TurnStart { turn: 1 }).await));
}

#[tokio::test]
async fn test_hook_chain_all_continue() {
    let mut chain = HookChain::new();
    chain.add(LoggingHook);
    chain.add(ToolAnnouncerHook);
    let result = chain.fire(&tool_event("read_file")).await;
    assert!(is_continue(&result));
}

#[tokio::test]
async fn test_hook_chain_stops_at_first_abort() {
    let mut chain = HookChain::new();
    chain.add(LoggingHook);                          // continue
    chain.add(ConfirmCommandHook::auto_reject());    // abort here
    chain.add(ToolAnnouncerHook);                    // must not be reached

    let result = chain.fire(&run_command_event("ls")).await;
    assert!(is_abort(&result));
}

#[tokio::test]
async fn test_hook_chain_abort_message_propagates() {
    let mut chain = HookChain::new();
    chain.add(ConfirmCommandHook::custom(|cmd| {
        if cmd == "safe" { true } else { false }
    }));

    let ok = chain.fire(&run_command_event("safe")).await;
    assert!(is_continue(&ok));

    let err = chain.fire(&run_command_event("dangerous")).await;
    if let HookResult::Abort(msg) = err {
        assert!(msg.contains("dangerous"));
    } else {
        panic!("expected Abort");
    }
}

#[tokio::test]
async fn test_hook_chain_multiple_hooks_all_fire_on_continue() {
    use std::sync::{Arc, Mutex};
    use agentx::hooks::Hook;

    // A counting hook that records which events it saw.
    struct Counter(Arc<Mutex<Vec<String>>>);
    #[async_trait::async_trait]
    impl Hook for Counter {
        async fn on_event(&self, event: &HookEvent) -> HookResult {
            let label = format!("{event:?}");
            self.0.lock().unwrap().push(label);
            HookResult::Continue
        }
    }

    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let mut chain = HookChain::new();
    chain.add(Counter(Arc::clone(&log)));
    chain.add(Counter(Arc::clone(&log)));
    chain.add(Counter(Arc::clone(&log)));

    chain.fire(&HookEvent::TurnStart { turn: 42 }).await;

    // All three hooks should have fired once.
    assert_eq!(log.lock().unwrap().len(), 3);
}
