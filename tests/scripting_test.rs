//! Tests for dynamic extensions: YAML config, Markdown skills, Rhai commands.
//!
//! Each test writes a throwaway `agentx.d/` under a TempDir and points the
//! loader at it, exercising the real on-disk load path.

use std::fs;
use std::path::Path;

use agentx::scripting::{load_commands, load_config, load_skills, Extensions};
use serde_json::json;
use tempfile::TempDir;

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

#[test]
fn config_yaml_overrides_defaults() {
    let home = TempDir::new().unwrap();
    write(
        &home.path().join("config.yaml"),
        "system_prompt: \"You are a test bot.\"\nmax_iterations: 7\n",
    );
    let cfg = load_config(home.path());
    assert_eq!(cfg.system_prompt.as_deref(), Some("You are a test bot."));
    assert_eq!(cfg.max_iterations, Some(7));
}

#[test]
fn missing_config_yields_defaults() {
    let home = TempDir::new().unwrap();
    let cfg = load_config(home.path());
    assert!(cfg.system_prompt.is_none());
    assert!(cfg.max_iterations.is_none());
}

#[test]
fn skills_concatenated_in_filename_order() {
    let home = TempDir::new().unwrap();
    write(&home.path().join("skills/02-second.md"), "Second skill.");
    write(&home.path().join("skills/01-first.md"), "First skill.");
    let skills = load_skills(home.path());
    // Sorted by filename → first, then second.
    assert_eq!(skills, "First skill.\n\nSecond skill.");
}

#[tokio::test]
async fn rhai_command_loads_and_executes() {
    let home = TempDir::new().unwrap();
    write(
        &home.path().join("commands/greet.rhai"),
        r#"
fn meta() {
    #{ description: "Greets a person by name.",
       parameters: #{ type: "object", properties: #{ name: #{ type: "string" } } } }
}
fn run(input) {
    "Hello, " + input.name + "!"
}
"#,
    );

    let workspace = TempDir::new().unwrap();
    let tools = load_commands(home.path(), workspace.path().to_path_buf());
    assert_eq!(tools.len(), 1);

    let tool = &tools[0];
    assert_eq!(tool.name(), "greet");
    assert_eq!(tool.description(), "Greets a person by name.");
    assert_eq!(tool.parameters_schema()["type"], "object");

    let out = tool.execute(json!({ "name": "Ada" })).await.unwrap();
    assert_eq!(out, "Hello, Ada!");
}

#[tokio::test]
async fn rhai_command_host_functions_touch_workspace() {
    let home = TempDir::new().unwrap();
    write(
        &home.path().join("commands/save.rhai"),
        r#"
fn run(input) {
    write_file(input.path, input.text);
    read_file(input.path)
}
"#,
    );

    let workspace = TempDir::new().unwrap();
    let tools = load_commands(home.path(), workspace.path().to_path_buf());
    let out = tools[0]
        .execute(json!({ "path": "note.txt", "text": "scripted write" }))
        .await
        .unwrap();
    assert_eq!(out, "scripted write");
    // The file really landed in the workspace.
    let on_disk = fs::read_to_string(workspace.path().join("note.txt")).unwrap();
    assert_eq!(on_disk, "scripted write");
}

#[test]
fn broken_command_is_skipped_not_fatal() {
    let home = TempDir::new().unwrap();
    write(&home.path().join("commands/ok.rhai"), "fn run(input) { \"fine\" }");
    write(&home.path().join("commands/broken.rhai"), "fn run(input { syntax error");
    let workspace = TempDir::new().unwrap();
    let tools = load_commands(home.path(), workspace.path().to_path_buf());
    // Only the valid command survives; the broken one is logged and skipped.
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "ok");
}

#[test]
fn extensions_merge_skills_into_system_prompt() {
    let home = TempDir::new().unwrap();
    write(&home.path().join("config.yaml"), "system_prompt: \"Base.\"\n");
    write(&home.path().join("skills/a.md"), "Extra skill.");

    let ext = Extensions {
        config: load_config(home.path()),
        skills: load_skills(home.path()),
        commands: vec![],
    };
    let agent_cfg = ext.agent_config();
    assert!(agent_cfg.system_prompt.starts_with("Base."));
    assert!(agent_cfg.system_prompt.contains("# Skills"));
    assert!(agent_cfg.system_prompt.contains("Extra skill."));
}
