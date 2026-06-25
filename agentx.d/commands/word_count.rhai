// Example scripted command. Drop a .rhai file in this directory and it becomes
// a tool the agent can call — no recompile.
//
// `meta()` (optional) describes the tool to the model.
// `run(input)` is the entry point; `input` is a map of the model's arguments,
// and the return value (stringified) is the tool result.
//
// Host functions available: read_file(path), write_file(path, content),
// list_files(path), sh(cmd) — all scoped to the workspace directory.

fn meta() {
    #{
        description: "Count the words in a workspace file.",
        parameters: #{
            type: "object",
            properties: #{
                path: #{ type: "string", description: "Relative path to the file." }
            },
            required: ["path"]
        }
    }
}

fn run(input) {
    let text = read_file(input.path);
    if text == "" {
        return "File is empty or not found: " + input.path;
    }
    let words = text.split(' ').len;
    "" + words + " words in " + input.path
}
