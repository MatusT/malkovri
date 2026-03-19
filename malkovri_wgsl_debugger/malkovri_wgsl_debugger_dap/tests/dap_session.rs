use std::{collections::HashMap, path::PathBuf};

use malkovri_wgsl_debugger_dap::DebugAdapter;
use serde_json::{Value, json};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn shader_path(file_name: &str) -> String {
    workspace_root()
        .join("test_shaders")
        .join(file_name)
        .to_string_lossy()
        .to_string()
}

struct Session {
    adapter: DebugAdapter,
    seq: i64,
}

impl Session {
    fn new() -> Self {
        Self {
            adapter: DebugAdapter::new(),
            seq: 0,
        }
    }

    fn send(&mut self, command: &str, arguments: Value) -> Vec<Value> {
        self.seq += 1;
        let request = json!({
            "seq": self.seq,
            "type": "request",
            "command": command,
            "arguments": arguments,
        });
        self.adapter
            .handle_message(&request.to_string())
            .unwrap()
            .into_iter()
            .map(|s| serde_json::from_str(&s).unwrap())
            .collect()
    }

    fn last_seq(&self) -> i64 {
        self.seq
    }
}

fn find_response(messages: &[Value], request_seq: i64) -> Option<&Value> {
    messages.iter().find(|m| m["request_seq"] == request_seq)
}

fn response_body(messages: &[Value], request_seq: i64) -> &Value {
    &find_response(messages, request_seq)
        .unwrap_or_else(|| panic!("missing response for request {request_seq}"))["body"]
}

fn event_body<'a>(messages: &'a [Value], event_name: &str) -> &'a Value {
    &messages
        .iter()
        .find(|m| m["event"] == event_name)
        .unwrap_or_else(|| panic!("missing event {event_name}"))["body"]
}

fn scope_reference(scopes_body: &Value, scope_name: &str) -> u32 {
    scopes_body["scopes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|scope| scope["name"] == scope_name)
        .unwrap_or_else(|| panic!("missing scope {scope_name}"))["variablesReference"]
        .as_u64()
        .unwrap() as u32
}

fn variables_map(variables_body: &Value) -> HashMap<String, String> {
    variables_body["variables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v["name"].as_str().unwrap().to_string(),
                v["value"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

#[test]
fn control_flow_session_matches_vscode_request_flow() {
    let mut s = Session::new();
    let shader = shader_path("test_control_flow.wgsl");

    let init = s.send("initialize", json!({}));
    assert_eq!(response_body(&init, 1)["supportsConfigurationDoneRequest"], true);
    assert_eq!(event_body(&init, "initialized"), &json!({}));

    let launch = s.send("launch", json!({
        "program": shader,
        "shaderInputs": { "global_invocation_id": [5, 0, 0] },
    }));
    assert!(find_response(&launch, s.last_seq()).is_none());
    assert_eq!(event_body(&launch, "stopped")["reason"], "entry");

    let bp = s.send("setBreakpoints", json!({
        "source": { "name": "test_control_flow.wgsl", "path": shader },
        "breakpoints": [{ "line": 24 }],
    }));
    assert_eq!(response_body(&bp, s.last_seq())[0]["verified"], true);

    let cfg = s.send("configurationDone", json!({}));
    let cfg_seq = s.last_seq();
    assert!(find_response(&cfg, cfg_seq).is_some());
    assert!(find_response(&cfg, 2).is_some()); // delayed launch response
    assert_eq!(event_body(&cfg, "stopped")["reason"], "entry");

    let threads = s.send("threads", json!({}));
    assert_eq!(response_body(&threads, s.last_seq())["threads"][0]["name"], "Main Thread");

    // After skip_emits at entry, the first visible line is the for loop (line 9),
    // past the let/var declarations which are Emit/init-only.
    let stack = s.send("stackTrace", json!({ "threadId": 1 }));
    let frames = &response_body(&stack, s.last_seq())["stackFrames"];
    assert_eq!(frames[0]["source"]["path"], json!(shader));

    let source = s.send("source", json!({
        "source": { "path": shader },
        "sourceReference": 0,
    }));
    assert!(response_body(&source, s.last_seq())["content"]
        .as_str()
        .unwrap()
        .contains("var count = idx % 16u;"));

    let scopes = s.send("scopes", json!({ "frameId": 1 }));
    let scopes_body = response_body(&scopes, s.last_seq());
    let locals_ref = scope_reference(scopes_body, "Locals");
    let arguments_ref = scope_reference(scopes_body, "Function Arguments");

    let args = s.send("variables", json!({ "variablesReference": arguments_ref }));
    let args = variables_map(response_body(&args, s.last_seq()));
    assert_eq!(args["global_id"], "Primitive(U32x3([5, 0, 0]))");

    // At entry, named expression `idx` is visible (its Emit was skipped through).
    let locals = s.send("variables", json!({ "variablesReference": locals_ref }));
    let locals = variables_map(response_body(&locals, s.last_seq()));
    assert_eq!(locals["idx"], "Primitive(U32(5))");

    let next = s.send("next", json!({ "threadId": 1 }));
    assert!(find_response(&next, s.last_seq()).is_some());
    assert_eq!(event_body(&next, "stopped")["reason"], "step");

    let stack2 = s.send("stackTrace", json!({ "threadId": 1 }));
    assert!(response_body(&stack2, s.last_seq())["stackFrames"][0]["line"].is_number());
}

#[test]
fn expression_shader_variables_include_binding_backed_values() {
    let mut s = Session::new();
    let shader = shader_path("test_expressions.wgsl");

    s.send("initialize", json!({}));
    s.send("launch", json!({
        "program": shader,
        "shaderInputs": { "global_invocation_id": [2, 0, 0] },
        "bindings": {
            "0:0": { "type": "f32", "inline": [1.0, 2.0, 3.0, 4.0] },
            "0:1": { "type": "u32", "inline": [0, 0, 0, 0] },
        },
    }));
    s.send("setBreakpoints", json!({
        "source": { "name": "test_expressions.wgsl", "path": shader },
        "breakpoints": [{ "line": 457 }],
    }));
    s.send("configurationDone", json!({}));

    // Run to the breakpoint so all variables are in scope.
    let cont = s.send("continue", json!({ "threadId": 1 }));
    assert_eq!(event_body(&cont, "stopped")["reason"], "breakpoint");

    let scopes = s.send("scopes", json!({ "frameId": 1 }));
    let scopes_body = response_body(&scopes, s.last_seq());
    let locals_ref = scope_reference(scopes_body, "Locals");
    let arguments_ref = scope_reference(scopes_body, "Function Arguments");

    let args = s.send("variables", json!({ "variablesReference": arguments_ref }));
    let args = variables_map(response_body(&args, s.last_seq()));
    assert_eq!(args["gid"], "Primitive(U32x3([2, 0, 0]))");

    let locals = s.send("variables", json!({ "variablesReference": locals_ref }));
    let locals = variables_map(response_body(&locals, s.last_seq()));

    assert_eq!(locals["idx"], "Primitive(U32(2))");
    assert_eq!(locals["lv_f32"], "Primitive(F32(3.14))");
    assert_eq!(locals["lv_v3f"], "Primitive(F32x3([0.5, 1.5, 2.5]))");
    assert_eq!(locals["arr_len"], "Primitive(U32(4))");
    assert_eq!(locals["g_val"], "Primitive(F32(1.0))");
    assert_eq!(locals["acc_g"], "Primitive(F32(3.0))");
    assert_eq!(locals["dyn_i"], "Primitive(U32(2))");
    assert_eq!(locals["acc_vf"], "Primitive(F32(2.5))");
}
