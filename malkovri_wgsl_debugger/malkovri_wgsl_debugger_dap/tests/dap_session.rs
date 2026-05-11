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

fn launch_and_configure(session: &mut Session, shader: &str, breakpoints: &[u32]) -> Vec<Value> {
    session.send("initialize", json!({}));
    session.send(
        "launch",
        json!({
            "program": shader,
        }),
    );
    session.send("setBreakpoints", json!({
        "source": { "name": PathBuf::from(shader).file_name().unwrap().to_string_lossy(), "path": shader },
        "breakpoints": breakpoints.iter().map(|line| json!({ "line": line })).collect::<Vec<_>>(),
    }));
    session.send("configurationDone", json!({}))
}

#[test]
fn initialize_advertises_only_implemented_capabilities() {
    let mut s = Session::new();
    let init = s.send("initialize", json!({}));
    let body = response_body(&init, s.last_seq());

    assert_eq!(body["supportsConfigurationDoneRequest"], true);
    assert_eq!(body["supportsSingleThreadExecutionRequests"], true);
    assert_eq!(body["supportsTerminateRequest"], true);

    let unsupported = [
        "supportsCancelRequest",
        "supportsConditionalBreakpoints",
        "supportsExceptionInfoRequest",
        "supportsHitConditionalBreakpoints",
        "supportsRestartRequest",
        "supportsSetVariable",
    ];
    for capability in unsupported {
        assert!(
            body.get(capability).is_none(),
            "{capability} should not be advertised"
        );
    }
}

#[test]
fn control_flow_session_matches_vscode_request_flow() {
    let mut s = Session::new();
    let shader = shader_path("test_control_flow.wgsl");

    let init = s.send("initialize", json!({}));
    assert_eq!(
        response_body(&init, 1)["supportsConfigurationDoneRequest"],
        true
    );
    assert_eq!(event_body(&init, "initialized"), &json!({}));

    let launch = s.send(
        "launch",
        json!({
            "program": shader,
            "stopOnEntry": true,
            "workgroupConfig": { "workgroupId": [5, 0, 0] },
        }),
    );
    assert!(find_response(&launch, s.last_seq()).is_none());

    let bp = s.send(
        "setBreakpoints",
        json!({
            "source": { "name": "test_control_flow.wgsl", "path": shader },
            "breakpoints": [{ "line": 24 }],
        }),
    );
    assert_eq!(response_body(&bp, s.last_seq())[0]["verified"], true);

    let cfg = s.send("configurationDone", json!({}));
    let cfg_seq = s.last_seq();
    assert!(find_response(&cfg, cfg_seq).is_some());
    assert!(find_response(&cfg, 2).is_some()); // delayed launch response
    assert_eq!(event_body(&cfg, "stopped")["reason"], "entry");

    let threads = s.send("threads", json!({}));
    assert_eq!(
        response_body(&threads, s.last_seq())["threads"][0]["name"],
        "[5, 0, 0]"
    );

    // After skip_emits at entry, the first visible line is the for loop (line 9),
    // past the let/var declarations which are Emit/init-only.
    let stack = s.send("stackTrace", json!({ "threadId": 1 }));
    let frames = &response_body(&stack, s.last_seq())["stackFrames"];
    assert_eq!(frames[0]["source"]["path"], json!(shader));

    let source = s.send(
        "source",
        json!({
            "source": { "path": shader },
            "sourceReference": 0,
        }),
    );
    assert!(
        response_body(&source, s.last_seq())["content"]
            .as_str()
            .unwrap()
            .contains("var count = idx % 16u;")
    );

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
    s.send(
        "launch",
        json!({
            "program": shader,
            "workgroupConfig": { "workgroupId": [2, 0, 0] },
            "bindings": {
                "0:0": { "type": "f32", "inline": [1.0, 2.0, 3.0, 4.0] },
                "0:1": { "type": "u32", "inline": [0, 0, 0, 0] },
            },
        }),
    );
    s.send(
        "setBreakpoints",
        json!({
            "source": { "name": "test_expressions.wgsl", "path": shader },
            "breakpoints": [{ "line": 457 }],
        }),
    );
    let cfg = s.send("configurationDone", json!({}));

    // Run to the breakpoint so all variables are in scope.
    assert_eq!(event_body(&cfg, "stopped")["reason"], "breakpoint");

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

    let cont = s.send("continue", json!({ "threadId": 1 }));
    assert_eq!(event_body(&cont, "terminated"), &json!({}));
    assert!(
        cont.iter().all(|message| message["event"] != "output"),
        "continuing through storage-buffer stores should not report evaluator errors: {cont:?}"
    );
}

#[test]
fn local_variables_initialize_at_declaration_time() {
    let mut s = Session::new();
    let shader = shader_path("test_local_initialization_timing.wgsl");

    let cfg = launch_and_configure(&mut s, &shader, &[10]);
    assert_eq!(event_body(&cfg, "stopped")["reason"], "breakpoint");

    let scopes = s.send("scopes", json!({ "frameId": 1 }));
    let locals_ref = scope_reference(response_body(&scopes, s.last_seq()), "Locals");

    let locals = s.send("variables", json!({ "variablesReference": locals_ref }));
    let locals = variables_map(response_body(&locals, s.last_seq()));

    assert_eq!(locals["x"], "Primitive(U32(5))");
    assert_eq!(locals["y"], "Primitive(U32(5))");
    assert_eq!(locals["stop_here"], "Primitive(U32(5))");
}

#[test]
fn loop_local_variables_reinitialize_when_the_loop_body_reenters() {
    let mut s = Session::new();
    let shader = shader_path("test_loop_local_initialization_timing.wgsl");

    let cfg = launch_and_configure(&mut s, &shader, &[12]);
    assert_eq!(event_body(&cfg, "stopped")["reason"], "breakpoint");

    let scopes = s.send("scopes", json!({ "frameId": 1 }));
    let locals_ref = scope_reference(response_body(&scopes, s.last_seq()), "Locals");

    let locals = s.send("variables", json!({ "variablesReference": locals_ref }));
    let locals = variables_map(response_body(&locals, s.last_seq()));

    assert_eq!(locals["i"], "Primitive(U32(1))");
    assert_eq!(locals["snapshot"], "Primitive(U32(1))");
    assert_eq!(locals["acc"], "Primitive(U32(2))");
}

#[test]
fn nested_named_expressions_stay_visible_inside_their_block() {
    let mut s = Session::new();
    let shader = shader_path("test_nested_named_expression_scope.wgsl");

    s.send("initialize", json!({}));
    s.send(
        "launch",
        json!({
            "program": shader,
            "workgroupConfig": { "workgroupId": [5, 0, 0] },
        }),
    );
    s.send("setBreakpoints", json!({
        "source": { "name": PathBuf::from(&shader).file_name().unwrap().to_string_lossy(), "path": shader },
        "breakpoints": [{ "line": 9 }],
    }));
    let cfg = s.send("configurationDone", json!({}));
    assert_eq!(event_body(&cfg, "stopped")["reason"], "breakpoint");

    let scopes = s.send("scopes", json!({ "frameId": 1 }));
    let locals_ref = scope_reference(response_body(&scopes, s.last_seq()), "Locals");

    let locals = s.send("variables", json!({ "variablesReference": locals_ref }));
    let locals = variables_map(response_body(&locals, s.last_seq()));

    assert!(
        locals.contains_key("outer"),
        "expected outer let binding `outer` to stay visible inside nested block, got {locals:?}"
    );
    assert_eq!(locals["outer"], "Primitive(U32(5))");
    assert!(
        locals.contains_key("stop_here"),
        "expected nested let binding `stop_here` to stay visible inside its block, got {locals:?}"
    );
    assert_eq!(locals["stop_here"], "Primitive(U32(6))");
}

#[test]
fn workgroup_threads_route_stack_and_variables_by_thread_id() {
    let mut s = Session::new();
    let shader = shader_path("test_nested_named_expression_scope.wgsl");

    s.send("initialize", json!({}));
    s.send(
        "launch",
        json!({
            "program": shader,
            "stopOnEntry": true,
            "workgroupConfig": { "workgroupSize": [2, 1, 1] },
        }),
    );
    s.send("configurationDone", json!({}));

    let threads = s.send("threads", json!({}));
    let threads_body = response_body(&threads, s.last_seq());
    assert_eq!(threads_body["threads"][0]["name"], "[0, 0, 0]");
    assert_eq!(threads_body["threads"][1]["name"], "[1, 0, 0]");

    let stack = s.send("stackTrace", json!({ "threadId": 2 }));
    let frame_id = response_body(&stack, s.last_seq())["stackFrames"][0]["id"]
        .as_u64()
        .unwrap();
    assert_eq!(frame_id, 2);

    let scopes = s.send("scopes", json!({ "frameId": frame_id }));
    let arguments_ref = scope_reference(response_body(&scopes, s.last_seq()), "Function Arguments");
    let args = s.send("variables", json!({ "variablesReference": arguments_ref }));
    let args = variables_map(response_body(&args, s.last_seq()));
    assert_eq!(args["global_id"], "Primitive(U32x3([1, 0, 0]))");
}

#[test]
fn single_thread_execution_launch_mode_steps_only_selected_thread() {
    let mut s = Session::new();
    let shader = shader_path("test_nested_named_expression_scope.wgsl");

    s.send("initialize", json!({}));
    s.send(
        "launch",
        json!({
            "program": shader,
            "stopOnEntry": true,
            "singleThreadExecution": true,
            "workgroupConfig": { "workgroupSize": [2, 1, 1] },
        }),
    );
    let cfg = s.send("configurationDone", json!({}));
    assert_eq!(event_body(&cfg, "stopped")["reason"], "entry");

    let next = s.send("next", json!({ "threadId": 2, "singleThread": false }));
    assert_eq!(event_body(&next, "stopped")["reason"], "step");
    assert_eq!(event_body(&next, "stopped")["threadId"], 2);

    let thread_2_stack = s.send("stackTrace", json!({ "threadId": 2 }));
    let thread_2_line = response_body(&thread_2_stack, s.last_seq())["stackFrames"][0]["line"]
        .as_u64()
        .unwrap();
    let thread_1_stack = s.send("stackTrace", json!({ "threadId": 1 }));
    let thread_1_line = response_body(&thread_1_stack, s.last_seq())["stackFrames"][0]["line"]
        .as_u64()
        .unwrap();

    assert_ne!(
        thread_1_line, thread_2_line,
        "thread 1 should stay at entry while thread 2 steps"
    );
}

#[test]
fn private_globals_are_initialized_and_mutable() {
    let mut s = Session::new();
    let shader = shader_path("test_private_global.wgsl");

    let cfg = launch_and_configure(&mut s, &shader, &[8]);
    assert_eq!(event_body(&cfg, "stopped")["reason"], "breakpoint");

    let scopes = s.send("scopes", json!({ "frameId": 1 }));
    let scopes_body = response_body(&scopes, s.last_seq());
    let locals_ref = scope_reference(scopes_body, "Locals");
    let globals_ref = scope_reference(scopes_body, "Globals");

    let locals = s.send("variables", json!({ "variablesReference": locals_ref }));
    let locals = variables_map(response_body(&locals, s.last_seq()));
    assert_eq!(locals["stop_here"], "Primitive(U32(8))");

    let globals = s.send("variables", json!({ "variablesReference": globals_ref }));
    let globals = variables_map(response_body(&globals, s.last_seq()));
    assert_eq!(globals["counter"], "Primitive(U32(8))");
}

#[test]
fn pointer_arguments_write_through_places_and_zero_nested_values() {
    let mut s = Session::new();
    let shader = shader_path("test_pointer_places.wgsl");

    let cfg = launch_and_configure(&mut s, &shader, &[27]);
    assert_eq!(event_body(&cfg, "stopped")["reason"], "breakpoint");

    let scopes = s.send("scopes", json!({ "frameId": 1 }));
    let scopes_body = response_body(&scopes, s.last_seq());
    let locals_ref = scope_reference(scopes_body, "Locals");
    let globals_ref = scope_reference(scopes_body, "Globals");

    let locals = s.send("variables", json!({ "variablesReference": locals_ref }));
    let locals = variables_map(response_body(&locals, s.last_seq()));
    assert_eq!(locals["x"], "Primitive(U32(5))");
    assert_eq!(locals["stop_here"], "Primitive(U32(21))");
    assert_eq!(
        locals["s"],
        "Struct([(\"value\", Primitive(U32(7))), (\"more\", Array([Primitive(U32(0)), Primitive(U32(9))]))])"
    );

    let cont = s.send("continue", json!({ "threadId": 1 }));
    assert_eq!(event_body(&cont, "terminated"), &json!({}));

    let globals = s.send("variables", json!({ "variablesReference": globals_ref }));
    let globals = variables_map(response_body(&globals, s.last_seq()));
    assert_eq!(globals["sink"], "Primitive(U32(21))");
}
