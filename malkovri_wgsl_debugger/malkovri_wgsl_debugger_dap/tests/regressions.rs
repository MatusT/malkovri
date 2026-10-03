use malkovri_wgsl_debugger_dap::DebugAdapter;
use serde_json::{Value, json};

#[derive(Default)]
struct Session {
    adapter: DebugAdapter,
    seq: i64,
}

impl Session {
    fn send(&mut self, command: &str, arguments: Value) -> Vec<Value> {
        self.seq += 1;
        self.adapter
            .handle_message(
                &json!({
                    "seq": self.seq, "type": "request", "command": command, "arguments": arguments,
                })
                .to_string(),
            )
            .unwrap()
            .into_iter()
            .map(|message| serde_json::from_str(&message).unwrap())
            .collect()
    }

    fn request(&mut self, command: &str, arguments: Value) -> Value {
        let messages = self.send(command, arguments);
        let response = messages
            .into_iter()
            .find(|message| message["request_seq"] == self.seq)
            .expect("request must receive a response");
        assert_eq!(response["command"], command);
        response
    }

    fn launch_at(source: &str, marker: &str) -> Self {
        let mut session = Self::default();
        session.send("initialize", json!({}));
        session.send(
            "launch",
            json!({"program": "/shader.wgsl", "source": source}),
        );
        session.send(
            "setBreakpoints",
            json!({
                "source": {"path": "/shader.wgsl"}, "breakpoints": [{"line": line(source, marker)}],
            }),
        );
        let messages = session.send("configurationDone", json!({}));
        assert!(messages.iter().any(|message| message["event"] == "stopped"));
        session
    }

    fn frames(&mut self) -> Vec<Value> {
        self.request("stackTrace", json!({"threadId": 1}))["body"]["stackFrames"]
            .as_array()
            .unwrap()
            .clone()
    }

    fn variables(&mut self, frame: &Value, scope_name: &str) -> Vec<Value> {
        let response = self.request("scopes", json!({"frameId": frame["id"]}));
        let reference = response["body"]["scopes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|scope| scope["name"] == scope_name)
            .unwrap()["variablesReference"]
            .clone();
        self.request("variables", json!({"variablesReference": reference}))["body"]["variables"]
            .as_array()
            .unwrap()
            .clone()
    }
}

fn line(source: &str, marker: &str) -> usize {
    source
        .lines()
        .position(|line| line.contains(marker))
        .unwrap()
        + 1
}

const CALLS: &str = r#"var<private> sink: u32;
fn helper(value: u32) -> u32 {
    var helper_local = value;
    sink = helper_local; // inside
    return sink;
}
@compute @workgroup_size(1)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    var caller_local = 7u;
    let result = helper(caller_local + lid.x); // call site
    sink = result; // after call
}
"#;

#[test]
fn caller_frames_have_their_own_locations_arguments_and_locals() {
    let mut session = Session::launch_at(CALLS, "// inside");
    let frames = session.frames();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[1]["line"], line(CALLS, "// call site"));
    let caller_arguments = session.variables(&frames[1], "Function Arguments");
    assert_eq!(caller_arguments[0]["name"], "lid");
    let caller_locals = session.variables(&frames[1], "Locals");
    assert!(
        caller_locals
            .iter()
            .any(|variable| variable["name"] == "caller_local")
    );
    assert!(
        !caller_locals
            .iter()
            .any(|variable| variable["name"] == "helper_local" || variable["name"] == "result")
    );
    let callee_arguments = session.variables(&frames[0], "Function Arguments");
    assert_eq!(callee_arguments[0]["name"], "value");
}

#[test]
fn next_steps_over_a_call() {
    let mut session = Session::launch_at(CALLS, "// call site");
    session.send("next", json!({"threadId": 1}));
    let frames = session.frames();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0]["name"], "main");
    assert_eq!(frames[0]["line"], line(CALLS, "// after call"));
}

#[test]
fn resumed_frames_and_scopes_are_invalidated_without_ending_the_session() {
    let mut session = Session::launch_at(CALLS, "// call site");
    let frame = session.frames().remove(0);
    let scopes = session.request("scopes", json!({"frameId": frame["id"]}));
    let reference = scopes["body"]["scopes"][0]["variablesReference"].clone();
    session.send("next", json!({"threadId": 1}));
    assert_eq!(
        session.request("scopes", json!({"frameId": frame["id"]}))["success"],
        false
    );
    assert_eq!(
        session.request("variables", json!({"variablesReference": reference}))["success"],
        false
    );
    assert_eq!(session.request("threads", json!({}))["success"], true);
}

#[test]
fn unknown_requests_receive_failure_responses() {
    let mut session = Session::default();
    assert_eq!(
        session.request("unsupportedCommand", json!({}))["success"],
        false
    );
    assert_eq!(session.request("initialize", json!({}))["success"], true);
}

#[test]
fn invalid_shader_returns_a_failed_launch_response() {
    for source in ["this is invalid WGSL", ""] {
        let mut session = Session::default();
        let response = session.request(
            "launch",
            json!({"program": "/shader.wgsl", "source": source}),
        );
        assert_eq!(response["success"], false);
        assert!(!response["message"].as_str().unwrap().is_empty());
    }
}

#[test]
fn launch_rejects_invalid_binding_elements_with_their_location() {
    for (ty, value) in [
        ("u32", json!("oops")),
        ("u32", json!(-1)),
        ("u32", json!(4294967296u64)),
        ("i32", json!(2147483648u64)),
        ("i32", json!(1.5)),
        ("f32", json!(true)),
        ("f32", json!(1e100)),
    ] {
        let mut session = Session::default();
        let response = session.request("launch", json!({
            "program": "/shader.wgsl",
            "source": "@group(0) @binding(0) var<storage, read> data: array<u32>; @compute @workgroup_size(1) fn main() {}",
            "bindings": {"0:0": {"type": ty, "inline": [0, value]}},
        }));
        assert_eq!(response["success"], false, "{ty}: {value}");
        let message = response["message"].as_str().unwrap();
        assert!(
            message.contains("0:0") && message.contains("1"),
            "{message}"
        );
    }
}

#[test]
fn ron_bindings_reject_integer_overflow() {
    let mut session = Session::default();
    let response = session.request(
        "launch",
        json!({
            "program": "/shader.wgsl", "source": "@compute @workgroup_size(1) fn main() {}",
            "bindings": {"0:0": {"type": "u32", "fileContent": "[0, 4294967296]"}},
        }),
    );
    assert_eq!(response["success"], false);
    assert!(response["message"].as_str().unwrap().contains("element 1"));
}

#[test]
fn binary_bindings_preserve_values_and_reject_partial_elements() {
    let source = "@group(0) @binding(0) var<storage, read> data: array<u32>; @compute @workgroup_size(1) fn main() {}";
    for length in [1, 3, 5] {
        let mut session = Session::default();
        let response = session.request(
            "launch",
            json!({
                "program": "/shader.wgsl", "source": source,
                "bindings": {"0:0": {"type": "u32", "fileBytes": vec![0u8; length]}},
            }),
        );
        assert_eq!(response["success"], false);
        assert!(
            response["message"]
                .as_str()
                .unwrap()
                .contains("multiple of 4")
        );
    }
    let mut session = Session::default();
    session.send(
        "launch",
        json!({
            "program": "/shader.wgsl", "source": source, "stopOnEntry": true,
            "bindings": {"0:0": {"type": "u32", "fileBytes": [120, 86, 52, 18]}},
        }),
    );
    session.send("configurationDone", json!({}));
    let frame = session.frames().remove(0);
    let globals = session.variables(&frame, "Globals");
    assert_eq!(globals[0]["value"], "Array([Primitive(U32(305419896))])");
}

#[test]
fn stepping_over_a_call_still_honors_a_breakpoint_in_the_callee() {
    let mut session = Session::launch_at(CALLS, "// call site");
    session.send("setBreakpoints", json!({"source": {"path": "/shader.wgsl"}, "breakpoints": [{"line": line(CALLS, "// inside")}]}));
    let messages = session.send("next", json!({"threadId": 1}));
    assert!(
        messages
            .iter()
            .any(|message| message["event"] == "stopped"
                && message["body"]["reason"] == "breakpoint")
    );
    assert_eq!(session.frames()[0]["name"], "helper");
}

#[test]
fn inspection_keeps_a_saved_let_value_after_its_input_changes() {
    let source = r#"var<private> sink: u32;
@compute @workgroup_size(1) fn main() {
    var value = 1u;
    let saved = value;
    value = 2u;
    sink = saved; // inspect
}"#;
    let mut session = Session::launch_at(source, "// inspect");
    let frame = session.frames().remove(0);
    let variables = session.variables(&frame, "Locals");
    assert!(variables.iter().any(|variable| variable["name"] == "saved" && variable["value"] == "Primitive(U32(1))"));
}

#[test]
fn failed_relaunch_preserves_the_previous_session() {
    let mut session = Session::launch_at(CALLS, "// inside");
    let before = session.frames();
    let response = session.request(
        "launch",
        json!({"program": "/bad.wgsl", "source": "invalid"}),
    );
    assert_eq!(response["success"], false);
    let after = session.frames();
    assert_eq!(after[0]["source"], before[0]["source"]);
    assert_eq!(after[0]["line"], before[0]["line"]);
}

#[test]
fn execution_error_during_configuration_still_completes_the_launch_request() {
    let mut session = Session::default();
    session.send("initialize", json!({}));
    session.send("launch", json!({"program": "/shader.wgsl", "source":
        "var<workgroup> counter: atomic<u32>; @compute @workgroup_size(1) fn main() { let previous = atomicAdd(&counter, 1u); }"}));
    let launch_seq = session.seq;
    let messages = session.send("configurationDone", json!({}));
    assert!(
        messages
            .iter()
            .any(|message| message["request_seq"] == launch_seq)
    );
    assert!(
        messages
            .iter()
            .any(|message| message["request_seq"] == session.seq)
    );
    assert!(
        messages.iter().any(
            |message| message["event"] == "stopped" && message["body"]["reason"] == "exception"
        )
    );
}
