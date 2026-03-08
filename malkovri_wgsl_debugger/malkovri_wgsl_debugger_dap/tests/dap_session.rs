use std::{collections::HashMap, path::PathBuf};

use malkovri_wgsl_debugger_dap::{DebugAdapter, OutgoingMessage};
use serde_json::{Value, json};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn shader_path(file_name: &str) -> PathBuf {
    workspace_root().join("test_shaders").join(file_name)
}

fn request(seq: i64, command: &str, arguments: Value) -> dapts::Request {
    serde_json::from_value(json!({
        "seq": seq,
        "type": "request",
        "command": command,
        "arguments": arguments,
    }))
    .unwrap()
}

fn has_response(messages: &[OutgoingMessage], request_seq: i64) -> bool {
    messages
        .iter()
        .any(|message| message.request_seq() == Some(request_seq))
}

fn response_body(messages: &[OutgoingMessage], request_seq: i64) -> &Value {
    messages
        .iter()
        .find_map(|message| (message.request_seq() == Some(request_seq)).then_some(message.body()))
        .unwrap_or_else(|| panic!("missing response for request {request_seq}"))
}

fn event_body<'a>(messages: &'a [OutgoingMessage], event_name: &str) -> &'a Value {
    messages
        .iter()
        .find_map(|message| (message.event_name() == Some(event_name)).then_some(message.body()))
        .unwrap_or_else(|| panic!("missing event {event_name}"))
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
    let mut variables = HashMap::new();
    for variable in variables_body["variables"].as_array().unwrap() {
        let name = variable["name"].as_str().unwrap().to_string();
        let value = variable["value"].as_str().unwrap().to_string();
        variables.entry(name).or_insert(value);
    }
    variables
}

#[test]
fn control_flow_session_matches_vscode_request_flow() {
    let mut adapter = DebugAdapter::new();
    let shader = shader_path("test_control_flow.wgsl");
    let shader_path = shader.to_string_lossy().to_string();

    let initialize = adapter
        .handle_request(&request(1, "initialize", json!({})))
        .unwrap();
    assert_eq!(
        response_body(&initialize, 1)["supportsConfigurationDoneRequest"],
        json!(true)
    );
    assert_eq!(event_body(&initialize, "initialized"), &json!({}));

    let launch = adapter
        .handle_request(&request(
            2,
            "launch",
            json!({
                "program": shader_path,
                "global_invocation_id": [5, 0, 0],
            }),
        ))
        .unwrap();
    assert!(!has_response(&launch, 2));
    assert_eq!(event_body(&launch, "stopped")["reason"], json!("entry"));

    let breakpoints = adapter
        .handle_request(&request(
            3,
            "setBreakpoints",
            json!({
                "source": {
                    "name": "test_control_flow.wgsl",
                    "path": shader_path,
                },
                "breakpoints": [{ "line": 24 }],
            }),
        ))
        .unwrap();
    assert_eq!(response_body(&breakpoints, 3)[0]["verified"], json!(true));

    let configuration_done = adapter
        .handle_request(&request(4, "configurationDone", json!({})))
        .unwrap();
    assert!(has_response(&configuration_done, 4));
    assert!(has_response(&configuration_done, 2));
    assert_eq!(
        event_body(&configuration_done, "stopped")["reason"],
        json!("entry")
    );

    let threads = adapter
        .handle_request(&request(5, "threads", json!({})))
        .unwrap();
    assert_eq!(
        response_body(&threads, 5)["threads"][0]["name"],
        json!("Main Thread")
    );

    let stack_trace = adapter
        .handle_request(&request(6, "stackTrace", json!({ "threadId": 1 })))
        .unwrap();
    assert_eq!(
        response_body(&stack_trace, 6)["stackFrames"][0]["line"],
        json!(5)
    );
    assert_eq!(
        response_body(&stack_trace, 6)["stackFrames"][0]["source"]["path"],
        json!(shader_path),
    );

    let source = adapter
        .handle_request(&request(
            7,
            "source",
            json!({
                "source": {
                    "path": shader_path,
                },
                "sourceReference": 0,
            }),
        ))
        .unwrap();
    assert!(
        response_body(&source, 7)["content"]
            .as_str()
            .unwrap()
            .contains("var count = idx % 16u;")
    );

    let scopes = adapter
        .handle_request(&request(8, "scopes", json!({ "frameId": 1 })))
        .unwrap();
    let locals_ref = scope_reference(response_body(&scopes, 8), "Locals");
    let arguments_ref = scope_reference(response_body(&scopes, 8), "Function Arguments");

    let arguments = adapter
        .handle_request(&request(
            9,
            "variables",
            json!({ "variablesReference": arguments_ref }),
        ))
        .unwrap();
    let arguments = variables_map(response_body(&arguments, 9));
    assert_eq!(
        arguments.get("global_id").unwrap(),
        "Primitive(U32x3([5, 0, 0]))"
    );

    let locals = adapter
        .handle_request(&request(
            10,
            "variables",
            json!({ "variablesReference": locals_ref }),
        ))
        .unwrap();
    let locals = variables_map(response_body(&locals, 10));
    assert_eq!(locals.get("idx").unwrap(), "Primitive(U32(5))");
    assert_eq!(locals.get("result").unwrap(), "Primitive(U32(0))");

    let next = adapter
        .handle_request(&request(11, "next", json!({ "threadId": 1 })))
        .unwrap();
    assert!(has_response(&next, 11));
    assert_eq!(event_body(&next, "stopped")["reason"], json!("step"));

    let stack_after_step = adapter
        .handle_request(&request(12, "stackTrace", json!({ "threadId": 1 })))
        .unwrap();
    assert_ne!(
        response_body(&stack_after_step, 12)["stackFrames"][0]["line"],
        json!(5),
    );
}

#[test]
fn expression_shader_variables_include_binding_backed_values() {
    let mut adapter = DebugAdapter::new();
    let shader = shader_path("test_expressions.wgsl");
    let shader_path = shader.to_string_lossy().to_string();

    adapter
        .handle_request(&request(1, "initialize", json!({})))
        .unwrap();
    adapter
        .handle_request(&request(
            2,
            "launch",
            json!({
                "program": shader_path,
                "global_invocation_id": [2, 0, 0],
                "bindings": {
                    "0:0": {
                        "type": "f32",
                        "inline": [1.0, 2.0, 3.0, 4.0],
                    },
                    "0:1": {
                        "type": "u32",
                        "inline": [0, 0, 0, 0],
                    },
                },
            }),
        ))
        .unwrap();
    adapter
        .handle_request(&request(
            3,
            "setBreakpoints",
            json!({
                "source": {
                    "name": "test_expressions.wgsl",
                    "path": shader_path,
                },
                "breakpoints": [{ "line": 45 }],
            }),
        ))
        .unwrap();
    adapter
        .handle_request(&request(4, "configurationDone", json!({})))
        .unwrap();

    let scopes = adapter
        .handle_request(&request(5, "scopes", json!({ "frameId": 1 })))
        .unwrap();
    let locals_ref = scope_reference(response_body(&scopes, 5), "Locals");
    let arguments_ref = scope_reference(response_body(&scopes, 5), "Function Arguments");

    let arguments = adapter
        .handle_request(&request(
            6,
            "variables",
            json!({ "variablesReference": arguments_ref }),
        ))
        .unwrap();
    let arguments = variables_map(response_body(&arguments, 6));
    assert_eq!(arguments.get("gid").unwrap(), "Primitive(U32x3([2, 0, 0]))");

    let locals = adapter
        .handle_request(&request(
            7,
            "variables",
            json!({ "variablesReference": locals_ref }),
        ))
        .unwrap();
    let locals = variables_map(response_body(&locals, 7));

    assert_eq!(locals.get("idx").unwrap(), "Primitive(U32(2))");
    assert_eq!(locals.get("lv_f32").unwrap(), "Primitive(F32(3.14))");
    assert_eq!(
        locals.get("lv_v3f").unwrap(),
        "Primitive(F32x3([0.5, 1.5, 2.5]))",
    );
    assert_eq!(locals.get("arr_len").unwrap(), "Primitive(U32(4))");
    assert_eq!(locals.get("g_val").unwrap(), "Primitive(F32(1.0))");
    assert_eq!(locals.get("acc_g").unwrap(), "Primitive(F32(3.0))");
    assert_eq!(locals.get("dyn_i").unwrap(), "Primitive(U32(2))");
    assert_eq!(locals.get("acc_vf").unwrap(), "Primitive(F32(2.5))");
}
