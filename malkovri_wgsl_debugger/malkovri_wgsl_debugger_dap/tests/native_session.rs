use std::{
    io::{Read, Write},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

// Run the actual transport in a child process so a regression cannot hang the test suite.
fn exchange(requests: Vec<(&str, Value)>) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_malkovri_wgsl_debugger_dap"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let output = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).unwrap();
        bytes
    });
    let mut stdin = child.stdin.take().unwrap();
    for (index, (command, arguments)) in requests.into_iter().enumerate() {
        let body = json!({"seq": index + 1, "type": "request", "command": command, "arguments": arguments}).to_string();
        write!(stdin, "Content-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
    }
    drop(stdin);
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            output.join().unwrap();
            panic!("adapter failed to finish a finite request sequence");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(status.success(), "{stderr}");
    let bytes = output.join().unwrap();
    let mut remaining = bytes.as_slice();
    let mut messages = Vec::new();
    while !remaining.is_empty() {
        let end = remaining
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap();
        let header = std::str::from_utf8(&remaining[..end]).unwrap();
        let length: usize = header
            .strip_prefix("Content-Length: ")
            .unwrap()
            .parse()
            .unwrap();
        remaining = &remaining[end + 4..];
        messages.push(serde_json::from_slice(&remaining[..length]).unwrap());
        remaining = &remaining[length..];
    }
    messages
}

#[test]
fn single_thread_continue_returns_when_that_thread_finishes() {
    let messages = exchange(vec![
        ("initialize", json!({})),
        (
            "launch",
            json!({"program": "/shader.wgsl", "stopOnEntry": true, "singleThreadExecution": true,
            "workgroupConfig": {"workgroupSize": [2, 1, 1]},
            "source": "var<private> result: u32; @compute @workgroup_size(2) fn main() { result = 1u; }"}),
        ),
        ("configurationDone", json!({})),
        ("continue", json!({"threadId": 1})),
        ("stackTrace", json!({"threadId": 2})),
    ]);
    assert!(
        messages
            .iter()
            .any(|message| message["request_seq"] == 4 && message["success"] == true)
    );
    assert!(
        !messages
            .iter()
            .any(|message| message["event"] == "terminated")
    );
    let stack = messages
        .iter()
        .find(|message| message["request_seq"] == 5)
        .unwrap();
    assert_eq!(stack["body"]["stackFrames"].as_array().unwrap().len(), 1);
}

#[test]
fn invalid_launch_does_not_close_the_native_transport() {
    let messages = exchange(vec![
        (
            "launch",
            json!({"program": "/shader.wgsl", "source": "invalid WGSL"}),
        ),
        ("initialize", json!({})),
    ]);
    assert!(messages.iter().any(|message| message["request_seq"] == 1
        && message["success"] == false
        && message["command"] == "launch"));
    assert!(
        messages
            .iter()
            .any(|message| message["request_seq"] == 2 && message["success"] == true)
    );
}

#[test]
fn single_thread_continue_returns_when_waiting_at_a_barrier() {
    let messages = exchange(vec![
        ("initialize", json!({})),
        (
            "launch",
            json!({"program": "/shader.wgsl", "stopOnEntry": true,
            "workgroupConfig": {"workgroupSize": [2, 1, 1]},
            "source": "@compute @workgroup_size(2) fn main() { workgroupBarrier(); }"}),
        ),
        ("configurationDone", json!({})),
        ("continue", json!({"threadId": 1, "singleThread": true})),
        ("continue", json!({"threadId": 2})),
    ]);
    let response = messages
        .iter()
        .find(|message| message["request_seq"] == 4)
        .unwrap();
    assert_eq!(response["body"]["allThreadsContinued"], false);
    assert!(messages.iter().any(|message| {
        message["event"] == "stopped"
            && message["body"]["description"]
                .as_str()
                .is_some_and(|text| text.contains("synchronization"))
    }));
    assert!(
        messages
            .iter()
            .any(|message| message["event"] == "terminated")
    );
}

#[test]
fn continuing_an_infinite_shader_returns_control_to_the_client() {
    for source in [
        "var<private> value: u32; @compute @workgroup_size(1) fn main() { loop { value += 1u; } }",
        "@compute @workgroup_size(1) fn main() { loop {} }",
    ] {
        let messages = exchange(vec![
            ("initialize", json!({})),
            (
                "launch",
                json!({"program": "/shader.wgsl", "source": source, "stopOnEntry": true}),
            ),
            ("configurationDone", json!({})),
            ("continue", json!({"threadId": 1})),
            ("disconnect", json!({})),
        ]);
        let response_index = messages
            .iter()
            .position(|message| message["request_seq"] == 4)
            .unwrap();
        assert!(messages[response_index + 1..].iter().any(|message| {
            message["event"] == "stopped" && message["body"]["reason"] == "pause"
        }));
        assert!(
            messages
                .iter()
                .any(|message| message["request_seq"] == 5 && message["success"] == true)
        );
    }
}
