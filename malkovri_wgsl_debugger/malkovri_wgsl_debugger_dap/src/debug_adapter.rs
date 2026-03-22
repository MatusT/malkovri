use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
};

use dapts::Breakpoint;
use serde::Serialize;

use crate::error::DebugAdapterError;
use crate::parse_input;
use malkovri_wgsl_debugger::{Debugger, StepResult};

const LOCALS_SCOPE_REF: u32 = 1;
const ARGUMENTS_SCOPE_REF: u32 = 2;
const GLOBALS_SCOPE_REF: u32 = 3;
const MAIN_THREAD_ID: u64 = 1;

#[derive(Clone, Debug)]
pub enum OutgoingMessage {
    Response {
        seq: i64,
        request_seq: i64,
        body: serde_json::Value,
    },
    Event {
        seq: i64,
        event: String,
        body: serde_json::Value,
    },
}

impl OutgoingMessage {
    pub fn request_seq(&self) -> Option<i64> {
        match self {
            OutgoingMessage::Response { request_seq, .. } => Some(*request_seq),
            OutgoingMessage::Event { .. } => None,
        }
    }

    pub fn event_name(&self) -> Option<&str> {
        match self {
            OutgoingMessage::Event { event, .. } => Some(event),
            OutgoingMessage::Response { .. } => None,
        }
    }

    pub fn body(&self) -> &serde_json::Value {
        match self {
            OutgoingMessage::Response { body, .. } | OutgoingMessage::Event { body, .. } => body,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        match self {
            OutgoingMessage::Response {
                seq,
                request_seq,
                body,
            } => serde_json::json!({
                "seq": seq,
                "type": "response",
                "request_seq": request_seq,
                "success": true,
                "message": null,
                "body": body,
            }),
            OutgoingMessage::Event { seq, event, body } => serde_json::json!({
                "seq": seq,
                "type": "event",
                "event": event,
                "body": body,
            }),
        }
    }
}

fn make_variable(name: Option<String>, value: &str) -> dapts::Variable {
    dapts::Variable {
        name: name.clone().unwrap_or("unnamed".to_string()),
        evaluate_name: name,
        value: value.to_string(),
        variables_reference: 0,
        declaration_location_reference: None,
        indexed_variables: None,
        memory_reference: None,
        named_variables: None,
        presentation_hint: None,
        ty: None,
        value_location_reference: None,
    }
}

pub struct DebugAdapter {
    sequence_number: i64,
    breakpoints: Vec<Breakpoint>,
    program_name: Option<String>,
    program_path: Option<PathBuf>,
    debugger: Option<Debugger>,
    delayed_init_seq: Option<i64>,
    configuration_done: bool,
}

impl Default for DebugAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugAdapter {
    pub fn new() -> Self {
        DebugAdapter {
            sequence_number: 1,
            breakpoints: Vec::new(),
            debugger: None,
            program_name: None,
            program_path: None,
            delayed_init_seq: None,
            configuration_done: false,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_stdio(&mut self) -> Result<(), DebugAdapterError> {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let mut reader = BufReader::new(stdin.lock());
        let mut writer = BufWriter::new(stdout.lock());
        self.from_streams(&mut reader, &mut writer)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_streams<R: BufRead, W: Write>(
        &mut self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<(), DebugAdapterError> {
        loop {
            match Self::poll_request(reader)? {
                Some(req) => {
                    for message in self.handle_request(&req)? {
                        Self::write_message(writer, &message)?;
                    }
                }
                None => break,
            }
        }

        Ok(())
    }

    /// Process a single DAP request from a raw JSON string.
    /// Returns serialized JSON responses/events.
    pub fn handle_message(&mut self, json: &str) -> Result<Vec<String>, DebugAdapterError> {
        let request: dapts::Request = serde_json::from_str(json)?;
        let messages = self.handle_request(&request)?;
        messages
            .iter()
            .map(|m| serde_json::to_string(&m.to_json()).map_err(DebugAdapterError::from))
            .collect()
    }

    pub fn handle_request(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        match self.dispatch_request(req) {
            Ok(messages) => Ok(messages),
            Err(DebugAdapterError::Evaluator(e)) => {
                self.debugger = None;
                Ok(vec![
                    self.make_response(req.seq, &serde_json::json!({}))?,
                    self.make_event(
                        "output",
                        &serde_json::json!({
                            "category": "console",
                            "output": format!("Internal evaluator error: {}\n", e),
                        }),
                    )?,
                    self.make_event("terminated", &serde_json::json!({}))?,
                ])
            }
            Err(e) => Err(e),
        }
    }

    fn dispatch_request(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        match req.command.as_str() {
            "initialize" => self.handle_initialize(req.seq),
            "launch" => self.handle_launch(req),
            "stackTrace" => self.handle_stack_trace(req.seq),
            "scopes" => self.handle_scopes(req.seq),
            "source" => self.handle_source(req),
            "setBreakpoints" => self.handle_set_breakpoints(req),
            "configurationDone" => self.handle_configuration_done(req.seq),
            "threads" => self.handle_threads(req.seq),
            "next" => self.handle_next(req.seq),
            "continue" => self.handle_continue(req.seq),
            "variables" => self.handle_variables(req),
            "disconnect" => self.handle_disconnect(req.seq),
            "terminate" => self.handle_terminate(req.seq),
            _ => Ok(vec![]),
        }
    }

    fn debugger(&self) -> Result<&Debugger, DebugAdapterError> {
        self.debugger
            .as_ref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("debugger not initialized".into()))
    }

    fn debugger_mut(&mut self) -> Result<&mut Debugger, DebugAdapterError> {
        self.debugger
            .as_mut()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("debugger not initialized".into()))
    }

    fn handle_initialize(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        Ok(vec![
            self.make_response(
                seq,
                &dapts::Capabilities {
                    supports_cancel_request: Some(true),
                    supports_exception_info_request: Some(true),
                    supports_terminate_request: Some(true),
                    supports_restart_request: Some(true),
                    supports_set_variable: Some(true),
                    supports_configuration_done_request: Some(true),
                    supports_conditional_breakpoints: Some(true),
                    supports_hit_conditional_breakpoints: Some(true),
                    ..Default::default()
                },
            )?,
            self.make_event("initialized", &serde_json::json!({}))?,
        ])
    }

    fn handle_launch(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let arguments = req
            .arguments
            .as_object()
            .ok_or_else(|| DebugAdapterError::Parse("arguments is not an object".to_string()))?;

        let program = arguments
            .get("program")
            .ok_or_else(|| DebugAdapterError::Parse("missing 'program' argument".to_string()))?
            .as_str()
            .ok_or_else(|| DebugAdapterError::Parse("'program' is not a string".to_string()))?;

        let program_path = Path::new(program).to_path_buf();
        let program_name = program_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(program)
            .to_string();

        self.program_path = Some(program_path.clone());
        self.program_name = Some(program_name);

        let source = if let Some(source) = arguments.get("source").and_then(|v| v.as_str()) {
            source.to_string()
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            {
                fs::read_to_string(&program_path)?
            }
            #[cfg(target_arch = "wasm32")]
            {
                return Err(DebugAdapterError::Parse(
                    "missing 'source' argument (required in WASM)".to_string(),
                ));
            }
        };

        let entry_point_inputs = parse_input::parse_shader_inputs(arguments)?;
        #[cfg(not(target_arch = "wasm32"))]
        let bindings = {
            let program_dir = program_path
                .parent()
                .unwrap_or_else(|| Path::new(""))
                .to_path_buf();
            parse_input::parse_bindings(arguments, &program_dir)?
        };
        #[cfg(target_arch = "wasm32")]
        let bindings = parse_input::parse_bindings(arguments)?;

        self.debugger = Some(Debugger::new(&source, 0, entry_point_inputs, bindings)?);

        let mut messages = Vec::new();
        if !self.configuration_done {
            self.delayed_init_seq = Some(req.seq);
        } else {
            messages.push(self.make_response(req.seq, &serde_json::json!({}))?);
        }

        messages.push(self.make_stopped_event(dapts::StoppedEventReason::Entry)?);
        Ok(messages)
    }

    fn handle_stack_trace(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let debugger = self.debugger()?;

        let frames = debugger.call_stack();
        let location = frames
            .first()
            .and_then(|f| f.location.as_ref())
            .ok_or_else(|| DebugAdapterError::InvalidProgram("no current location".into()))?;

        let path = self
            .program_path
            .as_ref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("program_path not set".into()))?
            .to_string_lossy()
            .to_string();

        Ok(vec![self.make_response(
            seq,
            &dapts::StackTraceResponse {
                stack_frames: vec![dapts::StackFrame {
                    id: 1,
                    name: frames
                        .first()
                        .and_then(|f| f.name.as_deref())
                        .unwrap_or("main")
                        .to_string(),
                    source: Some(dapts::Source {
                        name: self.program_name.clone(),
                        path: Some(path),
                        adapter_data: None,
                        checksums: None,
                        origin: None,
                        presentation_hint: None,
                        source_reference: None,
                        sources: None,
                    }),
                    can_restart: None,
                    instruction_pointer_reference: None,
                    module_id: None,
                    presentation_hint: Some(dapts::StackFramePresentationHint::Normal),
                    line: location.line,
                    column: location.column,
                    end_line: None,
                    end_column: None,
                }],
                total_frames: None,
            },
        )?])
    }

    fn handle_scopes(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let debugger = self.debugger()?;

        let local_count = debugger.local_variables().len();
        let argument_count = debugger.argument_variables().len();
        let globals = debugger.global_variables();

        let mut scopes = vec![dapts::Scope {
            name: "Locals".to_string(),
            variables_reference: LOCALS_SCOPE_REF,
            named_variables: Some(local_count as u32),
            indexed_variables: None,
            expensive: false,
            source: None,
            line: None,
            end_line: None,
            column: None,
            end_column: None,
            presentation_hint: Some(dapts::ScopePresentationHint::Locals),
        }];

        if argument_count > 0 {
            scopes.push(dapts::Scope {
                name: "Function Arguments".to_string(),
                variables_reference: ARGUMENTS_SCOPE_REF,
                named_variables: Some(argument_count as u32),
                indexed_variables: None,
                expensive: false,
                source: None,
                line: None,
                end_line: None,
                column: None,
                end_column: None,
                presentation_hint: Some(dapts::ScopePresentationHint::Arguments),
            });
        }

        if !globals.is_empty() {
            scopes.push(dapts::Scope {
                name: "Globals".to_string(),
                variables_reference: GLOBALS_SCOPE_REF,
                named_variables: Some(globals.len() as u32),
                indexed_variables: None,
                expensive: false,
                source: None,
                line: None,
                end_line: None,
                column: None,
                end_column: None,
                presentation_hint: None,
            });
        }

        Ok(vec![self.make_response(seq, &dapts::ScopesResponse { scopes })?])
    }

    fn handle_source(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let arguments = serde_json::from_value::<dapts::SourceArguments>(req.arguments.clone())?;
        let requested_path = arguments
            .source
            .ok_or_else(|| DebugAdapterError::Parse("missing source".to_string()))?
            .path
            .ok_or_else(|| DebugAdapterError::Parse("missing path".to_string()))?;

        let content = if self.program_path.as_deref() == Some(Path::new(&requested_path)) {
            self.debugger()?.source().to_string()
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            {
                fs::read_to_string(&requested_path)?
            }
            #[cfg(target_arch = "wasm32")]
            {
                return Err(DebugAdapterError::Parse(format!(
                    "cannot read file '{requested_path}' in WASM"
                )));
            }
        };

        Ok(vec![self.make_response(
            req.seq,
            &dapts::SourceResponse {
                content,
                mime_type: Some("text/plain".to_string()),
            },
        )?])
    }

    fn handle_set_breakpoints(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let arguments =
            serde_json::from_value::<dapts::SetBreakpointsArguments>(req.arguments.clone())?;

        let program_path = self
            .program_path
            .as_deref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("program_path not set".into()))?;
        let program_name = self
            .program_name
            .as_deref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("program_name not set".into()))?;

        let source_matches = arguments.source.path.as_deref().map(Path::new) == Some(program_path)
            || arguments.source.name.as_deref() == Some(program_name);

        self.breakpoints = arguments
            .breakpoints
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(i, bp)| Breakpoint {
                id: Some(i as u64 + 1),
                verified: source_matches,
                message: if source_matches {
                    None
                } else {
                    Some("Breakpoint not part of debugged file.".to_string())
                },
                source: None,
                line: Some(bp.line),
                column: None,
                end_line: None,
                end_column: None,
                reason: None,
                instruction_reference: None,
                offset: None,
            })
            .collect();

        Ok(vec![self.make_response(req.seq, &self.breakpoints.clone())?])
    }

    fn handle_configuration_done(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        self.configuration_done = true;
        let mut messages = vec![self.make_response(seq, &serde_json::json!({}))?];

        if let Some(delayed_init_seq) = self.delayed_init_seq.take() {
            messages.push(self.make_response(delayed_init_seq, &serde_json::json!({}))?);
        }

        messages.push(self.make_stopped_event(dapts::StoppedEventReason::Entry)?);
        Ok(messages)
    }

    fn handle_threads(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        Ok(vec![self.make_response(
            seq,
            &dapts::ThreadsResponse {
                threads: vec![dapts::Thread {
                    id: MAIN_THREAD_ID,
                    name: "Main Thread".to_string(),
                }],
            },
        )?])
    }

    fn handle_next(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let debugger = self.debugger_mut()?;
        let has_more = matches!(debugger.step()?, StepResult::Continue);

        let mut messages = vec![self.make_response(seq, &serde_json::json!({}))?];
        if has_more {
            messages.push(self.make_stopped_event(dapts::StoppedEventReason::Step)?);
        } else {
            messages.push(self.make_event("terminated", &serde_json::json!({}))?);
        }
        Ok(messages)
    }

    fn handle_continue(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let debugger = self.debugger.as_mut().ok_or_else(|| {
            DebugAdapterError::InvalidProgram("debugger not initialized".into())
        })?;
        let breakpoints = &self.breakpoints;

        let mut has_more = false;
        loop {
            match debugger.step()? {
                StepResult::Finished => break,
                StepResult::Continue => {
                    if let Some(loc) = debugger.current_location() {
                        if breakpoints
                            .iter()
                            .any(|bp| bp.verified && bp.line == Some(loc.line))
                        {
                            has_more = true;
                            break;
                        }
                    }
                }
            }
        }

        let mut messages = vec![self.make_response(seq, &serde_json::json!({}))?];
        if has_more {
            messages.push(self.make_stopped_event(dapts::StoppedEventReason::Breakpoint)?);
        } else {
            messages.push(self.make_event("terminated", &serde_json::json!({}))?);
        }
        Ok(messages)
    }

    fn handle_disconnect(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        self.debugger = None;
        Ok(vec![self.make_response(seq, &serde_json::json!({}))?])
    }

    fn handle_terminate(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        self.debugger = None;
        Ok(vec![
            self.make_response(seq, &serde_json::json!({}))?,
            self.make_event("terminated", &serde_json::json!({}))?,
        ])
    }

    fn handle_variables(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let argument = serde_json::from_value::<dapts::VariablesArguments>(req.arguments.clone())?;
        let debugger = self.debugger()?;

        let variables = match argument.variables_reference {
            LOCALS_SCOPE_REF => debugger
                .local_variables()
                .into_iter()
                .map(|var| make_variable(var.name, &format!("{:?}", var.value)))
                .collect(),
            ARGUMENTS_SCOPE_REF => debugger
                .argument_variables()
                .into_iter()
                .map(|var| make_variable(var.name, &format!("{:?}", var.value)))
                .collect(),
            GLOBALS_SCOPE_REF => debugger
                .global_variables()
                .into_iter()
                .map(|var| make_variable(var.name, &format!("{:?}", var.value)))
                .collect(),
            _ => vec![],
        };

        Ok(vec![self.make_response(req.seq, &dapts::VariablesResponse { variables })?])
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn poll_request<R: BufRead>(
        reader: &mut R,
    ) -> Result<Option<dapts::Request>, DebugAdapterError> {
        let mut buffer = String::new();

        // Read header line
        if reader.read_line(&mut buffer)? == 0 {
            return Ok(None);
        }
        let (name, value) = buffer.trim_end().split_once(':').ok_or_else(|| {
            DebugAdapterError::Parse("Header is incorrect".to_string())
        })?;
        let content_length: usize = match name {
            "Content-Length" => value.trim().parse().map_err(|_| {
                DebugAdapterError::Parse("Content-Length is not a valid number".to_string())
            })?,
            other => {
                return Err(DebugAdapterError::Parse(format!(
                    "Unknown header: {other}"
                )));
            }
        };

        // Skip blank line separator
        buffer.clear();
        reader.read_line(&mut buffer)?;

        // Read content body
        let mut content = vec![0; content_length];
        reader.read_exact(&mut content)?;
        let content = std::str::from_utf8(&content)
            .map_err(|e| DebugAdapterError::Parse(format!("Invalid UTF-8: {e}")))?;
        let request: dapts::Request = serde_json::from_str(content)?;

        Ok(Some(request))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_message<W: Write>(
        writer: &mut W,
        message: &OutgoingMessage,
    ) -> Result<(), DebugAdapterError> {
        let payload = serde_json::to_string(&message.to_json())?;
        let framed = format!("Content-Length: {}\r\n\r\n{}", payload.len(), payload);
        writer.write_all(framed.as_bytes())?;
        writer.flush()?;
        Ok(())
    }

    fn make_response<T: Serialize + Debug>(
        &mut self,
        request_seq: i64,
        body: &T,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        Ok(OutgoingMessage::Response {
            seq: self.next_sequence_number(),
            request_seq,
            body: serde_json::to_value(body)?,
        })
    }

    fn make_event<T: Serialize + Debug>(
        &mut self,
        event: &str,
        body: &T,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        Ok(OutgoingMessage::Event {
            seq: self.next_sequence_number(),
            event: event.to_string(),
            body: serde_json::to_value(body)?,
        })
    }

    fn make_stopped_event(
        &mut self,
        reason: dapts::StoppedEventReason,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        self.make_event(
            "stopped",
            &dapts::StoppedEvent {
                reason,
                description: None,
                thread_id: Some(MAIN_THREAD_ID),
                preserve_focus_hint: None,
                text: None,
                all_threads_stopped: Some(true),
                hit_breakpoint_ids: None,
            },
        )
    }

    fn next_sequence_number(&mut self) -> i64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }
}
