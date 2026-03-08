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
use naga::Statement;
use serde::Serialize;
use std::sync::Arc;

use crate::error::DebugAdapterError;
use crate::parse_input;
use malkovri_wgsl_debugger::{EntryPointInputs, Evaluator, NextStatement, Value};

const LOCALS_SCOPE_REF: u32 = 1;
const ARGUMENTS_SCOPE_REF: u32 = 2;
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
    program_source: String,
    evaluator: Option<Evaluator>,
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
            evaluator: None,
            program_source: String::new(),
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
        let mut messages = Vec::new();

        match req.command.as_str() {
            "initialize" => self.handle_initialize(req.seq, &mut messages)?,
            "launch" => self.handle_launch(req, &mut messages)?,
            "stackTrace" => self.handle_stack_trace(req.seq, &mut messages)?,
            "scopes" => self.handle_scopes(req.seq, &mut messages)?,
            "source" => self.handle_source(req, &mut messages)?,
            "setBreakpoints" => self.handle_set_breakpoints(req, &mut messages)?,
            "configurationDone" => self.handle_configuration_done(req.seq, &mut messages)?,
            "threads" => self.handle_threads(req.seq, &mut messages)?,
            "next" => self.handle_next(req.seq, &mut messages)?,
            "variables" => self.handle_variables(req, &mut messages)?,
            _ => {}
        }

        Ok(messages)
    }

    fn evaluator(&self) -> Result<&Evaluator, DebugAdapterError> {
        self.evaluator
            .as_ref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("evaluator not initialized".into()))
    }

    fn evaluator_mut(&mut self) -> Result<&mut Evaluator, DebugAdapterError> {
        self.evaluator
            .as_mut()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("evaluator not initialized".into()))
    }

    fn handle_initialize(
        &mut self,
        seq: i64,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
        self.queue_response(
            messages,
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
        )?;
        self.queue_event(messages, "initialized", &serde_json::json!({}))?;
        Ok(())
    }

    fn handle_launch(
        &mut self,
        req: &dapts::Request,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
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
        self.program_source = if let Some(source) = arguments.get("source").and_then(|v| v.as_str())
        {
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

        let module = Arc::new(malkovri_wgsl_debugger::wgsl_to_module(
            &self.program_source,
        )?);
        let global_invocation_id = parse_input::parse_global_invocation_id(arguments);
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

        self.evaluator = Some(Evaluator::new(
            module,
            0,
            EntryPointInputs {
                global_invocation_id,
                ..Default::default()
            },
            bindings,
        ));

        if !self.configuration_done {
            self.delayed_init_seq = Some(req.seq);
        } else {
            self.queue_response(messages, req.seq, &serde_json::json!({}))?;
        }

        self.queue_stopped_event(messages, dapts::StoppedEventReason::Entry)?;
        Ok(())
    }

    fn handle_stack_trace(
        &mut self,
        seq: i64,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
        let evaluator = self.evaluator()?;

        let current_fn = evaluator
            .current_function()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("no current function".into()))?;

        let (active_block, active_index) = evaluator
            .current_active_block()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("stack is empty".into()))?;

        let current_statement = active_block.get(active_index).ok_or_else(|| {
            DebugAdapterError::InvalidProgram("invalid statement index".into())
        })?;

        let spans = active_block
            .span_iter()
            .map(|(_, span)| span)
            .collect::<Vec<_>>();

        let relevant_span = spans[active_index].location(&self.program_source);
        let total_span = naga::Span::total_span(current_fn.body.span_iter().map(|(_, span)| *span));

        let (line, column) = if let Statement::Return { .. } = current_statement {
            let total_span_range = total_span.to_range().unwrap();
            let prefix = &self.program_source[..total_span_range.end];
            let line_number = prefix.matches('\n').count() as u32 + 2;
            (line_number, 0)
        } else {
            (relevant_span.line_number, relevant_span.line_position)
        };

        let path = self
            .program_path
            .as_ref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("program_path not set".into()))?
            .to_string_lossy()
            .to_string();

        self.queue_response(
            messages,
            seq,
            &dapts::StackTraceResponse {
                stack_frames: vec![dapts::StackFrame {
                    id: 1,
                    name: "main".to_string(),
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
                    line,
                    column,
                    end_line: None,
                    end_column: None,
                }],
                total_frames: None,
            },
        )?;

        Ok(())
    }

    fn handle_scopes(
        &mut self,
        seq: i64,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
        let evaluator = self.evaluator()?;

        let current_fn = evaluator.current_function().unwrap();
        let named_variables_len =
            current_fn.local_variables.len() + current_fn.named_expressions.len();
        let function_arguments_len = current_fn.arguments.len();

        let mut scopes = vec![dapts::Scope {
            name: "Locals".to_string(),
            variables_reference: LOCALS_SCOPE_REF,
            named_variables: Some(named_variables_len as u32),
            indexed_variables: None,
            expensive: false,
            source: None,
            line: None,
            end_line: None,
            column: None,
            end_column: None,
            presentation_hint: Some(dapts::ScopePresentationHint::Locals),
        }];

        if function_arguments_len > 0 {
            scopes.push(dapts::Scope {
                name: "Function Arguments".to_string(),
                variables_reference: ARGUMENTS_SCOPE_REF,
                named_variables: Some(function_arguments_len as u32),
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

        self.queue_response(messages, seq, &dapts::ScopesResponse { scopes })?;
        Ok(())
    }

    fn handle_source(
        &mut self,
        req: &dapts::Request,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
        let arguments = serde_json::from_value::<dapts::SourceArguments>(req.arguments.clone())?;
        let requested_path = arguments
            .source
            .ok_or_else(|| DebugAdapterError::Parse("missing source".to_string()))?
            .path
            .ok_or_else(|| DebugAdapterError::Parse("missing path".to_string()))?;

        let content =
            if self.program_path.as_deref() == Some(Path::new(&requested_path)) {
                self.program_source.clone()
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

        self.queue_response(
            messages,
            req.seq,
            &dapts::SourceResponse {
                content,
                mime_type: Some("text/plain".to_string()),
            },
        )?;

        Ok(())
    }

    fn handle_set_breakpoints(
        &mut self,
        req: &dapts::Request,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
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

        self.queue_response(messages, req.seq, &self.breakpoints.clone())?;
        Ok(())
    }

    fn handle_configuration_done(
        &mut self,
        seq: i64,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
        self.configuration_done = true;
        self.queue_response(messages, seq, &serde_json::json!({}))?;

        if let Some(delayed_init_seq) = self.delayed_init_seq.take() {
            self.queue_response(messages, delayed_init_seq, &serde_json::json!({}))?;
        }

        self.queue_stopped_event(messages, dapts::StoppedEventReason::Entry)?;
        Ok(())
    }

    fn handle_threads(
        &mut self,
        seq: i64,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
        self.queue_response(
            messages,
            seq,
            &dapts::ThreadsResponse {
                threads: vec![dapts::Thread {
                    id: MAIN_THREAD_ID,
                    name: "Main Thread".to_string(),
                }],
            },
        )?;
        Ok(())
    }

    fn handle_next(
        &mut self,
        seq: i64,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
        let evaluator = self.evaluator_mut()?;

        while let Some(statement) = evaluator.step() {
            if !matches!(
                statement,
                NextStatement {
                    statement: Statement::Emit(_),
                    ..
                }
            ) {
                break;
            }
        }

        self.queue_response(messages, seq, &serde_json::json!({}))?;
        self.queue_stopped_event(messages, dapts::StoppedEventReason::Step)?;
        Ok(())
    }

    fn handle_variables(
        &mut self,
        req: &dapts::Request,
        messages: &mut Vec<OutgoingMessage>,
    ) -> Result<(), DebugAdapterError> {
        let argument = serde_json::from_value::<dapts::VariablesArguments>(req.arguments.clone())?;
        let evaluator = self.evaluator()?;

        let current_fn = evaluator.current_function().unwrap();
        let current_state = evaluator.current_function_frame().unwrap();

        let variables = match argument.variables_reference {
            LOCALS_SCOPE_REF => {
                let mut vars: Vec<dapts::Variable> = current_fn
                    .local_variables
                    .iter()
                    .map(|(handle, local)| {
                        let val = current_state
                            .local_variables
                            .get(&handle)
                            .unwrap_or(&Value::Uninitialized);
                        make_variable(
                            local.name.clone(),
                            &format!("{:?}", val.leaf_value()),
                        )
                    })
                    .collect();

                for (name, value) in evaluator.named_expression_values() {
                    vars.push(make_variable(
                        Some(name),
                        &format!("{:?}", value.leaf_value()),
                    ));
                }

                vars
            }
            ARGUMENTS_SCOPE_REF => evaluator
                .current_function_argument_values()
                .into_iter()
                .map(|(name, value)| {
                    make_variable(name, &format!("{:?}", value.leaf_value()))
                })
                .collect(),
            _ => vec![],
        };

        self.queue_response(messages, req.seq, &dapts::VariablesResponse { variables })?;
        Ok(())
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

    fn queue_response<T: Serialize + Debug>(
        &mut self,
        messages: &mut Vec<OutgoingMessage>,
        request_seq: i64,
        body: &T,
    ) -> Result<(), DebugAdapterError> {
        messages.push(OutgoingMessage::Response {
            seq: self.next_sequence_number(),
            request_seq,
            body: serde_json::to_value(body)?,
        });
        Ok(())
    }

    fn queue_event<T: Serialize + Debug>(
        &mut self,
        messages: &mut Vec<OutgoingMessage>,
        event: &str,
        body: &T,
    ) -> Result<(), DebugAdapterError> {
        messages.push(OutgoingMessage::Event {
            seq: self.next_sequence_number(),
            event: event.to_string(),
            body: serde_json::to_value(body)?,
        });
        Ok(())
    }

    fn queue_stopped_event(
        &mut self,
        messages: &mut Vec<OutgoingMessage>,
        reason: dapts::StoppedEventReason,
    ) -> Result<(), DebugAdapterError> {
        self.queue_event(
            messages,
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
