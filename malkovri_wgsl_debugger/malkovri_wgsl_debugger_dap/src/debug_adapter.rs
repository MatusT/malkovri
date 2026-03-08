use std::{
    collections::HashMap,
    fmt::Debug,
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use dapts::Breakpoint;
use naga::{ResourceBinding, Statement};
use serde::Serialize;
use std::sync::Arc;

use crate::error::DebugAdapterError;
use malkovri_wgsl_debugger::{EntryPointInputs, Evaluator, NextStatement, Primitive, Value};

const LOCALS_SCOPE_REF: u32 = 1;
const ARGUMENTS_SCOPE_REF: u32 = 2;
const MAIN_THREAD_ID: u64 = 1;

#[derive(Debug)]
enum ServerState {
    Header,
    Content,
}

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

fn parse_global_invocation_id(arguments: &serde_json::Map<String, serde_json::Value>) -> [u32; 3] {
    arguments
        .get("global_invocation_id")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            Some([
                arr.first()?.as_u64()? as u32,
                arr.get(1)?.as_u64()? as u32,
                arr.get(2)?.as_u64()? as u32,
            ])
        })
        .unwrap_or([0, 0, 0])
}

fn parse_bindings(
    arguments: &serde_json::Map<String, serde_json::Value>,
    program_dir: &Path,
) -> Result<HashMap<ResourceBinding, Value>, DebugAdapterError> {
    let Some(bindings) = arguments.get("bindings").and_then(|v| v.as_object()) else {
        return Ok(HashMap::new());
    };

    bindings
        .iter()
        .map(|(key, config)| {
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() != 2 {
                return Err(DebugAdapterError::Parse(format!(
                    "Invalid binding key '{key}': expected 'group:binding'"
                )));
            }
            let group = parts[0].parse::<u32>().map_err(|_| {
                DebugAdapterError::Parse(format!("Invalid group in binding key '{key}'"))
            })?;
            let binding = parts[1].parse::<u32>().map_err(|_| {
                DebugAdapterError::Parse(format!("Invalid binding in binding key '{key}'"))
            })?;

            let obj = config.as_object().ok_or_else(|| {
                DebugAdapterError::Parse(format!("Binding '{key}' is not an object"))
            })?;

            let type_str = obj.get("type").and_then(|v| v.as_str()).unwrap_or("f32");
            let value = if let Some(inline) = obj.get("inline") {
                parse_inline(key, type_str, inline)?
            } else if let Some(path) = obj.get("file").and_then(|v| v.as_str()) {
                let format = obj.get("format").and_then(|v| v.as_str()).unwrap_or("ron");
                parse_file(key, type_str, format, &program_dir.join(path))?
            } else {
                return Err(DebugAdapterError::Parse(format!(
                    "Binding '{key}' has neither 'inline' nor 'file'"
                )));
            };

            Ok((ResourceBinding { group, binding }, value))
        })
        .collect()
}

fn parse_inline(
    key: &str,
    type_str: &str,
    inline: &serde_json::Value,
) -> Result<Value, DebugAdapterError> {
    let arr = inline.as_array().ok_or_else(|| {
        DebugAdapterError::Parse(format!("Binding '{key}' inline value is not an array"))
    })?;
    Ok(match type_str {
        "f32" => Value::Array(
            arr.iter()
                .map(|v| Primitive::F32(v.as_f64().unwrap_or(0.0) as f32).into())
                .collect(),
        ),
        "i32" => Value::Array(
            arr.iter()
                .map(|v| Primitive::I32(v.as_i64().unwrap_or(0) as i32).into())
                .collect(),
        ),
        "u32" => Value::Array(
            arr.iter()
                .map(|v| Primitive::U32(v.as_u64().unwrap_or(0) as u32).into())
                .collect(),
        ),
        _ => {
            return Err(DebugAdapterError::Parse(format!(
                "Unknown type '{type_str}' for binding '{key}'"
            )));
        }
    })
}

fn parse_file(
    key: &str,
    type_str: &str,
    format: &str,
    path: &Path,
) -> Result<Value, DebugAdapterError> {
    match format {
        "binary" => {
            let bytes = fs::read(path)?;
            Ok(match type_str {
                "f32" => Value::Array(
                    bytes
                        .chunks_exact(4)
                        .map(|c| {
                            Primitive::F32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).into()
                        })
                        .collect(),
                ),
                "i32" => Value::Array(
                    bytes
                        .chunks_exact(4)
                        .map(|c| {
                            Primitive::I32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])).into()
                        })
                        .collect(),
                ),
                "u32" => Value::Array(
                    bytes
                        .chunks_exact(4)
                        .map(|c| {
                            Primitive::U32(u32::from_le_bytes([c[0], c[1], c[2], c[3]])).into()
                        })
                        .collect(),
                ),
                _ => {
                    return Err(DebugAdapterError::Parse(format!(
                        "Unknown type '{type_str}' for binding '{key}'"
                    )));
                }
            })
        }
        "ron" => {
            let content = fs::read_to_string(path)?;
            Ok(match type_str {
                "f32" => {
                    let vals: Vec<f64> = ron::from_str(&content).map_err(|e| {
                        DebugAdapterError::Parse(format!(
                            "RON parse error for binding '{key}': {e}"
                        ))
                    })?;
                    Value::Array(
                        vals.into_iter()
                            .map(|v| Primitive::F32(v as f32).into())
                            .collect(),
                    )
                }
                "i32" => {
                    let vals: Vec<i64> = ron::from_str(&content).map_err(|e| {
                        DebugAdapterError::Parse(format!(
                            "RON parse error for binding '{key}': {e}"
                        ))
                    })?;
                    Value::Array(
                        vals.into_iter()
                            .map(|v| Primitive::I32(v as i32).into())
                            .collect(),
                    )
                }
                "u32" => {
                    let vals: Vec<u64> = ron::from_str(&content).map_err(|e| {
                        DebugAdapterError::Parse(format!(
                            "RON parse error for binding '{key}': {e}"
                        ))
                    })?;
                    Value::Array(
                        vals.into_iter()
                            .map(|v| Primitive::U32(v as u32).into())
                            .collect(),
                    )
                }
                _ => {
                    return Err(DebugAdapterError::Parse(format!(
                        "Unknown type '{type_str}' for binding '{key}'"
                    )));
                }
            })
        }
        _ => Err(DebugAdapterError::Parse(format!(
            "Unknown format '{format}' for binding '{key}'"
        ))),
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

    pub fn start(&mut self) -> Result<(), DebugAdapterError> {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let mut reader = BufReader::new(stdin.lock());
        let mut writer = BufWriter::new(stdout.lock());
        self.run(&mut reader, &mut writer)
    }

    pub fn run<R: BufRead, W: Write>(
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
        self.program_source = fs::read_to_string(&program_path)?;

        let module = Arc::new(malkovri_wgsl_debugger::wgsl_to_module(
            &self.program_source,
        )?);
        let global_invocation_id = parse_global_invocation_id(arguments);
        let program_dir = program_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .to_path_buf();
        let bindings = parse_bindings(arguments, &program_dir)?;

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
        let evaluator = self.evaluator.as_ref().ok_or_else(|| {
            DebugAdapterError::InvalidProgram("evaluator not initialized".to_string())
        })?;

        let current_fn = evaluator
            .current_function()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("no current function".to_string()))?;

        let (active_block, active_index) = evaluator
            .current_active_block()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("stack is empty".to_string()))?;

        let current_statement = active_block.get(active_index).ok_or_else(|| {
            DebugAdapterError::InvalidProgram("invalid statement index".to_string())
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
            .ok_or_else(|| DebugAdapterError::InvalidProgram("program_path not set".to_string()))?
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
        let evaluator = self.evaluator.as_ref().ok_or_else(|| {
            DebugAdapterError::InvalidProgram("evaluator not initialized".to_string())
        })?;

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
        let content = fs::read_to_string(
            arguments
                .source
                .ok_or_else(|| DebugAdapterError::Parse("missing source".to_string()))?
                .path
                .ok_or_else(|| DebugAdapterError::Parse("missing path".to_string()))?,
        )?;

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
            .ok_or_else(|| DebugAdapterError::InvalidProgram("program_path not set".to_string()))?;
        let program_name = self
            .program_name
            .as_deref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("program_name not set".to_string()))?;

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
        let evaluator = self.evaluator.as_mut().ok_or_else(|| {
            DebugAdapterError::InvalidProgram("evaluator not initialized".to_string())
        })?;

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
        let evaluator = self.evaluator.as_ref().ok_or_else(|| {
            DebugAdapterError::InvalidProgram("evaluator not initialized".to_string())
        })?;

        let current_fn = evaluator.current_function().unwrap();
        let current_state = evaluator.current_function_frame().unwrap();

        if argument.variables_reference == LOCALS_SCOPE_REF {
            let mut variables: Vec<dapts::Variable> = current_fn
                .local_variables
                .iter()
                .map(|(local_variable_handle, local_variable)| {
                    let local_variable_value = current_state
                        .local_variables
                        .get(&local_variable_handle)
                        .unwrap_or(&Value::Uninitialized);
                    dapts::Variable {
                        declaration_location_reference: None,
                        evaluate_name: local_variable.name.clone(),
                        indexed_variables: None,
                        memory_reference: None,
                        name: local_variable.name.clone().unwrap_or("unnamed".to_string()),
                        named_variables: None,
                        presentation_hint: None,
                        ty: None,
                        value: format!("{:?}", local_variable_value.leaf_value()),
                        value_location_reference: None,
                        variables_reference: 0,
                    }
                })
                .collect();

            for (name, value) in evaluator.named_expression_values() {
                variables.push(dapts::Variable {
                    declaration_location_reference: None,
                    evaluate_name: Some(name.clone()),
                    indexed_variables: None,
                    memory_reference: None,
                    name,
                    named_variables: None,
                    presentation_hint: None,
                    ty: None,
                    value: format!("{:?}", value.leaf_value()),
                    value_location_reference: None,
                    variables_reference: 0,
                });
            }

            self.queue_response(messages, req.seq, &dapts::VariablesResponse { variables })?;
            return Ok(());
        }

        if argument.variables_reference == ARGUMENTS_SCOPE_REF {
            let variables: Vec<dapts::Variable> = evaluator
                .current_function_argument_values()
                .into_iter()
                .map(|(name, value)| dapts::Variable {
                    declaration_location_reference: None,
                    evaluate_name: name.clone(),
                    indexed_variables: None,
                    memory_reference: None,
                    name: name.unwrap_or("unnamed".to_string()),
                    named_variables: None,
                    presentation_hint: None,
                    ty: None,
                    value: format!("{:?}", value.leaf_value()),
                    value_location_reference: None,
                    variables_reference: 0,
                })
                .collect();

            self.queue_response(messages, req.seq, &dapts::VariablesResponse { variables })?;
            return Ok(());
        }

        self.queue_response(
            messages,
            req.seq,
            &dapts::VariablesResponse { variables: vec![] },
        )?;
        Ok(())
    }

    pub fn poll_request<R: BufRead>(
        reader: &mut R,
    ) -> Result<Option<dapts::Request>, DebugAdapterError> {
        let mut state = ServerState::Header;
        let mut buffer = String::new();
        let mut content_length: usize = 0;

        loop {
            match reader.read_line(&mut buffer) {
                Ok(read_size) => {
                    if read_size == 0 {
                        return Ok(None);
                    }
                    match state {
                        ServerState::Header => {
                            let parts: Vec<&str> = buffer.trim_end().split(':').collect();
                            if parts.len() == 2 {
                                match parts[0] {
                                    "Content-Length" => {
                                        content_length = parts[1].trim().parse().map_err(|_| {
                                            DebugAdapterError::Parse(
                                                "Content-Length is not a valid number".to_string(),
                                            )
                                        })?;
                                        buffer.clear();
                                        buffer.reserve(content_length);
                                        state = ServerState::Content;
                                    }
                                    other => {
                                        return Err(DebugAdapterError::Parse(format!(
                                            "Unknown header: {other}"
                                        )));
                                    }
                                }
                            } else {
                                return Err(DebugAdapterError::Parse(
                                    "Header is incorrect".to_string(),
                                ));
                            }
                        }
                        ServerState::Content => {
                            buffer.clear();
                            let mut content = vec![0; content_length];
                            reader.read_exact(content.as_mut_slice())?;

                            let content = std::str::from_utf8(content.as_slice()).map_err(|e| {
                                DebugAdapterError::Parse(format!("Invalid UTF-8: {e}"))
                            })?;
                            let request: dapts::Request = serde_json::from_str(content)?;

                            return Ok(Some(request));
                        }
                    }
                }
                Err(e) => return Err(DebugAdapterError::Io(e)),
            }
        }
    }

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
