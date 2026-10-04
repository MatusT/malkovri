use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use std::fs;

use dapts::Breakpoint;

use crate::error::DebugAdapterError;
use crate::parse_input;
use crate::protocol::{BreakpointId, OutgoingMessage, StackFrameId, make_variable};
use crate::references::{FrameReference, References, ScopeKind};
use malkovri_wgsl_debugger::{
    DebugThreadId, Debugger, DebuggerError, EvaluatorError, RunResult, ShaderProgram,
};

pub struct DebugAdapter {
    sequence_number: i64,
    references: References,
    breakpoints: Vec<Breakpoint>,
    program_name: Option<String>,
    program_path: Option<PathBuf>,
    debugger: Option<Debugger>,
    delayed_init_seq: Option<i64>,
    configuration_done: bool,
    stop_on_entry: bool,
    single_thread_execution: bool,
    trace_enabled: bool,
    outputs_presented: bool,
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
            references: References::default(),
            breakpoints: Vec::new(),
            debugger: None,
            program_name: None,
            program_path: None,
            delayed_init_seq: None,
            configuration_done: false,
            stop_on_entry: false,
            single_thread_execution: false,
            trace_enabled: false,
            outputs_presented: false,
        }
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
            Err(error) => {
                let mut messages = vec![self.make_error_response(req, &error)];
                if matches!(req.command.as_str(), "next" | "continue")
                    && matches!(error, DebugAdapterError::Evaluator(_))
                {
                    messages.push(self.make_execution_error_event(&error)?);
                }
                Ok(messages)
            }
        }
    }

    fn dispatch_request(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        match req.command.as_str() {
            "initialize" => self.handle_initialize(req.seq),
            "launch" => self.handle_launch(req),
            "stackTrace" => self.handle_stack_trace(req),
            "scopes" => self.handle_scopes(req),
            "source" => self.handle_source(req),
            "setBreakpoints" => self.handle_set_breakpoints(req),
            "configurationDone" => self.handle_configuration_done(req.seq),
            "threads" => self.handle_threads(req.seq),
            "next" => self.handle_next(req),
            "continue" => self.handle_continue(req),
            "variables" => self.handle_variables(req),
            "disconnect" => self.handle_disconnect(req.seq),
            "terminate" => self.handle_terminate(req.seq),
            _ => Err(DebugAdapterError::Parse(format!(
                "unsupported request '{}'",
                req.command
            ))),
        }
    }

    /// Read the current execution state, including final globals after termination.
    pub fn debugger(&self) -> Result<&Debugger, DebugAdapterError> {
        self.debugger
            .as_ref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("debugger not initialized".into()))
    }

    fn debugger_mut(&mut self) -> Result<&mut Debugger, DebugAdapterError> {
        self.debugger
            .as_mut()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("debugger not initialized".into()))
    }

    fn handle_initialize(&mut self, seq: i64) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        Ok(vec![
            self.make_response(
                seq,
                "initialize",
                &dapts::Capabilities {
                    supports_terminate_request: Some(true),
                    supports_single_thread_execution_requests: Some(true),
                    supports_configuration_done_request: Some(true),
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

        let stop_on_entry = arguments
            .get("stopOnEntry")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let single_thread_execution = arguments
            .get("singleThreadExecution")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let trace_enabled = arguments
            .get("trace")
            .or_else(|| arguments.get("debugTrace"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

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

        let shader_program = ShaderProgram::new(&source).map_err(DebuggerError::from)?;
        let stage = shader_program
            .entry_points()
            .next()
            .ok_or_else(|| DebugAdapterError::Parse("shader has no entry points".into()))?
            .stage;
        let config = parse_input::parse_execution_config(arguments, stage)?;
        let global_constants = parse_input::parse_global_constants(arguments)?;
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

        self.debugger =
            Some(shader_program.create_debugger(0, config, global_constants, bindings)?);
        self.program_path = Some(program_path);
        self.program_name = Some(program_name);
        self.stop_on_entry = stop_on_entry;
        self.single_thread_execution = single_thread_execution;
        self.trace_enabled = trace_enabled;
        self.outputs_presented = false;
        self.references.clear();

        let mut messages = Vec::new();
        if !self.configuration_done {
            self.delayed_init_seq = Some(req.seq);
        } else {
            messages.push(self.make_response(req.seq, &req.command, &serde_json::json!({}))?);
            messages.push(self.initial_stop_event()?);
        }

        Ok(messages)
    }

    fn handle_stack_trace(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let arguments =
            serde_json::from_value::<dapts::StackTraceArguments>(req.arguments.clone())?;
        let thread_id: DebugThreadId = arguments.thread_id;
        let debugger = self.debugger()?;
        let mut frames = debugger
            .thread_call_stack(thread_id)?
            .into_iter()
            .map(|frame| {
                (
                    FrameReference::new(thread_id, frame.id),
                    frame.name,
                    frame.location,
                )
            })
            .collect::<Vec<_>>();
        // A completed shader has no live call frame. Keep its return site selectable
        // so the client can request output scopes while stopped after execution.
        if frames.is_empty()
            && let Some(location) = debugger.thread_entry_point_return_location(thread_id)?
        {
            frames.push((
                FrameReference::shader_outputs(thread_id),
                location.function_name.clone(),
                Some(location),
            ));
        }
        let path = self
            .program_path
            .as_ref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("program_path not set".into()))?
            .to_string_lossy()
            .to_string();

        let mut stack_frames = Vec::new();
        for (reference, name, location) in &frames {
            let frame_id = self.references.insert_frame(*reference);
            let location = location.as_ref();
            let line = location.map(|loc| loc.line).unwrap_or(1);
            let column = location.map(|loc| loc.column).unwrap_or(0);
            stack_frames.push(dapts::StackFrame {
                id: frame_id,
                name: name.as_deref().unwrap_or("main").to_string(),
                source: Some(dapts::Source {
                    name: self.program_name.clone(),
                    path: Some(path.clone()),
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
            });
        }

        Ok(vec![self.make_response(
            req.seq,
            &req.command,
            &dapts::StackTraceResponse {
                stack_frames,
                total_frames: Some(frames.len() as u64),
            },
        )?])
    }

    fn handle_scopes(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let arguments = serde_json::from_value::<dapts::ScopesArguments>(req.arguments.clone())?;
        let frame_id: StackFrameId = arguments.frame_id;
        let frame_reference = self.references.frame(frame_id)?;
        let thread_id = frame_reference.thread_id();
        let debugger = self.debugger()?;
        let (local_count, argument_count) = if frame_reference.is_shader_outputs() {
            (0, 0)
        } else {
            let frame_id = frame_reference.frame_id()?;
            (
                debugger.frame_local_variables(thread_id, frame_id)?.len(),
                debugger
                    .frame_argument_variables(thread_id, frame_id)?
                    .len(),
            )
        };
        let globals = debugger.thread_global_variables(thread_id)?;
        let output_count = debugger.thread_shader_outputs(thread_id)?.len();

        let mut scopes = Vec::new();
        if !frame_reference.is_shader_outputs() {
            scopes.push(dapts::Scope {
                name: "Locals".to_string(),
                variables_reference: self
                    .references
                    .insert_scope(frame_reference, ScopeKind::Locals),
                named_variables: Some(local_count as u32),
                indexed_variables: None,
                expensive: false,
                source: None,
                line: None,
                end_line: None,
                column: None,
                end_column: None,
                presentation_hint: Some(dapts::ScopePresentationHint::Locals),
            });
        }

        if argument_count > 0 {
            scopes.push(dapts::Scope {
                name: "Function Arguments".to_string(),
                variables_reference: self
                    .references
                    .insert_scope(frame_reference, ScopeKind::Arguments),
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
                variables_reference: self
                    .references
                    .insert_scope(frame_reference, ScopeKind::Globals),
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

        if output_count > 0 {
            scopes.push(dapts::Scope {
                name: "Shader Outputs".to_string(),
                variables_reference: self
                    .references
                    .insert_scope(frame_reference, ScopeKind::ShaderOutputs),
                named_variables: Some(output_count as u32),
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

        Ok(vec![self.make_response(
            req.seq,
            &req.command,
            &dapts::ScopesResponse { scopes },
        )?])
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
            &req.command,
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

        let breakpoints = arguments
            .breakpoints
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(i, bp)| Breakpoint {
                id: Some(i as BreakpointId + 1),
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
            .collect::<Vec<_>>();

        let mut trace = Vec::new();
        if self.trace_enabled {
            let source_path = arguments.source.path.as_deref().unwrap_or("<none>");
            let source_name = arguments.source.name.as_deref().unwrap_or("<none>");
            let lines = breakpoints
                .iter()
                .filter_map(|bp| bp.line)
                .map(|line| line.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            trace.push(format!(
                "setBreakpoints source_matches={source_matches} source_name={source_name} source_path={source_path} lines=[{lines}]"
            ));
        }

        if source_matches {
            self.breakpoints = breakpoints.clone();
        }

        let mut messages = self.make_trace_events(&trace)?;
        messages.push(self.make_response(
            req.seq,
            &req.command,
            &dapts::SetBreakpointsResponse { breakpoints },
        )?);
        Ok(messages)
    }

    fn handle_configuration_done(
        &mut self,
        seq: i64,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        self.configuration_done = true;
        let mut messages =
            vec![self.make_response(seq, "configurationDone", &serde_json::json!({}))?];

        if let Some(delayed_init_seq) = self.delayed_init_seq.take() {
            messages.push(self.make_response(
                delayed_init_seq,
                "launch",
                &serde_json::json!({}),
            )?);
        }

        messages.push(self.initial_stop_event()?);
        Ok(messages)
    }

    fn handle_threads(&mut self, seq: i64) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let threads = self
            .debugger()?
            .threads()
            .into_iter()
            .map(|thread| dapts::Thread {
                id: thread.id,
                name: thread.name,
            })
            .collect();
        Ok(vec![self.make_response(
            seq,
            "threads",
            &dapts::ThreadsResponse { threads },
        )?])
    }

    fn handle_next(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let arguments = serde_json::from_value::<dapts::NextArguments>(req.arguments.clone())?;
        let single_thread =
            self.single_thread_execution || arguments.single_thread.unwrap_or(false);
        let thread_id: DebugThreadId = arguments.thread_id;
        self.debugger_mut()?.focus_thread(thread_id)?;
        self.references.clear();
        let breakpoints = self.breakpoint_lines();
        let result = self
            .debugger_mut()?
            .step_over(thread_id, single_thread, &breakpoints)?;
        let response = self.make_response(req.seq, &req.command, &serde_json::json!({}))?;
        let event = self.make_run_event(result)?;
        Ok(vec![response, event])
    }

    fn handle_continue(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let arguments = serde_json::from_value::<dapts::ContinueArguments>(req.arguments.clone())?;
        let single_thread =
            self.single_thread_execution || arguments.single_thread.unwrap_or(false);
        let thread_id: DebugThreadId = arguments.thread_id;
        self.debugger_mut()?.focus_thread(thread_id)?;
        self.references.clear();
        let response = self.make_response(
            req.seq,
            &req.command,
            &serde_json::json!({"allThreadsContinued": !single_thread}),
        )?;
        let mut trace = Vec::new();
        let event = self.run_to_breakpoint(thread_id, single_thread, &mut trace)?;

        let mut messages = vec![response];
        messages.extend(self.make_trace_events(&trace)?);
        messages.push(event);
        Ok(messages)
    }

    fn handle_disconnect(&mut self, seq: i64) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        self.debugger = None;
        self.outputs_presented = false;
        self.references.clear();
        Ok(vec![self.make_response(
            seq,
            "disconnect",
            &serde_json::json!({}),
        )?])
    }

    fn handle_terminate(&mut self, seq: i64) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        self.debugger = None;
        self.outputs_presented = false;
        self.references.clear();
        Ok(vec![
            self.make_response(seq, "terminate", &serde_json::json!({}))?,
            self.make_event("terminated", &serde_json::json!({}))?,
        ])
    }

    fn handle_variables(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let argument = serde_json::from_value::<dapts::VariablesArguments>(req.arguments.clone())?;
        let (frame, kind) = self.references.scope(argument.variables_reference)?;
        let debugger = self.debugger()?;
        let make_variables = |variables: Vec<malkovri_wgsl_debugger::Variable>| {
            variables
                .into_iter()
                .map(|variable| make_variable(variable.name, &format!("{:?}", variable.value)))
                .collect::<Vec<_>>()
        };
        let variables = match kind {
            ScopeKind::Locals => make_variables(
                debugger.frame_local_variables(frame.thread_id(), frame.frame_id()?)?,
            ),
            ScopeKind::Arguments => make_variables(
                debugger.frame_argument_variables(frame.thread_id(), frame.frame_id()?)?,
            ),
            ScopeKind::Globals => {
                make_variables(debugger.thread_global_variables(frame.thread_id())?)
            }
            ScopeKind::ShaderOutputs => debugger
                .thread_shader_outputs(frame.thread_id())?
                .into_iter()
                .map(|output| {
                    let value = output.value.map_or_else(
                        || "not produced yet".to_string(),
                        |value| format!("{value:?}"),
                    );
                    let mut variable = make_variable(Some(output.name), &value);
                    // Interface annotations are labels, not evaluatable expressions.
                    variable.evaluate_name = None;
                    variable
                })
                .collect(),
        };

        Ok(vec![self.make_response(
            req.seq,
            &req.command,
            &dapts::VariablesResponse { variables },
        )?])
    }

    fn breakpoint_lines(&self) -> Vec<u32> {
        self.breakpoints
            .iter()
            .filter(|bp| bp.verified)
            .filter_map(|bp| bp.line)
            .collect()
    }

    fn run_to_breakpoint(
        &mut self,
        thread_id: DebugThreadId,
        single_thread: bool,
        trace: &mut Vec<String>,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        let breakpoints = self.breakpoint_lines();
        let trace = self.trace_enabled.then_some(trace);
        let result = self.debugger_mut()?.run_to_breakpoint(
            thread_id,
            single_thread,
            &breakpoints,
            trace,
        )?;
        self.make_run_event(result)
    }

    fn make_run_event(&mut self, result: RunResult) -> Result<OutgoingMessage, DebugAdapterError> {
        use dapts::StoppedEventReason as Reason;
        let (reason, description) = match result {
            RunResult::Finished => {
                let debugger = self.debugger()?;
                let has_outputs = debugger.threads().into_iter().any(|thread| {
                    debugger
                        .thread_shader_outputs(thread.id)
                        .is_ok_and(|outputs| outputs.iter().any(|output| output.value.is_some()))
                });
                if self.outputs_presented || !has_outputs {
                    return self.make_event("terminated", &serde_json::json!({}));
                }
                self.outputs_presented = true;
                (
                    Reason::Pause,
                    Some(
                        "Shader outputs are available for inspection. Continue to finish the session.",
                    ),
                )
            }
            RunResult::Step => (Reason::Step, None),
            RunResult::Breakpoint => (Reason::Breakpoint, None),
            RunResult::InvocationFinished => (
                Reason::Pause,
                Some("Selected invocation finished; other invocations remain paused."),
            ),
            RunResult::Waiting => (
                Reason::Pause,
                Some("Selected invocation is waiting for other invocations at synchronization."),
            ),
            RunResult::BudgetExhausted => (
                Reason::Pause,
                Some("Execution paused after reaching the step budget; continue to resume."),
            ),
        };
        self.make_stopped_event_with_description(reason, description)
    }

    fn initial_stop_event(&mut self) -> Result<OutgoingMessage, DebugAdapterError> {
        if self.stop_on_entry {
            self.make_stopped_event(dapts::StoppedEventReason::Entry)
        } else {
            let thread_id = self.debugger()?.focused_thread_id();
            match self.run_to_breakpoint(thread_id, false, &mut Vec::new()) {
                Ok(event) => Ok(event),
                Err(error) => self.make_execution_error_event(&error),
            }
        }
    }

    fn make_trace_events(
        &mut self,
        trace: &[String],
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        if !self.trace_enabled {
            return Ok(Vec::new());
        }

        trace
            .iter()
            .map(|line| {
                self.make_event(
                    "output",
                    &serde_json::json!({
                        "category": "console",
                        "output": format!("[wgsl-debugger] {line}\n"),
                    }),
                )
            })
            .collect()
    }

    fn make_stopped_event(
        &mut self,
        reason: dapts::StoppedEventReason,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        self.make_stopped_event_with_description(reason, None)
    }

    fn make_execution_error_event(
        &mut self,
        error: &DebugAdapterError,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        let reason = if matches!(
            error,
            DebugAdapterError::Evaluator(EvaluatorError::ExecutionBudgetExceeded)
        ) {
            dapts::StoppedEventReason::Pause
        } else {
            dapts::StoppedEventReason::Exception
        };
        self.make_stopped_event_with_description(reason, Some(&error.to_string()))
    }

    fn make_stopped_event_with_description(
        &mut self,
        reason: dapts::StoppedEventReason,
        description: Option<&str>,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        self.make_event(
            "stopped",
            &dapts::StoppedEvent {
                reason,
                description: description.map(str::to_owned),
                thread_id: self.debugger.as_ref().map(Debugger::focused_thread_id),
                preserve_focus_hint: None,
                text: None,
                all_threads_stopped: Some(true),
                hit_breakpoint_ids: None,
            },
        )
    }

    pub(crate) fn next_sequence_number(&mut self) -> i64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }
}
