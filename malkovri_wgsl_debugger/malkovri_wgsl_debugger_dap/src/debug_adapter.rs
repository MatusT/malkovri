use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[cfg(not(target_arch = "wasm32"))]
use std::fs;

use dapts::Breakpoint;

use crate::error::DebugAdapterError;
use crate::parse_input;
use crate::protocol::{
    ARGUMENTS_SCOPE_REF, BreakpointId, GLOBALS_SCOPE_REF, LOCALS_SCOPE_REF, OutgoingMessage,
    StackFrameId, make_scope_ref, make_variable, parse_scope_ref,
};
use malkovri_wgsl_debugger::{DebugThreadId, Debugger, StepResult};

// Defensive UI budget, not shader semantics: if catch-up cannot settle, stop anyway.
const BREAKPOINT_CATCH_UP_STEP_BUDGET: usize = 100_000;

#[derive(Clone, Copy, Debug)]
struct FrameReference {
    thread_id: DebugThreadId,
}

pub struct DebugAdapter {
    sequence_number: i64,
    next_frame_id: StackFrameId,
    frame_references: HashMap<StackFrameId, FrameReference>,
    breakpoints: Vec<Breakpoint>,
    program_name: Option<String>,
    program_path: Option<PathBuf>,
    debugger: Option<Debugger>,
    delayed_init_seq: Option<i64>,
    configuration_done: bool,
    stop_on_entry: bool,
    single_thread_execution: bool,
    trace_enabled: bool,
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
            next_frame_id: 1,
            frame_references: HashMap::new(),
            breakpoints: Vec::new(),
            debugger: None,
            program_name: None,
            program_path: None,
            delayed_init_seq: None,
            configuration_done: false,
            stop_on_entry: false,
            single_thread_execution: false,
            trace_enabled: false,
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

    fn register_frame_reference(&mut self, thread_id: DebugThreadId) -> StackFrameId {
        let frame_id = self.next_frame_id;
        self.next_frame_id += 1;
        self.frame_references
            .insert(frame_id, FrameReference { thread_id });
        frame_id
    }

    fn frame_reference(&self, frame_id: StackFrameId) -> Result<FrameReference, DebugAdapterError> {
        self.frame_references
            .get(&frame_id)
            .copied()
            .ok_or_else(|| {
                DebugAdapterError::InvalidProgram(format!(
                    "unknown stack frame id {frame_id}; request stackTrace before scopes"
                ))
            })
    }

    fn handle_initialize(&mut self, seq: i64) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        Ok(vec![
            self.make_response(
                seq,
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

        self.program_path = Some(program_path.clone());
        self.program_name = Some(program_name);
        self.stop_on_entry = arguments
            .get("stopOnEntry")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        self.single_thread_execution = arguments
            .get("singleThreadExecution")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        self.trace_enabled = arguments
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

        let workgroup_config = parse_input::parse_workgroup_config(arguments)?;
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

        self.debugger = Some(Debugger::new(
            &source,
            0,
            workgroup_config,
            global_constants,
            bindings,
        )?);
        self.frame_references.clear();
        self.next_frame_id = 1;

        let mut messages = Vec::new();
        if !self.configuration_done {
            self.delayed_init_seq = Some(req.seq);
        } else {
            messages.push(self.make_response(req.seq, &serde_json::json!({}))?);
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
        let frames = {
            let debugger = self.debugger_mut()?;
            debugger.focus_thread(thread_id)?;
            debugger.call_stack()
        };
        let path = self
            .program_path
            .as_ref()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("program_path not set".into()))?
            .to_string_lossy()
            .to_string();

        let mut stack_frames = Vec::new();
        for frame in &frames {
            let frame_id = self.register_frame_reference(thread_id);
            let location = frame.location.as_ref().or_else(|| {
                frames
                    .first()
                    .and_then(|innermost| innermost.location.as_ref())
            });
            let line = location.map(|loc| loc.line).unwrap_or(1);
            let column = location.map(|loc| loc.column).unwrap_or(0);
            stack_frames.push(dapts::StackFrame {
                id: frame_id,
                name: frame.name.as_deref().unwrap_or("main").to_string(),
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
        let frame_reference = self.frame_reference(frame_id)?;
        let thread_id = frame_reference.thread_id;
        let debugger = self.debugger_mut()?;
        debugger.focus_thread(thread_id)?;

        let local_count = debugger.local_variables().len();
        let argument_count = debugger.argument_variables().len();
        let globals = debugger.global_variables();

        let mut scopes = vec![dapts::Scope {
            name: "Locals".to_string(),
            variables_reference: make_scope_ref(thread_id, LOCALS_SCOPE_REF),
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
                variables_reference: make_scope_ref(thread_id, ARGUMENTS_SCOPE_REF),
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
                variables_reference: make_scope_ref(thread_id, GLOBALS_SCOPE_REF),
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

        Ok(vec![self.make_response(
            req.seq,
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
        messages.push(self.make_response(req.seq, &dapts::SetBreakpointsResponse { breakpoints })?);
        Ok(messages)
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
        Ok(vec![
            self.make_response(seq, &dapts::ThreadsResponse { threads })?,
        ])
    }

    fn handle_next(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let arguments = serde_json::from_value::<dapts::NextArguments>(req.arguments.clone())?;
        let single_thread =
            self.single_thread_execution || arguments.single_thread.unwrap_or(false);
        let thread_id: DebugThreadId = arguments.thread_id;
        let debugger = self.debugger_mut()?;
        debugger.focus_thread(thread_id)?;
        let has_more = if single_thread {
            matches!(debugger.step_thread(thread_id)?, StepResult::Continue)
        } else {
            matches!(debugger.step_all()?, StepResult::Continue)
        };

        let mut messages = vec![self.make_response(req.seq, &serde_json::json!({}))?];
        if has_more {
            messages.push(self.make_stopped_event(dapts::StoppedEventReason::Step)?);
        } else {
            messages.push(self.make_event("terminated", &serde_json::json!({}))?);
        }
        Ok(messages)
    }

    fn handle_continue(
        &mut self,
        req: &dapts::Request,
    ) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        let arguments = serde_json::from_value::<dapts::ContinueArguments>(req.arguments.clone())?;
        let single_thread =
            self.single_thread_execution || arguments.single_thread.unwrap_or(false);
        let response = self.make_response(req.seq, &serde_json::json!({}))?;
        let thread_id: DebugThreadId = arguments.thread_id;
        let mut trace = Vec::new();
        let event = self.run_to_breakpoint(thread_id, single_thread, &mut trace)?;

        let mut messages = vec![response];
        messages.extend(self.make_trace_events(&trace)?);
        messages.push(event);
        Ok(messages)
    }

    fn handle_disconnect(&mut self, seq: i64) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        self.debugger = None;
        self.frame_references.clear();
        Ok(vec![self.make_response(seq, &serde_json::json!({}))?])
    }

    fn handle_terminate(&mut self, seq: i64) -> Result<Vec<OutgoingMessage>, DebugAdapterError> {
        self.debugger = None;
        self.frame_references.clear();
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
        let (thread_id, scope_ref) = parse_scope_ref(argument.variables_reference);
        let debugger = self.debugger_mut()?;
        debugger.focus_thread(thread_id)?;

        let variables = match scope_ref {
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

        Ok(vec![self.make_response(
            req.seq,
            &dapts::VariablesResponse { variables },
        )?])
    }

    fn run_to_breakpoint(
        &mut self,
        thread_id: DebugThreadId,
        single_thread: bool,
        trace: &mut Vec<String>,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        let debugger = self
            .debugger
            .as_mut()
            .ok_or_else(|| DebugAdapterError::InvalidProgram("debugger not initialized".into()))?;
        debugger.focus_thread(thread_id)?;
        let breakpoints = &self.breakpoints;
        if self.trace_enabled {
            let lines = breakpoints
                .iter()
                .filter_map(|bp| bp.line)
                .map(|line| line.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            trace.push(format!(
                "continue start thread={thread_id} single_thread={single_thread} breakpoints=[{lines}] locations={}",
                Self::format_thread_locations(debugger),
            ));
        }

        let mut has_more = false;
        let mut hit_thread_id = thread_id;
        let mut steps = 0usize;
        loop {
            // Continue is implemented as repeated debugger steps until stop or termination.
            let result = if single_thread {
                debugger.step_thread(thread_id)?
            } else {
                debugger.step_all()?
            };
            steps += 1;
            if self.trace_enabled {
                trace.push(format!(
                    "continue step {steps}: result={result:?} locations={}",
                    Self::format_thread_locations(debugger),
                ));
            }
            match result {
                StepResult::Finished => break,
                StepResult::Continue => {
                    // Single-thread continue only inspects the selected DAP thread.
                    let hit = if single_thread {
                        debugger.thread_current_location(thread_id).and_then(|loc| {
                            Self::verified_breakpoint_line(breakpoints, loc.line)
                                .map(|line| (thread_id, line))
                        })
                    } else {
                        // Lockstep continue checks every thread because any lane may hit first.
                        Self::first_breakpoint_hit(debugger, breakpoints)
                    };

                    if let Some((candidate_thread_id, line)) = hit {
                        if self.trace_enabled {
                            trace.push(format!(
                                "breakpoint candidate thread={candidate_thread_id} line={line}"
                            ));
                        }
                        // Converged lines can be reached by fast lanes before slower lanes catch up.
                        if !single_thread {
                            hit_thread_id = Self::catch_up_threads_to_breakpoint(
                                debugger,
                                candidate_thread_id,
                                line,
                                trace,
                                self.trace_enabled,
                            )?;
                        } else {
                            hit_thread_id = candidate_thread_id;
                        }
                        debugger.focus_thread(hit_thread_id)?;
                        has_more = true;
                        break;
                    }
                }
            }
        }

        if has_more {
            debugger.focus_thread(hit_thread_id)?;
            self.make_stopped_event(dapts::StoppedEventReason::Breakpoint)
        } else {
            self.make_event("terminated", &serde_json::json!({}))
        }
    }

    // Return the verified breakpoint line when the current source line matches one.
    fn verified_breakpoint_line(breakpoints: &[Breakpoint], line: u32) -> Option<u32> {
        breakpoints
            .iter()
            .find_map(|bp| (bp.verified && bp.line == Some(line)).then_some(line))
    }

    // Find the first thread whose current line is a verified breakpoint.
    fn first_breakpoint_hit(
        debugger: &Debugger,
        breakpoints: &[Breakpoint],
    ) -> Option<(DebugThreadId, u32)> {
        debugger
            .all_thread_locations()
            .into_iter()
            .find_map(|(thread_id, loc)| {
                Self::verified_breakpoint_line(breakpoints, loc.line).map(|line| (thread_id, line))
            })
    }

    // Let lanes that are still before a hit breakpoint reach that same source line.
    fn catch_up_threads_to_breakpoint(
        debugger: &mut Debugger,
        hit_thread_id: DebugThreadId,
        target_line: u32,
        trace: &mut Vec<String>,
        trace_enabled: bool,
    ) -> Result<DebugThreadId, DebugAdapterError> {
        let mut remaining_step_budget = BREAKPOINT_CATCH_UP_STEP_BUDGET;

        loop {
            let locations = debugger.all_thread_locations();
            // Prefer reporting the earliest DAP thread already sitting on the breakpoint.
            let first_at_target = locations
                .iter()
                .find_map(|(thread_id, loc)| (loc.line == target_line).then_some(*thread_id));

            // If every live thread reports this line, VS Code will show a coherent stop.
            if locations.iter().all(|(_, loc)| loc.line == target_line) {
                if trace_enabled {
                    trace.push(format!(
                        "catch-up complete target_line={target_line} locations={}",
                        Self::format_thread_locations(debugger),
                    ));
                }
                return Ok(first_at_target.unwrap_or(hit_thread_id));
            }

            // Threads beyond the line are from divergent paths, so they cannot be caught up.
            let candidates = locations
                .iter()
                .filter_map(|(thread_id, loc)| (loc.line < target_line).then_some(*thread_id))
                .collect::<Vec<_>>();

            if candidates.is_empty() {
                if trace_enabled {
                    trace.push(format!(
                        "catch-up stopped target_line={target_line}; no candidates locations={}",
                        Self::format_thread_locations(debugger),
                    ));
                }
                return Ok(first_at_target.unwrap_or(hit_thread_id));
            }

            // Threads already at the breakpoint stay parked there. Threads still
            // visibly before the line may be draining divergent control flow that
            // reconverges at this breakpoint.
            for candidate_thread_id in candidates {
                debugger.step_thread(candidate_thread_id)?;
                if trace_enabled {
                    trace.push(format!(
                        "catch-up stepped thread={candidate_thread_id} locations={}",
                        Self::format_thread_locations(debugger),
                    ));
                }
                remaining_step_budget = remaining_step_budget.saturating_sub(1);
                if remaining_step_budget == 0 {
                    return Ok(first_at_target.unwrap_or(hit_thread_id));
                }
            }
        }
    }

    fn initial_stop_event(&mut self) -> Result<OutgoingMessage, DebugAdapterError> {
        if self.stop_on_entry {
            self.make_stopped_event(dapts::StoppedEventReason::Entry)
        } else {
            let thread_id = self.debugger()?.focused_thread_id();
            self.run_to_breakpoint(thread_id, false, &mut Vec::new())
        }
    }

    fn format_thread_locations(debugger: &Debugger) -> String {
        debugger
            .all_thread_locations()
            .into_iter()
            .map(|(thread_id, loc)| {
                let function = loc.function_name.unwrap_or_else(|| "unknown".to_string());
                format!("{thread_id}:{function}:{}:{}", loc.line, loc.column)
            })
            .collect::<Vec<_>>()
            .join(", ")
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
        self.make_event(
            "stopped",
            &dapts::StoppedEvent {
                reason,
                description: None,
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
