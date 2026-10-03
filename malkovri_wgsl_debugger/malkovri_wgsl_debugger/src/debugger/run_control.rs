use crate::error::EvaluatorError;

use super::{DebugThreadId, Debugger, StepResult, ThreadState};

// Bound synchronous requests so native and browser hosts regain control.
const EXECUTION_STEP_BUDGET: usize = 100_000;
// Breakpoint catch-up is a source-line heuristic, so it must also be bounded.
const BREAKPOINT_CATCH_UP_STEP_BUDGET: usize = 100_000;

/// Why a source-level execution request returned. The debugger retains its state
/// for inspection or another request, including after all invocations finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunResult {
    /// Every invocation finished.
    Finished,
    /// Step Over reached another line at the same or a shallower call depth.
    Step,
    /// The focused invocation reached a breakpoint.
    Breakpoint,
    /// The selected invocation finished, while other invocations remain.
    InvocationFinished,
    /// The selected invocation needs other invocations to reach synchronization.
    Waiting,
    /// Execution can resume with another request.
    BudgetExhausted,
}

impl Debugger {
    /// Advance past a source line and its calls, honoring breakpoints in callees.
    /// Breakpoints are one-based lines in this program's source. If `single_thread`
    /// is false, all invocations advance and any invocation can hit a breakpoint.
    pub fn step_over(
        &mut self,
        thread_id: DebugThreadId,
        single_thread: bool,
        breakpoints: &[u32],
    ) -> Result<RunResult, EvaluatorError> {
        self.focus_thread(thread_id)?;
        let initial_depth = self.call_stack().len();
        let initial_line = self.current_location().map(|location| location.line);

        for _ in 0..EXECUTION_STEP_BUDGET {
            if self.advance(thread_id, single_thread)? == StepResult::Finished {
                return Ok(RunResult::Finished);
            }
            match self.thread_state(thread_id)? {
                ThreadState::Finished => return Ok(RunResult::InvocationFinished),
                ThreadState::Waiting if single_thread => return Ok(RunResult::Waiting),
                _ => {}
            }

            let depth = self.thread_call_stack(thread_id)?.len();
            let current_line = self
                .thread_current_location(thread_id)
                .map(|location| location.line);
            let hit = self
                .all_thread_locations()
                .into_iter()
                .find(|(id, location)| {
                    (!single_thread || *id == thread_id)
                        && (*id != thread_id
                            || Some(location.line) != initial_line
                            || depth > initial_depth)
                        && breakpoints.contains(&location.line)
                });
            if let Some((id, _)) = hit {
                self.focus_thread(id)?;
                return Ok(RunResult::Breakpoint);
            }
            if depth <= initial_depth && current_line != initial_line {
                return Ok(RunResult::Step);
            }
        }
        Ok(RunResult::BudgetExhausted)
    }

    /// Continue until a breakpoint, completion, synchronization wait, or budget
    /// limit. Pass an empty breakpoint slice to run toward completion. Optional
    /// trace output describes execution without depending on a frontend protocol.
    pub fn run_to_breakpoint(
        &mut self,
        thread_id: DebugThreadId,
        single_thread: bool,
        breakpoints: &[u32],
        mut trace: Option<&mut Vec<String>>,
    ) -> Result<RunResult, EvaluatorError> {
        self.focus_thread(thread_id)?;
        if let Some(trace) = trace.as_mut() {
            let lines = breakpoints
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            trace.push(format!(
                "continue start thread={thread_id} single_thread={single_thread} breakpoints=[{lines}] locations={}",
                self.format_thread_locations(),
            ));
        }

        for step in 1..=EXECUTION_STEP_BUDGET {
            let result = self.advance(thread_id, single_thread)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(format!(
                    "continue step {step}: result={result:?} locations={}",
                    self.format_thread_locations(),
                ));
            }
            if result == StepResult::Finished {
                return Ok(RunResult::Finished);
            }
            if single_thread {
                match self.thread_state(thread_id)? {
                    ThreadState::Finished => return Ok(RunResult::InvocationFinished),
                    ThreadState::Waiting => return Ok(RunResult::Waiting),
                    ThreadState::Running => {}
                }
            }
            let hit = if single_thread {
                self.thread_current_location(thread_id)
                    .filter(|location| breakpoints.contains(&location.line))
                    .map(|location| (thread_id, location))
            } else {
                self.all_thread_locations()
                    .into_iter()
                    .find(|(_, location)| breakpoints.contains(&location.line))
            };
            if let Some((candidate, location)) = hit {
                let line = location.line;
                if let Some(trace) = trace.as_mut() {
                    trace.push(format!(
                        "breakpoint candidate thread={candidate} line={line}"
                    ));
                }
                let hit_thread = if single_thread {
                    candidate
                } else {
                    self.catch_up_threads_to_breakpoint(candidate, line, trace.as_deref_mut())?
                };
                self.focus_thread(hit_thread)?;
                return Ok(RunResult::Breakpoint);
            }
        }
        Ok(RunResult::BudgetExhausted)
    }

    fn advance(
        &mut self,
        thread_id: DebugThreadId,
        single_thread: bool,
    ) -> Result<StepResult, EvaluatorError> {
        if single_thread {
            self.step_thread(thread_id)
        } else {
            self.step_all()
        }
    }

    // Keep invocations already at the breakpoint there, while allowing invocations
    // on earlier source lines to catch up. Source order is only a heuristic for
    // reconvergence: loops and calls need not move monotonically through the file.
    fn catch_up_threads_to_breakpoint(
        &mut self,
        hit_thread_id: DebugThreadId,
        target_line: u32,
        mut trace: Option<&mut Vec<String>>,
    ) -> Result<DebugThreadId, EvaluatorError> {
        let mut remaining_step_budget = BREAKPOINT_CATCH_UP_STEP_BUDGET;
        loop {
            let locations = self.all_thread_locations();
            let first_at_target = locations
                .iter()
                .find_map(|(thread_id, loc)| (loc.line == target_line).then_some(*thread_id));

            if locations.iter().all(|(_, loc)| loc.line == target_line) {
                if let Some(trace) = trace.as_mut() {
                    trace.push(format!(
                        "catch-up complete target_line={target_line} locations={}",
                        self.format_thread_locations(),
                    ));
                }
                return Ok(first_at_target.unwrap_or(hit_thread_id));
            }

            let candidates = locations
                .iter()
                .filter_map(|(thread_id, loc)| (loc.line < target_line).then_some(*thread_id))
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                if let Some(trace) = trace.as_mut() {
                    trace.push(format!(
                        "catch-up stopped target_line={target_line}; no candidates locations={}",
                        self.format_thread_locations(),
                    ));
                }
                return Ok(first_at_target.unwrap_or(hit_thread_id));
            }

            for candidate_thread_id in candidates {
                self.step_thread(candidate_thread_id)?;
                if let Some(trace) = trace.as_mut() {
                    trace.push(format!(
                        "catch-up stepped thread={candidate_thread_id} locations={}",
                        self.format_thread_locations(),
                    ));
                }
                remaining_step_budget -= 1;
                if remaining_step_budget == 0 {
                    return Ok(first_at_target.unwrap_or(hit_thread_id));
                }
            }
        }
    }

    fn format_thread_locations(&self) -> String {
        self.all_thread_locations()
            .into_iter()
            .map(|(thread_id, loc)| {
                let function = loc.function_name.as_deref().unwrap_or("unknown");
                format!("{thread_id}:{function}:{}:{}", loc.line, loc.column)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}
