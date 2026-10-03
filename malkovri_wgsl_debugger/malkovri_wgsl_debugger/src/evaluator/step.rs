use crate::program::FunctionId;
use crate::{
    error::EvaluatorError,
    function_state::{BlockFrame, BlockKind, ControlFlow, StackFrame},
    program::{Instruction, StatementId},
    value::Value,
};

use std::sync::Arc;

use super::InvocationState;

impl InvocationState {
    // Core execution loop
    /// Advance past any pending control-flow signals and exhausted frames until
    /// a live statement is ready to execute (or the stack is empty).
    fn advance_to_live_statement(&mut self) -> Result<bool, EvaluatorError> {
        // An empty infinite loop has no live statement to yield to the scheduler.
        // Bound normalization as well as the outer execution loop.
        for _ in 0..100_000 {
            if self.stack.is_empty() {
                return Ok(false);
            }

            // Phase 1: Resolve pending control-flow signals.
            if self.resolve_control_flow_signal()? {
                continue;
            }

            // Phase 2: Pop exhausted frames.
            if self.pop_if_exhausted()? {
                continue;
            }

            return Ok(true);
        }
        Err(EvaluatorError::ExecutionBudgetExceeded)
    }

    /// If the current function frame has a pending control-flow signal, apply it
    /// and return `true`. Otherwise return `false`.
    fn resolve_control_flow_signal(&mut self) -> Result<bool, EvaluatorError> {
        let function_index = self.current_function_frame_index()?;
        let StackFrame::Function(ref mut frame) = self.stack[function_index] else {
            return Err(EvaluatorError::InternalError(
                "expected function frame at function_index".into(),
            ));
        };
        let signal = std::mem::take(&mut frame.control_flow);
        match signal {
            ControlFlow::None => Ok(false),
            ControlFlow::Break => {
                self.apply_break(function_index);
                Ok(true)
            }
            ControlFlow::Continue => {
                self.apply_continue(function_index);
                Ok(true)
            }
            ControlFlow::Return(return_val) => {
                self.apply_return(function_index, return_val);
                Ok(true)
            }
        }
    }

    /// If the top frame is exhausted, handle it and return `true`.
    fn pop_if_exhausted(&mut self) -> Result<bool, EvaluatorError> {
        let is_exhausted = self.current_frame()?.is_exhausted(&self.program);
        if is_exhausted {
            self.handle_exhausted_frame();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Execute the current statement and advance the program counter.
    /// Returns the *upcoming* statement that will execute on the next call,
    /// or `None` if execution has finished.
    pub(crate) fn step(&mut self) -> Result<Option<StatementId>, EvaluatorError> {
        if !self.advance_to_live_statement()? {
            return Ok(None);
        }

        let caller_frame_index = self.current_frame_index()?;

        let id = self.current_frame()?.position();
        let program = Arc::clone(&self.program);
        self.handle_instruction(id, program.instruction(id))?;

        self.stack[caller_frame_index].increment_statement_index();

        // Resolve any signals/exhaustion produced by the statement we just ran,
        // then return whatever is live next (or None if execution finished).
        self.advance_to_live_statement()?;

        Ok(self.peek_next_statement())
    }

    pub(crate) fn current_statement(&mut self) -> Result<Option<StatementId>, EvaluatorError> {
        if !self.advance_to_live_statement()? {
            return Ok(None);
        }

        Ok(self.peek_next_statement())
    }

    pub(crate) fn consume_current_statement_without_running(
        &mut self,
    ) -> Result<Option<StatementId>, EvaluatorError> {
        if !self.advance_to_live_statement()? {
            return Ok(None);
        }

        let current_frame_index = self.current_frame_index()?;
        self.stack[current_frame_index].increment_statement_index();
        self.advance_to_live_statement()?;

        Ok(self.peek_next_statement())
    }

    pub(crate) fn consume_current_statement_and_skip_emits(
        &mut self,
    ) -> Result<Option<StatementId>, EvaluatorError> {
        let mut next = self.consume_current_statement_without_running()?;
        while next.is_some_and(|id| self.program.instruction(id).is_emit()) {
            next = self.step()?;
        }
        Ok(next)
    }

    fn peek_next_statement(&self) -> Option<StatementId> {
        let id = self.current_frame().ok()?.position();
        self.program.block(id.block).get(id.index)?;
        Some(id)
    }

    // Control-flow signal handlers
    /// Pop block frames above `function_index` until (and including) the nearest `Loop` or `Switch` frame.
    fn apply_break(&mut self, function_index: usize) {
        while self.stack.len() > function_index + 1 {
            let is_target = matches!(
                self.current_frame().ok(),
                Some(StackFrame::Block(BlockFrame {
                    kind: BlockKind::Loop { .. } | BlockKind::Switch { .. },
                    ..
                }))
            );
            self.stack.pop();
            if is_target {
                break;
            }
        }
    }

    /// Pop block frames above `function_index` until the nearest `Loop` frame (keep it), then switch
    /// it to its `continuing` block.
    fn apply_continue(&mut self, function_index: usize) {
        while self.stack.len() > function_index + 1 {
            if matches!(
                self.current_frame().ok(),
                Some(StackFrame::Block(BlockFrame {
                    kind: BlockKind::Loop { .. },
                    ..
                }))
            ) {
                break;
            }
            self.stack.pop();
        }

        // Switch the loop frame to its continuing block.
        if let Ok(top) = self.current_frame_index()
            && top > function_index
            && let StackFrame::Block(ref mut block_frame) = self.stack[top]
        {
            block_frame.switch_to_continuing();
        }
    }

    /// Store the return value in the parent function frame, then truncate the stack
    /// to remove the returning function and everything above it.
    fn apply_return(&mut self, function_index: usize, value: Option<Value>) {
        // Read the result handle from the callee before truncating the stack.
        let (function_id, call_result_handle) = match &self.stack[function_index] {
            StackFrame::Function(frame) => (Some(frame.function_id), frame.call_result_handle),
            StackFrame::Block(_) => (None, None),
        };
        // Store the return value in the parent frame's expression cache, keyed
        // by the CallResult expression handle so that each call's result is
        // independently retrievable even when multiple calls appear in sequence.
        if let (Some(handle), Some(return_val)) = (call_result_handle, value.clone())
            && let Some(parent_function_index) = self.parent_function_frame_index(function_index)
            && let StackFrame::Function(ref mut parent_frame) = self.stack[parent_function_index]
        {
            parent_frame
                .evaluated_expressions
                .insert(handle, return_val.into());
        } else if matches!(function_id, Some(FunctionId::EntryPoint(_))) {
            self.entry_point_output = value;
        }
        self.stack.truncate(function_index);
    }

    // Exhausted-frame handler
    fn handle_exhausted_frame(&mut self) {
        let Ok(top) = self.current_frame_index() else {
            return;
        };
        match &self.stack[top] {
            StackFrame::Function(_) => {
                self.stack.pop();
            }
            StackFrame::Block(block_frame) => match &block_frame.kind {
                BlockKind::Plain => {
                    self.stack.pop();
                }
                BlockKind::Switch {
                    statement,
                    next_case,
                } => {
                    let next = next_case.map(|index| {
                        let Instruction::Switch { cases, .. } =
                            self.program.instruction(*statement)
                        else {
                            unreachable!("switch frame must refer to a switch instruction")
                        };
                        let case = &cases[index];
                        BlockFrame {
                            block: case.body,
                            current_statement_index: 0,
                            kind: BlockKind::Switch {
                                statement: *statement,
                                next_case: case.fall_through.then_some(index + 1),
                            },
                        }
                    });
                    self.stack.pop();
                    if let Some(frame) = next {
                        self.stack.push(StackFrame::Block(frame));
                    }
                }
                BlockKind::Loop {
                    in_continuing: false,
                    ..
                } => {
                    if let StackFrame::Block(ref mut bf) = self.stack[top] {
                        bf.switch_to_continuing();
                    }
                }
                BlockKind::Loop {
                    in_continuing: true,
                    break_if,
                    ..
                } => {
                    let should_break = if let Some(expr) = break_if {
                        self.evaluate_expression(*expr).is_truthy()
                    } else {
                        false
                    };

                    if should_break {
                        self.stack.pop();
                    } else if let StackFrame::Block(ref mut bf) = self.stack[top] {
                        bf.restart_body();
                    }
                }
            },
        }
    }
}
