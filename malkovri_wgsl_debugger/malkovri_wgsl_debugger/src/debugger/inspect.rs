use naga::Statement;

use crate::{
    error::EvaluatorError,
    invocation::InvocationState,
    invocation::frame::{FrameContext, StackFrame},
    value::Value,
};

use super::{
    DebugFrameId, DebugThreadId, Debugger, InvocationId, SourceLocation, StackFrameInfo,
    ThreadStatus, Variable,
};

impl Debugger {
    pub fn current_location(&self) -> Option<SourceLocation> {
        if self.is_empty() {
            return None;
        }
        self.location_for_gid(self.focused_thread)
    }

    pub fn thread_current_location(&self, thread_id: DebugThreadId) -> Option<SourceLocation> {
        let id = self.group.resolve(thread_id).ok()?;
        self.location_for_gid(id)
    }

    pub fn all_thread_locations(&self) -> Vec<(DebugThreadId, SourceLocation)> {
        self.group
            .ids()
            .filter_map(|id| {
                self.location_for_gid(id)
                    .map(|location| (id.thread_id(), location))
            })
            .collect()
    }

    fn location_for_gid(&self, id: InvocationId) -> Option<SourceLocation> {
        let invocation = self.group.get(id);
        if matches!(invocation.status(), ThreadStatus::Finished) {
            return None;
        }
        let invocation = invocation.state();
        let context = invocation
            .frame_context(invocation.current_function_frame_index().ok()?)
            .ok()?;
        self.frame_location(invocation, context)
    }

    fn frame_location(
        &self,
        invocation: &InvocationState,
        context: FrameContext,
    ) -> Option<SourceLocation> {
        let StackFrame::Function(frame) = &invocation.stack()[context.function_index()] else {
            return None;
        };
        let function = invocation.resolve_function(&frame.function_id());
        let function_name = function.name.clone();
        let (current_statement, span) = self
            .program
            .block(invocation.stack()[context.block_index()].block())
            .span_iter()
            .nth(context.statement_index())?;

        let (line, column) = if matches!(current_statement.leaf(), Some(Statement::Return { .. }))
            && span.to_range().is_none()
        {
            // For return statements, point to the line after the closing brace
            // of the function body rather than the span of the return itself.
            let func = function;
            let total_span = naga::Span::total_span(func.body.span_iter().map(|(_, s)| *s));
            let total_range = total_span.to_range()?;
            let prefix = &self.source()[..total_range.end];
            let line_number = prefix.matches('\n').count() as u32 + 2;
            (line_number, 0)
        } else {
            let loc = span.location(self.source());
            (loc.line_number, loc.line_position)
        };

        Some(SourceLocation {
            line,
            column,
            function_name,
        })
    }

    /// Active call stack frames, from innermost (current) to outermost (entry point).
    pub fn call_stack(&self) -> Vec<StackFrameInfo> {
        self.thread_call_stack(self.focused_thread_id())
            .unwrap_or_default()
    }

    /// Inspect an invocation without changing the focused thread.
    pub fn thread_call_stack(
        &self,
        thread_id: DebugThreadId,
    ) -> Result<Vec<StackFrameInfo>, EvaluatorError> {
        let invocation = self.invocation_for_thread(thread_id)?;
        invocation
            .stack()
            .iter()
            .enumerate()
            .rev()
            .filter_map(|(index, stack_frame)| {
                let StackFrame::Function(frame) = stack_frame else {
                    return None;
                };
                Some(invocation.frame_context(index).map(|context| {
                    StackFrameInfo {
                        id: DebugFrameId(index),
                        name: invocation
                            .resolve_function(&frame.function_id())
                            .name
                            .clone(),
                        location: self.frame_location(invocation, context),
                    }
                }))
            })
            .collect()
    }

    /// All local variables and `let` bindings visible at the current execution point.
    pub fn local_variables(&self) -> Vec<Variable> {
        if self.is_empty() {
            return Vec::new();
        }
        let Ok(index) = self.invocation().current_function_frame_index() else {
            return Vec::new();
        };
        self.frame_local_variables(self.focused_thread_id(), DebugFrameId(index))
            .unwrap_or_default()
    }

    pub fn frame_local_variables(
        &self,
        thread_id: DebugThreadId,
        frame_id: DebugFrameId,
    ) -> Result<Vec<Variable>, EvaluatorError> {
        let invocation = self.invocation_for_thread(thread_id)?;
        let context = invocation.frame_context(frame_id.0)?;
        let StackFrame::Function(frame) = &invocation.stack()[context.function_index()] else {
            unreachable!()
        };
        let function = invocation.resolve_function(&frame.function_id());
        let in_scope = invocation.local_variables_in_scope(context)?;
        let mut variables: Vec<_> = function
            .local_variables
            .iter()
            .filter(|(handle, _)| in_scope.contains(handle))
            .map(|(handle, local)| Variable {
                name: local.name.clone(),
                value: invocation.evaluate_local_variable(handle, context.function_index()),
            })
            .collect();
        variables.extend(
            invocation
                .named_expression_values(context)?
                .into_iter()
                .map(|(name, value)| Variable {
                    name: Some(name),
                    value,
                }),
        );
        Ok(variables)
    }

    /// Current function arguments with their names and values.
    pub fn argument_variables(&self) -> Vec<Variable> {
        if self.is_empty() {
            return Vec::new();
        }
        let Ok(index) = self.invocation().current_function_frame_index() else {
            return Vec::new();
        };
        self.frame_argument_variables(self.focused_thread_id(), DebugFrameId(index))
            .unwrap_or_default()
    }

    pub fn frame_argument_variables(
        &self,
        thread_id: DebugThreadId,
        frame_id: DebugFrameId,
    ) -> Result<Vec<Variable>, EvaluatorError> {
        let invocation = self.invocation_for_thread(thread_id)?;
        let context = invocation.frame_context(frame_id.0)?;
        Ok(invocation
            .function_argument_values(context)?
            .into_iter()
            .map(|(name, value)| Variable { name, value })
            .collect())
    }

    pub fn thread_global_variables(
        &self,
        thread_id: DebugThreadId,
    ) -> Result<Vec<Variable>, EvaluatorError> {
        Ok(self
            .invocation_for_thread(thread_id)?
            .global_variable_values()
            .into_iter()
            .map(|(name, value)| Variable { name, value })
            .collect())
    }

    /// All global variables with their names and values.
    pub fn global_variables(&self) -> Vec<Variable> {
        if self.is_empty() {
            return Vec::new();
        }
        self.invocation()
            .global_variable_values()
            .into_iter()
            .map(|(name, value)| Variable { name, value })
            .collect()
    }

    pub fn entry_point_output(&self) -> Option<Value> {
        if self.is_empty() {
            return None;
        }
        self.invocation().entry_point_output().cloned()
    }
}
