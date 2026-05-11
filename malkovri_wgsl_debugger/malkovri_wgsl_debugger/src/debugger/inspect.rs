use naga::Statement;

use crate::{function_state::StackFrame, value::Value};

use super::{Debugger, SourceLocation, StackFrameInfo, ThreadStatus, Variable};

impl Debugger {
    pub fn current_location(&self) -> Option<SourceLocation> {
        self.location_for_gid(self.focused_thread)
    }

    pub fn thread_current_location(&self, thread_id: u64) -> Option<SourceLocation> {
        let gid = *self.thread_ids.get(&thread_id)?;
        self.location_for_gid(gid)
    }

    pub fn all_thread_locations(&self) -> Vec<(u64, SourceLocation)> {
        self.thread_order
            .iter()
            .enumerate()
            .filter_map(|(index, gid)| {
                self.location_for_gid(*gid)
                    .map(|location| (index as u64 + 1, location))
            })
            .collect()
    }

    fn location_for_gid(&self, gid: [u32; 3]) -> Option<SourceLocation> {
        if matches!(self.thread_status.get(&gid), Some(ThreadStatus::Finished)) {
            return None;
        }

        let evaluator = self.evaluators.get(&gid)?;
        let function_name = evaluator.current_function().ok()?.name.clone();
        let (block, idx) = evaluator.current_active_block().ok()?;
        let (current_statement, span) = block.span_iter().nth(idx)?;

        let (line, column) = if matches!(current_statement, Statement::Return { .. }) {
            // For return statements, point to the line after the closing brace
            // of the function body rather than the span of the return itself.
            let func = evaluator.current_function().ok()?;
            let total_span = naga::Span::total_span(func.body.span_iter().map(|(_, s)| *s));
            let total_range = total_span.to_range()?;
            let prefix = &self.source[..total_range.end];
            let line_number = prefix.matches('\n').count() as u32 + 2;
            (line_number, 0)
        } else {
            let loc = span.location(&self.source);
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
        let mut frames = Vec::new();
        let mut is_innermost = true;

        let evaluator = self.evaluator();
        for stack_frame in evaluator.stack.iter().rev() {
            if let StackFrame::Function(func_frame) = stack_frame {
                let func = evaluator.resolve_function(&func_frame.function_ref);
                let location = if is_innermost {
                    self.current_location()
                } else {
                    None
                };
                frames.push(StackFrameInfo {
                    name: func.name.clone(),
                    location,
                });
                is_innermost = false;
            }
        }

        frames
    }

    /// All local variables and `let` bindings visible at the current execution point.
    pub fn local_variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();

        if let (Ok(in_scope), Ok(func), Ok(frame), Ok(func_idx)) = (
            self.evaluator().local_variables_in_current_scope(),
            self.evaluator().current_function(),
            self.evaluator().current_function_frame(),
            self.evaluator().current_function_frame_index(),
        ) {
            for (handle, local) in func.local_variables.iter() {
                if in_scope.contains(&handle) {
                    let value = frame
                        .local_variables
                        .get(&handle)
                        .cloned()
                        .unwrap_or_else(|| {
                            self.evaluator().evaluate_local_variable(handle, func_idx)
                        });
                    vars.push(Variable {
                        name: local.name.clone(),
                        value,
                    });
                }
            }
        }

        if let Ok(named) = self.evaluator().named_expression_values() {
            for (name, value) in named {
                vars.push(Variable {
                    name: Some(name),
                    value,
                });
            }
        }

        vars
    }

    /// Current function arguments with their names and values.
    pub fn argument_variables(&self) -> Vec<Variable> {
        self.evaluator()
            .current_function_argument_values()
            .unwrap_or_default()
            .into_iter()
            .map(|(name, value)| Variable { name, value })
            .collect()
    }

    /// All global variables with their names and values.
    pub fn global_variables(&self) -> Vec<Variable> {
        self.evaluator()
            .global_variable_values()
            .into_iter()
            .map(|(name, value)| Variable { name, value })
            .collect()
    }

    pub fn entry_point_output(&self) -> Option<Value> {
        self.evaluator().entry_point_output.clone()
    }
}
