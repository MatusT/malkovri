use std::collections::HashMap;
use std::sync::Arc;

use naga::Statement;

use crate::{
    entry_point_inputs::EntryPointInputs,
    error::EvaluatorError,
    evaluator::Evaluator,
    function_state::StackFrame,
    value::Value,
    wgsl::{WgslToModuleError, wgsl_to_module},
};

/// A resource binding identifier (group and binding index).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceBinding {
    pub group: u32,
    pub binding: u32,
}

/// Error returned by [`Debugger::new`].
#[derive(Debug, thiserror::Error)]
pub enum DebuggerError {
    #[error("WGSL error: {0}")]
    Wgsl(#[from] WgslToModuleError),
    #[error("Evaluator error: {0}")]
    Evaluator(#[from] EvaluatorError),
}

/// Result of a single [`Debugger::step`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepResult {
    /// Execution is still in progress; more statements remain.
    Continue,
    /// Execution has finished.
    Finished,
}

/// A named variable and its current value.
#[derive(Debug, Clone)]
pub struct Variable {
    pub name: Option<String>,
    pub value: Value,
}

/// Source location of the current execution point.
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub line: u32,
    pub column: u32,
    pub function_name: Option<String>,
}

/// Information about a single call stack frame.
#[derive(Debug, Clone)]
pub struct StackFrameInfo {
    pub name: Option<String>,
    pub location: Option<SourceLocation>,
}

/// A WGSL debugger session.
///
/// Create with [`Debugger::new`], then call [`Debugger::step`] to advance
/// execution and the inspection methods to read program state.
pub struct Debugger {
    evaluator: Evaluator,
    source: String,
}

impl Debugger {
    /// Create a new debugger session by parsing `source` and initializing the
    /// entry-point function at `entry_point_index`.
    pub fn new(
        source: &str,
        entry_point_index: usize,
        inputs: EntryPointInputs,
        bindings: HashMap<ResourceBinding, Value>,
    ) -> Result<Self, DebuggerError> {
        let module = Arc::new(wgsl_to_module(source)?);
        let naga_bindings = bindings
            .into_iter()
            .map(|(rb, v)| {
                (
                    naga::ResourceBinding {
                        group: rb.group,
                        binding: rb.binding,
                    },
                    v,
                )
            })
            .collect();
        let evaluator = Evaluator::new(module, entry_point_index, inputs, naga_bindings)?;
        Ok(Self {
            evaluator,
            source: source.to_string(),
        })
    }

    /// The WGSL source code for this session.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Execute one user-visible statement, skipping internal `Emit`/`let` declarations.
    ///
    /// Returns [`StepResult::Finished`] when execution is complete.
    pub fn step(&mut self) -> Result<StepResult, EvaluatorError> {
        loop {
            match self.evaluator.step()? {
                None => return Ok(StepResult::Finished),
                Some(next) => {
                    if !matches!(next.statement, Statement::Emit(_)) {
                        return Ok(StepResult::Continue);
                    }
                }
            }
        }
    }

    /// Source location of the current execution point.
    ///
    /// Returns `None` if execution has finished (stack is empty).
    pub fn current_location(&self) -> Option<SourceLocation> {
        let function_name = self.evaluator.current_function().ok()?.name.clone();
        let (block, idx) = self.evaluator.current_active_block().ok()?;
        let (current_statement, span) = block.span_iter().nth(idx)?;

        let (line, column) = if matches!(current_statement, Statement::Return { .. }) {
            // For return statements, point to the line after the closing brace
            // of the function body rather than the span of the return itself.
            let func = self.evaluator.current_function().ok()?;
            let total_span =
                naga::Span::total_span(func.body.span_iter().map(|(_, s)| *s));
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

        for stack_frame in self.evaluator.stack.iter().rev() {
            if let StackFrame::Function(func_frame) = stack_frame {
                let func = self.evaluator.resolve_function(&func_frame.function_ref);
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

        if let (Ok(in_scope), Ok(func), Ok(frame)) = (
            self.evaluator.local_variables_in_current_scope(),
            self.evaluator.current_function(),
            self.evaluator.current_function_frame(),
        ) {
            for (handle, local) in func.local_variables.iter() {
                if in_scope.contains(&handle) {
                    let value = frame
                        .local_variables
                        .get(&handle)
                        .cloned()
                        .unwrap_or(Value::Uninitialized)
                        .leaf_value();
                    vars.push(Variable {
                        name: local.name.clone(),
                        value,
                    });
                }
            }
        }

        if let Ok(named) = self.evaluator.named_expression_values() {
            for (name, value) in named {
                vars.push(Variable {
                    name: Some(name),
                    value: value.leaf_value(),
                });
            }
        }

        vars
    }

    /// Current function arguments with their names and values.
    pub fn argument_variables(&self) -> Vec<Variable> {
        self.evaluator
            .current_function_argument_values()
            .unwrap_or_default()
            .into_iter()
            .map(|(name, value)| Variable {
                name,
                value: value.leaf_value(),
            })
            .collect()
    }

    /// All global variables with their names and values.
    pub fn global_variables(&self) -> Vec<Variable> {
        self.evaluator
            .global_variable_values()
            .into_iter()
            .map(|(name, value)| Variable {
                name,
                value: value.leaf_value(),
            })
            .collect()
    }
}
