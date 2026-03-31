use std::collections::HashMap;
use std::sync::Arc;

use naga::Statement;

use crate::{
    entry_point_inputs::EntryPointInputs,
    error::EvaluatorError,
    evaluator::Evaluator,
    function_state::StackFrame,
    thread::ThreadIdentification,
    value::Value,
    wgsl::{WgslToModuleError, wgsl_to_module},
};

/// A resource binding identifier (group and binding index).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceBinding {
    pub group: u32,
    pub binding: u32,
}

/// Workgroup and subgroup configuration for a debug session.
///
/// Describes the size and position of the workgroup being debugged, and the
/// subgroup size used for subgroup operations.  All thread IDs
/// (`local_invocation_id`, `global_invocation_id`, `subgroup_id`, …) are
/// derived from these values automatically.
///
/// For a single-invocation session use the [`Default`] implementation, which
/// gives a 1×1×1 workgroup at position [0,0,0] with subgroup size 4.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct WorkgroupConfig {
    /// Number of threads along each dimension: [x, y, z].
    pub size: [u32; 3],
    /// Which workgroup in the dispatch is being debugged: [x, y, z].
    pub id: [u32; 3],
    /// Subgroup (warp) size. Must divide the total thread count evenly.
    pub subgroup_size: u32,
    /// Total number of workgroups in the dispatch: [x, y, z].
    pub count: [u32; 3],
}

impl Default for WorkgroupConfig {
    fn default() -> Self {
        Self {
            size: [1, 1, 1],
            id: [0, 0, 0],
            subgroup_size: 4,
            count: [1, 1, 1],
        }
    }
}

impl WorkgroupConfig {
    /// Validate the configuration against WGSL spec constraints:
    ///
    /// - `subgroup_size` must be a power of 2 in the range [4, 128].
    /// - `subgroup_size` must not exceed the total number of threads in the workgroup.
    /// - `workgroup_size` must have at least one thread (no zero dimension).
    pub fn validate(&self) -> Result<(), String> {
        let [wx, wy, wz] = self.size;
        if wx == 0 || wy == 0 || wz == 0 {
            return Err(format!(
                "workgroup_size {:?} must not have a zero dimension",
                self.size
            ));
        }

        let s = self.subgroup_size;
        if s < 4 || s > 128 {
            return Err(format!(
                "subgroup_size {s} is outside the WGSL-specified range [4, 128]"
            ));
        }
        if !s.is_power_of_two() {
            return Err(format!(
                "subgroup_size {s} must be a power of 2 (WGSL spec §\"Subgroup Operations\")"
            ));
        }

        let total = wx * wy * wz;
        if s > total {
            return Err(format!(
                "subgroup_size {s} exceeds total thread count {total} in workgroup {:?}",
                self.size
            ));
        }

        Ok(())
    }
}

/// Error returned by [`Debugger::new`].
#[derive(Debug, thiserror::Error)]
pub enum DebuggerError {
    #[error("WGSL error: {0}")]
    Wgsl(#[from] WgslToModuleError),
    #[error("Evaluator error: {0}")]
    Evaluator(#[from] EvaluatorError),
    #[error("Invalid WorkgroupConfig: {0}")]
    InvalidConfig(String),
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
        config: WorkgroupConfig,
        bindings: HashMap<ResourceBinding, Value>,
    ) -> Result<Self, DebuggerError> {
        config.validate().map_err(DebuggerError::InvalidConfig)?;

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

        // Derive all thread IDs for the first thread [0,0,0] from the config.
        // This will be replaced in Stage 2 when Evaluator manages all threads.
        let thread0 = ThreadIdentification::new(
            [0, 0, 0],
            config.size,
            config.id,
            config.subgroup_size,
        );
        let [wx, wy, wz] = config.size;
        let total_threads = wx * wy * wz;
        let inputs = EntryPointInputs {
            global_invocation_id: thread0.global_invocation_id(),
            local_invocation_id: thread0.local_invocation_id(),
            local_invocation_index: thread0.local_invocation_index(),
            workgroup_id: thread0.workgroup_id(),
            workgroup_size: config.size,
            num_workgroups: config.count,
            subgroup_id: thread0.subgroup_id(),
            subgroup_size: config.subgroup_size,
            subgroup_invocation_id: thread0.subgroup_invocation_id(),
            num_subgroups: total_threads / config.subgroup_size,
            ..EntryPointInputs::default()
        };

        let evaluator = Evaluator::new(module, entry_point_index, inputs, naga_bindings, config)?;

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
