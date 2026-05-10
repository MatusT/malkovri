use std::sync::Arc;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use naga::Statement;

use crate::{
    entry_point_inputs::GlobalConstants,
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
    #[serde(alias = "size")]
    pub workgroup_size: [u32; 3],
    /// Which workgroup in the dispatch is being debugged: [x, y, z].
    #[serde(alias = "id")]
    pub workgroup_id: [u32; 3],
    /// Subgroup (warp) size. The final subgroup may be partial.
    pub subgroup_size: u32,
    /// Total number of workgroups in the dispatch: [x, y, z].
    #[serde(alias = "count")]
    pub num_workgroups: [u32; 3],
}

impl Default for WorkgroupConfig {
    fn default() -> Self {
        Self {
            workgroup_size: [1, 1, 1],
            workgroup_id: [0, 0, 0],
            subgroup_size: 4,
            num_workgroups: [1, 1, 1],
        }
    }
}

impl WorkgroupConfig {
    /// Validate the configuration against WGSL spec constraints:
    ///
    /// - `subgroup_size` must be a power of 2 in the range [4, 128].
    /// - `workgroup_size` must have at least one thread (no zero dimension).
    pub fn validate(&self) -> Result<(), String> {
        let [wx, wy, wz] = self.workgroup_size;
        if wx == 0 || wy == 0 || wz == 0 {
            return Err(format!(
                "workgroup_size {:?} must not have a zero dimension",
                self.workgroup_size
            ));
        }

        let s = self.subgroup_size;
        if !(4..=128).contains(&s) {
            return Err(format!(
                "subgroup_size {s} is outside the WGSL-specified range [4, 128]"
            ));
        }
        if !s.is_power_of_two() {
            return Err(format!(
                "subgroup_size {s} must be a power of 2 (WGSL spec §\"Subgroup Operations\")"
            ));
        }

        Ok(())
    }
}

fn thread_order(config: &WorkgroupConfig) -> Vec<[u32; 3]> {
    let mut threads = Vec::new();
    let [wx, wy, wz] = config.workgroup_size;
    for z in 0..wz {
        for y in 0..wy {
            for x in 0..wx {
                threads.push([
                    config.workgroup_id[0] * wx + x,
                    config.workgroup_id[1] * wy + y,
                    config.workgroup_id[2] * wz + z,
                ]);
            }
        }
    }
    threads
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

/// One debuggable shader invocation exposed as a DAP thread.
#[derive(Debug, Clone)]
pub struct DebugThread {
    pub id: u64,
    pub global_invocation_id: [u32; 3],
    pub name: String,
}

/// A WGSL debugger session.
///
/// Create with [`Debugger::new`], then call [`Debugger::step`] to advance
/// execution and the inspection methods to read program state.
pub struct Debugger {
    evaluators: HashMap<[u32; 3], Evaluator>,
    thread_order: Vec<[u32; 3]>,
    thread_ids: HashMap<u64, [u32; 3]>,
    focused_thread: [u32; 3],
    source: String,
}

impl Debugger {
    /// Create a new debugger session by parsing `source` and initializing the
    /// entry-point function at `entry_point_index`.
    ///
    /// `global_constants` provides user-set shader constants (vertex, fragment,
    /// etc.).  The compute-related fields (`workgroup_size`, `num_workgroups`,
    /// `subgroup_size`, `num_subgroups`) are overwritten from `config`.
    pub fn new(
        source: &str,
        entry_point_index: usize,
        config: WorkgroupConfig,
        mut global_constants: GlobalConstants,
        bindings: HashMap<ResourceBinding, Value>,
    ) -> Result<Self, DebuggerError> {
        config.validate().map_err(DebuggerError::InvalidConfig)?;

        let module = Arc::new(wgsl_to_module(source)?);
        let naga_bindings: HashMap<naga::ResourceBinding, Value> = bindings
            .into_iter()
            .map(|(rb, v)| {
                (
                    naga::ResourceBinding {
                        group: rb.group,
                        binding: rb.binding,
                    },
                    Value::Pointer(Rc::new(RefCell::new(v))),
                )
            })
            .collect();

        // Compute-related constants are derived from the workgroup config.
        let [wx, wy, wz] = config.workgroup_size;
        let total_threads = wx * wy * wz;
        global_constants.workgroup_size = config.workgroup_size;
        global_constants.num_workgroups = config.num_workgroups;
        global_constants.subgroup_size = config.subgroup_size;
        global_constants.num_subgroups = total_threads.div_ceil(config.subgroup_size);

        let thread_order = thread_order(&config);
        let focused_thread = thread_order.first().copied().ok_or_else(|| {
            DebuggerError::InvalidConfig("workgroup must contain at least one thread".into())
        })?;
        let thread_ids = thread_order
            .iter()
            .enumerate()
            .map(|(index, gid)| (index as u64 + 1, *gid))
            .collect();

        let mut evaluators = HashMap::new();
        for gid in &thread_order {
            let mut evaluator = Evaluator::new(
                module.clone(),
                entry_point_index,
                global_constants,
                naga_bindings.clone(),
                config.clone(),
            )?;
            evaluator.set_active_thread_gid(*gid)?;
            evaluators.insert(*gid, evaluator);
        }

        Ok(Self {
            evaluators,
            thread_order,
            thread_ids,
            focused_thread,
            source: source.to_string(),
        })
    }

    /// The WGSL source code for this session.
    pub fn source(&self) -> &str {
        &self.source
    }

    fn evaluator(&self) -> &Evaluator {
        &self.evaluators[&self.focused_thread]
    }

    fn evaluator_mut(&mut self) -> &mut Evaluator {
        self.evaluators
            .get_mut(&self.focused_thread)
            .expect("focused thread must have an evaluator")
    }

    pub fn threads(&self) -> Vec<DebugThread> {
        self.thread_order
            .iter()
            .enumerate()
            .map(|(index, gid)| DebugThread {
                id: index as u64 + 1,
                global_invocation_id: *gid,
                name: format!("[{}, {}, {}]", gid[0], gid[1], gid[2]),
            })
            .collect()
    }

    pub fn focus_thread(&mut self, thread_id: u64) -> Result<(), EvaluatorError> {
        let gid = self.thread_ids.get(&thread_id).copied().ok_or_else(|| {
            EvaluatorError::InternalError(format!("unknown DAP thread id {thread_id}"))
        })?;
        self.focused_thread = gid;
        Ok(())
    }

    pub fn focused_thread_id(&self) -> u64 {
        self.thread_order
            .iter()
            .position(|gid| *gid == self.focused_thread)
            .map(|index| index as u64 + 1)
            .unwrap_or(1)
    }

    /// Execute one user-visible statement, skipping internal `Emit`/`let` declarations.
    ///
    /// Returns [`StepResult::Finished`] when execution is complete.
    pub fn step(&mut self) -> Result<StepResult, EvaluatorError> {
        self.step_focused()
    }

    pub fn step_thread(&mut self, thread_id: u64) -> Result<StepResult, EvaluatorError> {
        self.focus_thread(thread_id)?;
        self.step_focused()
    }

    pub fn step_all(&mut self) -> Result<StepResult, EvaluatorError> {
        let mut any_continue = false;
        let focused_thread = self.focused_thread;
        let thread_order = self.thread_order.clone();
        for gid in thread_order {
            self.focused_thread = gid;
            if matches!(self.step_focused()?, StepResult::Continue) {
                any_continue = true;
            }
        }
        self.focused_thread = focused_thread;
        Ok(if any_continue {
            StepResult::Continue
        } else {
            StepResult::Finished
        })
    }

    fn step_focused(&mut self) -> Result<StepResult, EvaluatorError> {
        loop {
            match self.evaluator_mut().step()? {
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
        let evaluator = self.evaluator();
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
                        })
                        .leaf_value();
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
                    value: value.leaf_value(),
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
            .map(|(name, value)| Variable {
                name,
                value: value.leaf_value(),
            })
            .collect()
    }

    /// All global variables with their names and values.
    pub fn global_variables(&self) -> Vec<Variable> {
        self.evaluator()
            .global_variable_values()
            .into_iter()
            .map(|(name, value)| Variable {
                name,
                value: value.leaf_value(),
            })
            .collect()
    }

    pub fn entry_point_output(&self) -> Option<Value> {
        self.evaluator().entry_point_output.clone()
    }
}
