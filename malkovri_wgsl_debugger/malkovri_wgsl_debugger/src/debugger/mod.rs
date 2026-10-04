mod collectives;
mod config;
mod group;
mod inspect;
mod outputs;
mod quad;

mod run_control;
mod scheduler;
mod sync;

use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

pub use config::{DrawConfig, ExecutionConfig, VertexAttribute, VertexConfig, VertexStepMode};
use group::{ExecutionGroup, Invocation, InvocationId};
pub use outputs::ShaderOutput;
pub use run_control::RunResult;

use naga::{
    AddressSpace, Barrier, CollectiveOperation, Expression, GatherMode, Handle, ResourceBinding,
    SubgroupOperation,
};

use crate::{
    error::EvaluatorError,
    invocation::inputs::{
        ComputeThreadInputs, FragmentThreadInputs, GlobalConstants, InvocationInputs,
        VertexThreadInputs,
    },
    invocation::{InvocationState, evaluate_global_expression},
    program::ShaderProgram,
    program::WgslToModuleError,
    value::Value,
};

/// Workgroup and subgroup configuration for a compute debug session.
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
    workgroup_size: [u32; 3],
    /// Which workgroup in the dispatch is being debugged: [x, y, z].
    #[serde(alias = "id")]
    workgroup_id: [u32; 3],
    /// Subgroup (warp) size. The final subgroup may be partial.
    subgroup_size: u32,
    /// Total number of workgroups in the dispatch: [x, y, z].
    #[serde(alias = "count")]
    num_workgroups: [u32; 3],
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
    pub fn new(
        workgroup_size: [u32; 3],
        workgroup_id: [u32; 3],
        subgroup_size: u32,
        num_workgroups: [u32; 3],
    ) -> Result<Self, String> {
        let config = Self {
            workgroup_size,
            workgroup_id,
            subgroup_size,
            num_workgroups,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }
    pub fn workgroup_id(&self) -> [u32; 3] {
        self.workgroup_id
    }
    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }
    pub fn num_workgroups(&self) -> [u32; 3] {
        self.num_workgroups
    }

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

/// Error returned by [`ShaderProgram::create_debugger`].
#[derive(Debug, thiserror::Error)]
pub enum DebuggerError {
    #[error("WGSL error: {0}")]
    Wgsl(#[from] WgslToModuleError),
    #[error("Execution error: {0}")]
    Evaluator(#[from] EvaluatorError),
    #[error("Invalid execution configuration: {0}")]
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

/// Execution state of one invocation, independent of whole-session completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
    Running,
    Waiting,
    Finished,
}

/// Identifies a call frame. Valid only until execution resumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DebugFrameId(usize);

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParkReason {
    Quad,
    Barrier(Barrier),
    WorkGroupUniformLoad {
        result: Handle<Expression>,
    },
    SubgroupBallot {
        result: Handle<Expression>,
    },
    SubgroupCollective {
        op: SubgroupOperation,
        collective_op: CollectiveOperation,
        result: Handle<Expression>,
    },
    SubgroupGather {
        mode: GatherMode,
        result: Handle<Expression>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParkScope {
    Quad(usize),
    Workgroup,
    Subgroup(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ThreadStatus {
    Running,
    Parked(ParkReason),
    Finished,
}

/// A named variable and its current value.
#[derive(Debug, Clone)]
pub struct Variable {
    pub name: Option<String>,
    pub value: Value,
}

/// DAP-facing thread id for one debuggable shader invocation.
pub type DebugThreadId = u64;

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
    pub id: DebugFrameId,
    pub name: Option<String>,
    pub location: Option<SourceLocation>,
}

/// One debuggable shader invocation exposed as a DAP thread.
#[derive(Debug, Clone)]
pub struct DebugThread {
    pub id: DebugThreadId,
    /// Compute coordinates, or [vertex_index, instance_index, 0] for vertex threads.
    pub global_invocation_id: [u32; 3],
    pub name: String,
}

/// A WGSL debugger session.
///
/// Create with [`ShaderProgram::create_debugger`], then call [`Debugger::step`] to advance
/// execution and the inspection methods to read program state.
pub struct Debugger {
    group: ExecutionGroup,
    focused_thread: InvocationId,
    program: Arc<ShaderProgram>,
    entry_point_index: usize,
}

impl Debugger {
    /// Start an independent execution using an already parsed immutable program.
    pub(crate) fn new(
        program: Arc<ShaderProgram>,
        entry_point_index: usize,
        config: ExecutionConfig,
        mut global_constants: GlobalConstants,
        bindings: HashMap<ResourceBinding, Value>,
    ) -> Result<Self, DebuggerError> {
        let module = program.module();
        if module.entry_points.get(entry_point_index).is_none() {
            return Err(DebuggerError::InvalidConfig(format!(
                "entry point index {entry_point_index} is invalid; shader has {} entry points",
                module.entry_points.len()
            )));
        }
        let naga_bindings: HashMap<naga::ResourceBinding, Rc<RefCell<Value>>> = bindings
            .into_iter()
            .map(|(binding, value)| (binding, Rc::new(RefCell::new(value))))
            .collect();

        let shared_workgroup_globals: HashMap<_, _> = module
            .global_variables
            .iter()
            .filter(|(_, global)| {
                global.binding.is_none() && global.space == AddressSpace::WorkGroup
            })
            .map(|(handle, global)| {
                let value = match global.init {
                    Some(expr) => evaluate_global_expression(module, expr),
                    None => Value::zero(module, global.ty),
                };
                (handle, Rc::new(RefCell::new(value)))
            })
            .collect();

        let stage = module.entry_points[entry_point_index].stage;
        let thread_inputs = match (stage, config) {
            (naga::ShaderStage::Compute, ExecutionConfig::Compute(config)) => {
                config.validate().map_err(DebuggerError::InvalidConfig)?;
                let [wx, wy, wz] = config.workgroup_size;
                global_constants.set_compute_configuration(
                    config.workgroup_size,
                    config.num_workgroups,
                    config.subgroup_size,
                );
                thread_order(&config)
                    .into_iter()
                    .map(|gid| {
                        (
                            gid,
                            InvocationInputs::Compute(ComputeThreadInputs::new(
                                [gid[0] % wx, gid[1] % wy, gid[2] % wz],
                                config.workgroup_size,
                                config.workgroup_id,
                                config.subgroup_size,
                            )),
                        )
                    })
                    .collect::<Vec<_>>()
            }
            (naga::ShaderStage::Vertex, ExecutionConfig::Vertex(config)) => {
                config
                    .validate(&program, entry_point_index)
                    .map_err(DebuggerError::InvalidConfig)?;
                let mut inputs = Vec::new();
                for instance in 0..config.draw.instance_count {
                    for vertex in 0..config.draw.vertex_count {
                        let vertex_index = config.draw.first_vertex + vertex;
                        let instance_index = config.draw.first_instance + instance;
                        inputs.push((
                            [vertex_index, instance_index, 0],
                            InvocationInputs::Vertex(VertexThreadInputs::new(
                                vertex_index,
                                instance_index,
                                config.locations(vertex, instance),
                            )),
                        ));
                    }
                }
                inputs
            }
            (naga::ShaderStage::Fragment, ExecutionConfig::FragmentQuads(quads)) => {
                if quads.len() > crate::graphics::MAX_FRAGMENT_INVOCATIONS / 4 {
                    return Err(DebuggerError::InvalidConfig(
                        "fragment invocation limit exceeded".into(),
                    ));
                }
                let fields = program
                    .input_locations(entry_point_index)
                    .map_err(DebuggerError::InvalidConfig)?;
                let mut inputs = Vec::with_capacity(quads.len() * 4);
                for (index, quad) in quads.iter().enumerate() {
                    if quad.selected == 0
                        || quad.selected & !quad.coverage != 0
                        || (quad.coverage | quad.selected) & !15 != 0
                        || quad.origin.iter().any(|v| v % 2 != 0 || *v >= 1 << 23)
                    {
                        return Err(DebuggerError::InvalidConfig(
                            "invalid fragment quad masks or origin".into(),
                        ));
                    }
                    for lane in 0..4 {
                        for field in &fields {
                            let value = quad.inputs[lane]
                                .locations
                                .get(&field.location)
                                .ok_or_else(|| {
                                    DebuggerError::InvalidConfig(format!(
                                        "missing @location({})",
                                        field.location
                                    ))
                                })?;
                            field
                                .ty
                                .validate(value)
                                .map_err(DebuggerError::InvalidConfig)?;
                        }
                        let input = FragmentThreadInputs::from_quad(quad, index, lane);
                        let info = input.info.as_ref().unwrap();
                        inputs.push((
                            [info.pixel[0], info.pixel[1], quad.primitive_index],
                            InvocationInputs::Fragment(input),
                        ));
                    }
                }
                inputs
            }
            (naga::ShaderStage::Fragment, ExecutionConfig::Fragment) => vec![(
                [0, 0, 0],
                InvocationInputs::Fragment(FragmentThreadInputs::default()),
            )],
            _ => {
                return Err(DebuggerError::InvalidConfig(format!(
                    "configuration does not match {stage:?} shader stage; use WorkgroupConfig for compute, DrawConfig for vertex, or ExecutionConfig::Fragment for fragment"
                )));
            }
        };
        let mut invocations = Vec::with_capacity(thread_inputs.len());
        for (gid, inputs) in thread_inputs {
            let invocation = InvocationState::new(
                program.clone(),
                entry_point_index,
                global_constants,
                naga_bindings.clone(),
                shared_workgroup_globals.clone(),
                inputs,
            )?;
            invocations.push(Invocation::new(gid, invocation));
        }

        let group = ExecutionGroup::new(invocations);
        let focused_thread = group.ids().next().unwrap_or(InvocationId::placeholder());
        Ok(Self {
            group,
            focused_thread,
            program,
            entry_point_index,
        })
    }

    /// The WGSL source code for this session.
    pub fn source(&self) -> &str {
        self.program.source()
    }

    fn invocation(&self) -> &InvocationState {
        self.group.get(self.focused_thread).state()
    }

    fn invocation_mut(&mut self) -> &mut InvocationState {
        self.group.get_mut(self.focused_thread).state_mut()
    }

    fn invocation_for_thread(
        &self,
        thread_id: DebugThreadId,
    ) -> Result<&InvocationState, EvaluatorError> {
        Ok(self.group.get(self.group.resolve(thread_id)?).state())
    }

    pub fn thread_state(&self, thread_id: DebugThreadId) -> Result<ThreadState, EvaluatorError> {
        Ok(
            match self.group.get(self.group.resolve(thread_id)?).status() {
                ThreadStatus::Running => ThreadState::Running,
                ThreadStatus::Parked(_) => ThreadState::Waiting,
                ThreadStatus::Finished => ThreadState::Finished,
            },
        )
    }

    pub fn is_empty(&self) -> bool {
        self.group.ids().next().is_none()
    }

    pub fn thread_fragment_info(
        &self,
        thread: DebugThreadId,
    ) -> Result<Option<crate::graphics::FragmentInfo>, EvaluatorError> {
        Ok(self.invocation_for_thread(thread)?.fragment_info())
    }

    pub fn threads(&self) -> Vec<DebugThread> {
        self.group
            .ids()
            .map(|id| {
                let global_id = self.group.get(id).global_id();
                DebugThread {
                    id: id.thread_id(),
                    global_invocation_id: global_id,
                    name: match self.program.module().entry_points[self.entry_point_index].stage {
                        naga::ShaderStage::Vertex => {
                            format!("vertex {}, instance {}", global_id[0], global_id[1])
                        }
                        naga::ShaderStage::Fragment => {
                            self.group.get(id).state().fragment_info().map_or_else(
                                || "fragment 0".into(),
                                |info| {
                                    format!(
                                        "pixel [{}, {}], primitive {}, instance {}, lane {}{}",
                                        info.pixel[0],
                                        info.pixel[1],
                                        info.primitive_index,
                                        info.instance_index,
                                        info.lane,
                                        if info.discarded {
                                            " (discarded helper)"
                                        } else if info.helper {
                                            " (helper)"
                                        } else {
                                            ""
                                        }
                                    )
                                },
                            )
                        }
                        _ => format!("[{}, {}, {}]", global_id[0], global_id[1], global_id[2]),
                    },
                }
            })
            .collect()
    }

    pub fn focus_thread(&mut self, thread_id: DebugThreadId) -> Result<(), EvaluatorError> {
        self.focused_thread = self.group.resolve(thread_id)?;
        Ok(())
    }

    pub fn focused_thread_id(&self) -> DebugThreadId {
        self.focused_thread.thread_id()
    }
}
