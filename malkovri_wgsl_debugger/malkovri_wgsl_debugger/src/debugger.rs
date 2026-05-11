use std::sync::Arc;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use naga::{
    AddressSpace, Barrier, CollectiveOperation, Expression, GatherMode, Handle, Statement,
    SubgroupOperation,
};

use crate::{
    entry_point_inputs::GlobalConstants,
    error::EvaluatorError,
    eval_expressions::evaluate_global_expression,
    evaluator::Evaluator,
    function_state::StackFrame,
    primitive::Primitive,
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParkReason {
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
    thread_status: HashMap<[u32; 3], ThreadStatus>,
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
        let naga_bindings: HashMap<naga::ResourceBinding, Rc<RefCell<Value>>> = bindings
            .into_iter()
            .map(|(rb, v)| {
                (
                    naga::ResourceBinding {
                        group: rb.group,
                        binding: rb.binding,
                    },
                    Rc::new(RefCell::new(v)),
                )
            })
            .collect();

        let shared_workgroup_globals: HashMap<_, _> = module
            .global_variables
            .iter()
            .filter_map(|(handle, global)| {
                (global.binding.is_none() && global.space == AddressSpace::WorkGroup).then(|| {
                    let value = match global.init {
                        Some(expr) => evaluate_global_expression(&module, expr),
                        None => Value::zero(&module, global.ty),
                    };
                    (handle, Rc::new(RefCell::new(value)))
                })
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
                shared_workgroup_globals.clone(),
                config.clone(),
            )?;
            evaluator.set_active_thread_gid(*gid)?;
            evaluators.insert(*gid, evaluator);
        }

        let thread_status = thread_order
            .iter()
            .map(|gid| (*gid, ThreadStatus::Running))
            .collect();

        Ok(Self {
            evaluators,
            thread_status,
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
        self.step_gid(self.focused_thread)
    }

    pub fn step_thread(&mut self, thread_id: u64) -> Result<StepResult, EvaluatorError> {
        self.focus_thread(thread_id)?;
        self.step_gid(self.focused_thread)
    }

    pub fn step_all(&mut self) -> Result<StepResult, EvaluatorError> {
        let focused_thread = self.focused_thread;
        let thread_order = self.thread_order.clone();
        for gid in thread_order {
            self.focused_thread = gid;
            self.step_gid(gid)?;
        }
        self.focused_thread = focused_thread;
        self.release_ready_parked_threads()?;
        self.detect_deadlock()?;
        Ok(self.session_step_result())
    }

    fn step_gid(&mut self, gid: [u32; 3]) -> Result<StepResult, EvaluatorError> {
        if matches!(self.thread_status.get(&gid), Some(ThreadStatus::Finished)) {
            return Ok(self.session_step_result());
        }
        if matches!(self.thread_status.get(&gid), Some(ThreadStatus::Parked(_))) {
            self.release_ready_parked_threads()?;
            self.detect_deadlock()?;
            return Ok(self.session_step_result());
        }

        self.focused_thread = gid;

        loop {
            let Some(next) = self.evaluator_mut().current_statement()? else {
                self.thread_status.insert(gid, ThreadStatus::Finished);
                self.release_ready_parked_threads()?;
                self.detect_deadlock()?;
                return Ok(self.session_step_result());
            };

            if let Some(reason) = Self::park_reason_for_statement(&next.statement) {
                self.thread_status.insert(gid, ThreadStatus::Parked(reason));
                self.release_ready_parked_threads()?;
                self.detect_deadlock()?;
                return Ok(self.session_step_result());
            }

            match self.evaluator_mut().step()? {
                None => {
                    self.thread_status.insert(gid, ThreadStatus::Finished);
                    self.release_ready_parked_threads()?;
                    self.detect_deadlock()?;
                    return Ok(self.session_step_result());
                }
                Some(next) if matches!(next.statement, Statement::Emit(_)) => continue,
                Some(_) => return Ok(StepResult::Continue),
            }
        }
    }

    fn session_step_result(&self) -> StepResult {
        if self
            .thread_order
            .iter()
            .all(|gid| matches!(self.thread_status.get(gid), Some(ThreadStatus::Finished)))
        {
            StepResult::Finished
        } else {
            StepResult::Continue
        }
    }

    fn park_reason_for_statement(statement: &Statement) -> Option<ParkReason> {
        match statement {
            Statement::ControlBarrier(barrier) | Statement::MemoryBarrier(barrier) => {
                Some(ParkReason::Barrier(*barrier))
            }
            Statement::WorkGroupUniformLoad { result, .. } => {
                Some(ParkReason::WorkGroupUniformLoad { result: *result })
            }
            Statement::SubgroupBallot { result, .. } => {
                Some(ParkReason::SubgroupBallot { result: *result })
            }
            Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                result,
                ..
            } => Some(ParkReason::SubgroupCollective {
                op: *op,
                collective_op: *collective_op,
                result: *result,
            }),
            Statement::SubgroupGather { mode, result, .. } => Some(ParkReason::SubgroupGather {
                mode: *mode,
                result: *result,
            }),
            _ => None,
        }
    }

    fn release_ready_parked_threads(&mut self) -> Result<(), EvaluatorError> {
        loop {
            let Some((reason, members)) = self.find_ready_parked_group()? else {
                return Ok(());
            };
            self.release_parked_group(reason, members)?;
        }
    }

    fn find_ready_parked_group(
        &self,
    ) -> Result<Option<(ParkReason, Vec<[u32; 3]>)>, EvaluatorError> {
        for gid in &self.thread_order {
            let Some(ThreadStatus::Parked(reason)) = self.thread_status.get(gid) else {
                continue;
            };
            let members = self.live_members_for_reason(*gid, reason);
            if members.is_empty() {
                continue;
            }

            let all_parked = members.iter().all(|member| {
                matches!(
                    self.thread_status.get(member),
                    Some(ThreadStatus::Parked(_))
                )
            });
            if !all_parked {
                continue;
            }

            let all_compatible = members.iter().all(|member| {
                matches!(
                    self.thread_status.get(member),
                    Some(ThreadStatus::Parked(member_reason)) if member_reason == reason
                )
            });

            if all_compatible {
                return Ok(Some((reason.clone(), members)));
            }

            return Err(self.synchronization_error("divergent synchronization point", &members));
        }

        Ok(None)
    }

    fn detect_deadlock(&self) -> Result<(), EvaluatorError> {
        let live: Vec<_> = self
            .thread_order
            .iter()
            .copied()
            .filter(|gid| !matches!(self.thread_status.get(gid), Some(ThreadStatus::Finished)))
            .collect();

        if live.is_empty() {
            return Ok(());
        }

        if live
            .iter()
            .all(|gid| matches!(self.thread_status.get(gid), Some(ThreadStatus::Parked(_))))
        {
            return Err(self.synchronization_error("deadlocked synchronization", &live));
        }

        Ok(())
    }

    fn synchronization_error(&self, label: &str, gids: &[[u32; 3]]) -> EvaluatorError {
        let threads = gids
            .iter()
            .map(|gid| {
                let thread_id = self.thread_id_for_gid(*gid);
                let status = self.thread_status.get(gid);
                format!("{thread_id}:{gid:?}={status:?}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        EvaluatorError::SynchronizationError(format!("{label}: {threads}"))
    }

    fn thread_id_for_gid(&self, gid: [u32; 3]) -> u64 {
        self.thread_order
            .iter()
            .position(|candidate| *candidate == gid)
            .map(|index| index as u64 + 1)
            .unwrap_or(1)
    }

    fn live_members_for_reason(&self, gid: [u32; 3], reason: &ParkReason) -> Vec<[u32; 3]> {
        self.members_for_scope(self.scope_for_reason(gid, reason))
            .into_iter()
            .filter(|member| {
                !matches!(self.thread_status.get(member), Some(ThreadStatus::Finished))
            })
            .collect()
    }

    fn scope_for_reason(&self, gid: [u32; 3], reason: &ParkReason) -> ParkScope {
        match reason {
            ParkReason::Barrier(barrier)
                if barrier.contains(Barrier::SUB_GROUP)
                    && !barrier
                        .intersects(Barrier::WORK_GROUP | Barrier::STORAGE | Barrier::TEXTURE) =>
            {
                ParkScope::Subgroup(self.subgroup_id(gid))
            }
            ParkReason::SubgroupBallot { .. }
            | ParkReason::SubgroupCollective { .. }
            | ParkReason::SubgroupGather { .. } => ParkScope::Subgroup(self.subgroup_id(gid)),
            ParkReason::Barrier(_) | ParkReason::WorkGroupUniformLoad { .. } => {
                ParkScope::Workgroup
            }
        }
    }

    fn members_for_scope(&self, scope: ParkScope) -> Vec<[u32; 3]> {
        match scope {
            ParkScope::Workgroup => self.thread_order.clone(),
            ParkScope::Subgroup(subgroup_id) => self
                .thread_order
                .iter()
                .copied()
                .filter(|gid| self.subgroup_id(*gid) == subgroup_id)
                .collect(),
        }
    }

    fn subgroup_id(&self, gid: [u32; 3]) -> u32 {
        self.evaluators[&gid]
            .active_thread()
            .compute_inputs
            .subgroup_id
    }

    fn subgroup_lane(&self, gid: [u32; 3]) -> u32 {
        self.evaluators[&gid]
            .active_thread()
            .compute_inputs
            .subgroup_invocation_id
    }

    fn release_parked_group(
        &mut self,
        reason: ParkReason,
        members: Vec<[u32; 3]>,
    ) -> Result<(), EvaluatorError> {
        match reason {
            ParkReason::Barrier(_) => self.release_barrier(members),
            ParkReason::WorkGroupUniformLoad { result } => {
                self.release_workgroup_uniform_load(members, result)
            }
            ParkReason::SubgroupBallot { result } => self.release_subgroup_ballot(members, result),
            ParkReason::SubgroupCollective {
                op,
                collective_op,
                result,
            } => self.release_subgroup_collective(members, op, collective_op, result),
            ParkReason::SubgroupGather { mode, result } => {
                self.release_subgroup_gather(members, mode, result)
            }
        }
    }

    fn release_barrier(&mut self, members: Vec<[u32; 3]>) -> Result<(), EvaluatorError> {
        for gid in members {
            let next = {
                let evaluator = self.evaluators.get_mut(&gid).ok_or_else(|| {
                    EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
                })?;
                evaluator.consume_current_statement_without_running()?
            };
            self.thread_status.insert(
                gid,
                if next.is_some() {
                    ThreadStatus::Running
                } else {
                    ThreadStatus::Finished
                },
            );
        }
        Ok(())
    }

    fn release_workgroup_uniform_load(
        &mut self,
        members: Vec<[u32; 3]>,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        let mut agreed_place = None;
        let mut loaded_value = None;

        for gid in &members {
            let evaluator = self.evaluators.get_mut(gid).ok_or_else(|| {
                EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
            })?;
            let Some(next) = evaluator.current_statement()? else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} finished while parked at workGroupUniformLoad",
                    self.thread_id_for_gid(*gid)
                )));
            };
            let Statement::WorkGroupUniformLoad {
                pointer,
                result: found,
            } = next.statement
            else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} is not parked at workGroupUniformLoad",
                    self.thread_id_for_gid(*gid)
                )));
            };
            if found != result {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} reached a different workGroupUniformLoad site",
                    self.thread_id_for_gid(*gid)
                )));
            }

            let place = evaluator.resolve_pointer_place(pointer)?;
            if let Some(expected) = &agreed_place {
                if expected != &place {
                    return Err(EvaluatorError::SynchronizationError(format!(
                        "workGroupUniformLoad pointer mismatch at thread {}",
                        self.thread_id_for_gid(*gid)
                    )));
                }
            } else {
                loaded_value = Some(evaluator.read_place(&place));
                agreed_place = Some(place);
            }
        }

        let value = loaded_value.unwrap_or(Value::Uninitialized);
        self.inject_statement_result_and_consume(members, result, value)
    }

    fn release_subgroup_ballot(
        &mut self,
        members: Vec<[u32; 3]>,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        let mut ballot = [0u32; 4];
        for gid in &members {
            let lane = self.subgroup_lane(*gid);
            let predicate = {
                let evaluator = self.evaluators.get_mut(gid).ok_or_else(|| {
                    EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
                })?;
                let Some(next) = evaluator.current_statement()? else {
                    return Err(EvaluatorError::SynchronizationError(format!(
                        "thread {} finished while parked at subgroupBallot",
                        self.thread_id_for_gid(*gid)
                    )));
                };
                let Statement::SubgroupBallot {
                    result: found,
                    predicate,
                } = next.statement
                else {
                    return Err(EvaluatorError::SynchronizationError(format!(
                        "thread {} is not parked at subgroupBallot",
                        self.thread_id_for_gid(*gid)
                    )));
                };
                if found != result {
                    return Err(EvaluatorError::SynchronizationError(format!(
                        "thread {} reached a different subgroupBallot site",
                        self.thread_id_for_gid(*gid)
                    )));
                }
                predicate
                    .map(|expr| evaluator.evaluate_expression(expr).is_truthy())
                    .unwrap_or(true)
            };
            if predicate {
                ballot[(lane / 32) as usize] |= 1u32 << (lane % 32);
            }
        }

        self.inject_statement_result_and_consume(
            members,
            result,
            Value::Primitive(Primitive::U32x4(ballot)),
        )
    }

    fn release_subgroup_collective(
        &mut self,
        mut members: Vec<[u32; 3]>,
        op: SubgroupOperation,
        collective_op: CollectiveOperation,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        members.sort_by_key(|gid| self.subgroup_lane(*gid));

        let mut lane_values = Vec::new();
        for gid in &members {
            let evaluator = self.evaluators.get_mut(gid).ok_or_else(|| {
                EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
            })?;
            let Some(next) = evaluator.current_statement()? else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} finished while parked at subgroup collective",
                    self.thread_id_for_gid(*gid)
                )));
            };
            let Statement::SubgroupCollectiveOperation {
                op: found_op,
                collective_op: found_collective_op,
                argument,
                result: found_result,
            } = next.statement
            else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} is not parked at subgroup collective",
                    self.thread_id_for_gid(*gid)
                )));
            };
            if found_op != op || found_collective_op != collective_op || found_result != result {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} reached a different subgroup collective site",
                    self.thread_id_for_gid(*gid)
                )));
            }
            lane_values.push((*gid, evaluator.evaluate_expression(argument)));
        }

        let results = match collective_op {
            CollectiveOperation::Reduce => {
                let value = Self::reduce_subgroup_values(
                    op,
                    lane_values.iter().map(|(_, value)| value.clone()).collect(),
                )?;
                lane_values
                    .iter()
                    .map(|(gid, _)| (*gid, value.clone()))
                    .collect()
            }
            CollectiveOperation::InclusiveScan => {
                Self::scan_subgroup_values(op, &lane_values, true)?
            }
            CollectiveOperation::ExclusiveScan => {
                Self::scan_subgroup_values(op, &lane_values, false)?
            }
        };

        self.inject_many_statement_results_and_consume(results, result)
    }

    fn release_subgroup_gather(
        &mut self,
        mut members: Vec<[u32; 3]>,
        mode: GatherMode,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        members.sort_by_key(|gid| self.subgroup_lane(*gid));

        let mut lane_values = HashMap::new();
        let mut target_lanes = HashMap::new();
        for gid in &members {
            let lane = self.subgroup_lane(*gid);
            let evaluator = self.evaluators.get_mut(gid).ok_or_else(|| {
                EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
            })?;
            let Some(next) = evaluator.current_statement()? else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} finished while parked at subgroup gather",
                    self.thread_id_for_gid(*gid)
                )));
            };
            let Statement::SubgroupGather {
                mode: found_mode,
                argument,
                result: found_result,
            } = next.statement
            else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} is not parked at subgroup gather",
                    self.thread_id_for_gid(*gid)
                )));
            };
            if found_mode != mode || found_result != result {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} reached a different subgroup gather site",
                    self.thread_id_for_gid(*gid)
                )));
            }
            lane_values.insert(lane, evaluator.evaluate_expression(argument));
            target_lanes.insert(lane, Self::gather_target_lane(evaluator, lane, mode));
        }

        let first_lane = lane_values.keys().min().copied().unwrap_or(0);
        let results = members
            .iter()
            .map(|gid| {
                let lane = self.subgroup_lane(*gid);
                let target = match mode {
                    GatherMode::BroadcastFirst => first_lane,
                    _ => target_lanes.get(&lane).copied().unwrap_or(lane),
                };
                let value = lane_values
                    .get(&target)
                    .cloned()
                    .unwrap_or(Value::Uninitialized);
                (*gid, value)
            })
            .collect();

        self.inject_many_statement_results_and_consume(results, result)
    }

    fn inject_statement_result_and_consume(
        &mut self,
        members: Vec<[u32; 3]>,
        result: Handle<Expression>,
        value: Value,
    ) -> Result<(), EvaluatorError> {
        let results = members
            .into_iter()
            .map(|gid| (gid, value.clone()))
            .collect();
        self.inject_many_statement_results_and_consume(results, result)
    }

    fn inject_many_statement_results_and_consume(
        &mut self,
        results: Vec<([u32; 3], Value)>,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        for (gid, value) in results {
            let next = {
                let evaluator = self.evaluators.get_mut(&gid).ok_or_else(|| {
                    EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
                })?;
                evaluator.set_current_expression_value(result, value)?;
                evaluator.consume_current_statement_without_running()?
            };
            self.thread_status.insert(
                gid,
                if next.is_some() {
                    ThreadStatus::Running
                } else {
                    ThreadStatus::Finished
                },
            );
        }
        Ok(())
    }

    fn gather_target_lane(evaluator: &Evaluator, lane: u32, mode: GatherMode) -> u32 {
        match mode {
            GatherMode::BroadcastFirst => lane,
            GatherMode::Broadcast(expr) | GatherMode::Shuffle(expr) => {
                Self::evaluate_u32(evaluator, expr).unwrap_or(lane)
            }
            GatherMode::ShuffleDown(expr) => {
                lane.saturating_add(Self::evaluate_u32(evaluator, expr).unwrap_or(0))
            }
            GatherMode::ShuffleUp(expr) => {
                lane.saturating_sub(Self::evaluate_u32(evaluator, expr).unwrap_or(0))
            }
            GatherMode::ShuffleXor(expr) => lane ^ Self::evaluate_u32(evaluator, expr).unwrap_or(0),
            GatherMode::QuadBroadcast(expr) => {
                let index = Self::evaluate_u32(evaluator, expr).unwrap_or(0) % 4;
                (lane / 4) * 4 + index
            }
            GatherMode::QuadSwap(direction) => {
                lane ^ match direction {
                    naga::Direction::X => 1,
                    naga::Direction::Y => 2,
                    naga::Direction::Diagonal => 3,
                }
            }
        }
    }

    fn evaluate_u32(evaluator: &Evaluator, expr: Handle<Expression>) -> Option<u32> {
        match evaluator.evaluate_expression(expr) {
            Value::Primitive(Primitive::U32(value)) => Some(value),
            Value::Primitive(Primitive::I32(value)) if value >= 0 => Some(value as u32),
            _ => None,
        }
    }

    fn reduce_subgroup_values(
        op: SubgroupOperation,
        values: Vec<Value>,
    ) -> Result<Value, EvaluatorError> {
        if values.is_empty() {
            return Ok(Value::Uninitialized);
        }

        match op {
            SubgroupOperation::All => Ok(Value::Primitive(Primitive::U32(u32::from(
                values.iter().all(Value::is_truthy),
            )))),
            SubgroupOperation::Any => Ok(Value::Primitive(Primitive::U32(u32::from(
                values.iter().any(Value::is_truthy),
            )))),
            SubgroupOperation::Add
            | SubgroupOperation::Mul
            | SubgroupOperation::Min
            | SubgroupOperation::Max
            | SubgroupOperation::And
            | SubgroupOperation::Or
            | SubgroupOperation::Xor => {
                let mut iter = values.into_iter();
                let first = iter.next().unwrap();
                iter.try_fold(first, |acc, value| {
                    Self::combine_subgroup_values(op, acc, value)
                })
            }
        }
    }

    fn scan_subgroup_values(
        op: SubgroupOperation,
        lane_values: &[([u32; 3], Value)],
        inclusive: bool,
    ) -> Result<Vec<([u32; 3], Value)>, EvaluatorError> {
        let Some((_, first_value)) = lane_values.first() else {
            return Ok(Vec::new());
        };
        let mut acc = Self::identity_subgroup_value(op, first_value)?;
        let mut results = Vec::new();
        for (gid, value) in lane_values {
            if inclusive {
                acc = Self::combine_subgroup_values(op, acc, value.clone())?;
                results.push((*gid, acc.clone()));
            } else {
                results.push((*gid, acc.clone()));
                acc = Self::combine_subgroup_values(op, acc, value.clone())?;
            }
        }
        Ok(results)
    }

    fn combine_subgroup_values(
        op: SubgroupOperation,
        left: Value,
        right: Value,
    ) -> Result<Value, EvaluatorError> {
        let (Value::Primitive(left), Value::Primitive(right)) = (left, right) else {
            return Ok(Value::Uninitialized);
        };

        let primitive = match op {
            SubgroupOperation::Add => left + right,
            SubgroupOperation::Mul => left * right,
            SubgroupOperation::Min => Self::primitive_min(left, right)?,
            SubgroupOperation::Max => Self::primitive_max(left, right)?,
            SubgroupOperation::And => left & right,
            SubgroupOperation::Or => left | right,
            SubgroupOperation::Xor => left ^ right,
            SubgroupOperation::All | SubgroupOperation::Any => {
                return Ok(Value::Primitive(Primitive::U32(u32::from(match op {
                    SubgroupOperation::All => {
                        Value::Primitive(left).is_truthy() && Value::Primitive(right).is_truthy()
                    }
                    SubgroupOperation::Any => {
                        Value::Primitive(left).is_truthy() || Value::Primitive(right).is_truthy()
                    }
                    _ => unreachable!(),
                }))));
            }
        };

        Ok(Value::Primitive(primitive))
    }

    fn identity_subgroup_value(
        op: SubgroupOperation,
        sample: &Value,
    ) -> Result<Value, EvaluatorError> {
        let Value::Primitive(sample) = sample else {
            return Ok(Value::Uninitialized);
        };
        let primitive = match op {
            SubgroupOperation::Add => {
                Self::map_primitive_components(*sample, |_| 0.0, |_| 0, |_| 0)
            }
            SubgroupOperation::Mul => {
                Self::map_primitive_components(*sample, |_| 1.0, |_| 1, |_| 1)
            }
            other => {
                return Err(EvaluatorError::UnsupportedStatement(format!(
                    "subgroup {:?} scan",
                    other
                )));
            }
        };
        Ok(Value::Primitive(primitive))
    }

    fn primitive_min(left: Primitive, right: Primitive) -> Result<Primitive, EvaluatorError> {
        left.zip_map_numeric(right, f32::min, i32::min, u32::min)
            .or_else(|| match (left, right) {
                (Primitive::F64(a), Primitive::F64(b)) => Some(Primitive::F64(a.min(b))),
                (Primitive::I64(a), Primitive::I64(b)) => Some(Primitive::I64(a.min(b))),
                (Primitive::U64(a), Primitive::U64(b)) => Some(Primitive::U64(a.min(b))),
                _ => None,
            })
            .ok_or_else(|| EvaluatorError::UnsupportedStatement("subgroup min type".to_string()))
    }

    fn primitive_max(left: Primitive, right: Primitive) -> Result<Primitive, EvaluatorError> {
        left.zip_map_numeric(right, f32::max, i32::max, u32::max)
            .or_else(|| match (left, right) {
                (Primitive::F64(a), Primitive::F64(b)) => Some(Primitive::F64(a.max(b))),
                (Primitive::I64(a), Primitive::I64(b)) => Some(Primitive::I64(a.max(b))),
                (Primitive::U64(a), Primitive::U64(b)) => Some(Primitive::U64(a.max(b))),
                _ => None,
            })
            .ok_or_else(|| EvaluatorError::UnsupportedStatement("subgroup max type".to_string()))
    }

    fn map_primitive_components(
        primitive: Primitive,
        ff32: impl Fn(f32) -> f32 + Copy,
        fi32: impl Fn(i32) -> i32 + Copy,
        fu32: impl Fn(u32) -> u32 + Copy,
    ) -> Primitive {
        primitive
            .map_numeric(ff32, fi32, fu32)
            .unwrap_or(match primitive {
                Primitive::F64(_) => Primitive::F64(ff32(0.0) as f64),
                Primitive::I64(_) => Primitive::I64(fi32(0) as i64),
                Primitive::U64(_) => Primitive::U64(fu32(0) as u64),
                other => other,
            })
    }

    /// Source location of the current execution point.
    ///
    /// Returns `None` if execution has finished (stack is empty).
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
