use std::collections::HashMap;

use naga::{GlobalVariable, Handle};

use crate::{
    entry_point_inputs::{ComputeThreadInputs, FragmentThreadInputs, VertexThreadInputs},
    function_state::StackFrame,
    value::Value,
};

/// The kind of barrier or synchronization primitive a thread is waiting at.
#[derive(Clone, Debug)]
pub(crate) enum BarrierKind {
    WorkGroup,
    Storage,
    SubGroup,
    WorkGroupUniformLoad,
    SubgroupOp,
}

/// Execution status of a single thread.
#[derive(Clone, Debug)]
pub(crate) enum ThreadStatus {
    Running,
    AtBarrier(BarrierKind),
    Finished,
}

/// All per-invocation state for a single shader thread.
#[derive(Clone, Debug)]
pub(crate) struct EvaluatorThread {
    pub(crate) compute_inputs: ComputeThreadInputs,
    pub(crate) vertex_inputs: VertexThreadInputs,
    pub(crate) fragment_inputs: FragmentThreadInputs,
    pub(crate) stack: Vec<StackFrame>,
    /// `var<private>` globals — one copy per thread, not shared.
    pub(crate) private_globals: HashMap<Handle<GlobalVariable>, Value>,
    pub(crate) status: ThreadStatus,
}
