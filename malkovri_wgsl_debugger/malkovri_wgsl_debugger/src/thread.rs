use std::collections::HashMap;

use naga::{GlobalVariable, Handle};

use crate::{
    function_state::StackFrame,
    value::Value,
};

/// All WGSL builtin IDs that uniquely identify a thread within a dispatch.
/// Constructed from the minimal inputs; all other fields are derived.
/// Fields are private — use the accessor methods.
#[derive(Clone, Debug)]
pub(crate) struct ThreadIdentification {
    local_invocation_id: [u32; 3],
    local_invocation_index: u32,
    global_invocation_id: [u32; 3],
    workgroup_id: [u32; 3],
    subgroup_id: u32,
    subgroup_invocation_id: u32,
}

impl ThreadIdentification {
    pub(crate) fn new(
        local_invocation_id: [u32; 3],
        workgroup_size: [u32; 3],
        workgroup_id: [u32; 3],
        subgroup_size: u32,
    ) -> Self {
        let [wx, wy, wz] = workgroup_size;
        let [x, y, z] = local_invocation_id;
        let local_invocation_index = x + y * wx + z * wx * wy;
        Self {
            local_invocation_id,
            local_invocation_index,
            global_invocation_id: [
                workgroup_id[0] * wx + x,
                workgroup_id[1] * wy + y,
                workgroup_id[2] * wz + z,
            ],
            workgroup_id,
            subgroup_id: local_invocation_index / subgroup_size,
            subgroup_invocation_id: local_invocation_index % subgroup_size,
        }
    }

    pub(crate) fn local_invocation_id(&self) -> [u32; 3] {
        self.local_invocation_id
    }

    pub(crate) fn local_invocation_index(&self) -> u32 {
        self.local_invocation_index
    }

    pub(crate) fn global_invocation_id(&self) -> [u32; 3] {
        self.global_invocation_id
    }

    pub(crate) fn workgroup_id(&self) -> [u32; 3] {
        self.workgroup_id
    }

    pub(crate) fn subgroup_id(&self) -> u32 {
        self.subgroup_id
    }

    pub(crate) fn subgroup_invocation_id(&self) -> u32 {
        self.subgroup_invocation_id
    }
}

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
    pub(crate) id: ThreadIdentification,
    pub(crate) stack: Vec<StackFrame>,
    /// `var<private>` globals — one copy per thread, not shared.
    pub(crate) private_globals: HashMap<Handle<GlobalVariable>, Value>,
    pub(crate) status: ThreadStatus,
}
