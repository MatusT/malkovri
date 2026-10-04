use crate::{error::EvaluatorError, invocation::InvocationState};

use super::{DebugThreadId, ThreadStatus};

/// Internal identity is independent of compute coordinates or graphics inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct InvocationId(usize);

impl InvocationId {
    pub(super) fn placeholder() -> Self {
        Self(0)
    }
    pub fn thread_id(self) -> DebugThreadId {
        self.0 as DebugThreadId + 1
    }
}

pub(super) struct Invocation {
    global_id: [u32; 3],
    state: InvocationState,
    status: ThreadStatus,
}

impl Invocation {
    pub fn new(global_id: [u32; 3], state: InvocationState) -> Self {
        Self {
            global_id,
            state,
            status: ThreadStatus::Running,
        }
    }

    pub fn global_id(&self) -> [u32; 3] {
        self.global_id
    }
    pub fn state(&self) -> &InvocationState {
        &self.state
    }
    pub fn state_mut(&mut self) -> &mut InvocationState {
        &mut self.state
    }
    pub fn status(&self) -> &ThreadStatus {
        &self.status
    }
    pub fn set_status(&mut self, status: ThreadStatus) {
        self.status = status;
    }
}

/// One active stage's invocations, in deterministic scheduler order.
pub(super) struct ExecutionGroup {
    invocations: Vec<Invocation>,
}

impl ExecutionGroup {
    pub fn new(invocations: Vec<Invocation>) -> Self {
        Self { invocations }
    }

    pub fn ids(&self) -> impl Iterator<Item = InvocationId> + '_ {
        (0..self.invocations.len()).map(InvocationId)
    }

    pub fn resolve(&self, thread_id: DebugThreadId) -> Result<InvocationId, EvaluatorError> {
        thread_id
            .checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|&index| index < self.invocations.len())
            .map(InvocationId)
            .ok_or_else(|| EvaluatorError::InternalError(format!("unknown thread id {thread_id}")))
    }

    pub fn get(&self, id: InvocationId) -> &Invocation {
        &self.invocations[id.0]
    }
    pub fn get_mut(&mut self, id: InvocationId) -> &mut Invocation {
        &mut self.invocations[id.0]
    }
}
