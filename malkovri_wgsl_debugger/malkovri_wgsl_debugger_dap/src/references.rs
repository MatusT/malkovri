use std::collections::HashMap;

use malkovri_wgsl_debugger::{DebugFrameId, DebugThreadId};

use crate::{error::DebugAdapterError, protocol::StackFrameId};

#[derive(Clone, Copy)]
pub(crate) struct FrameReference {
    pub thread_id: DebugThreadId,
    pub frame_id: DebugFrameId,
}

#[derive(Clone, Copy)]
pub(crate) enum ScopeKind {
    Locals,
    Arguments,
    Globals,
}

/// DAP references belong to one suspension. IDs are never reused when resumed.
#[derive(Default)]
pub(crate) struct References {
    next_frame: StackFrameId,
    next_scope: u32,
    frames: HashMap<StackFrameId, FrameReference>,
    scopes: HashMap<u32, (FrameReference, ScopeKind)>,
}

impl References {
    pub fn clear(&mut self) {
        self.frames.clear();
        self.scopes.clear();
    }

    pub fn insert_frame(&mut self, frame: FrameReference) -> StackFrameId {
        self.next_frame += 1;
        self.frames.insert(self.next_frame, frame);
        self.next_frame
    }

    pub fn frame(&self, id: StackFrameId) -> Result<FrameReference, DebugAdapterError> {
        self.frames.get(&id).copied().ok_or_else(|| {
            DebugAdapterError::InvalidProgram(format!(
                "unknown or expired stack frame {id}; request stackTrace again"
            ))
        })
    }

    pub fn insert_scope(&mut self, frame: FrameReference, kind: ScopeKind) -> u32 {
        self.next_scope += 1;
        self.scopes.insert(self.next_scope, (frame, kind));
        self.next_scope
    }

    pub fn scope(&self, id: u32) -> Result<(FrameReference, ScopeKind), DebugAdapterError> {
        self.scopes.get(&id).copied().ok_or_else(|| {
            DebugAdapterError::InvalidProgram(format!(
                "unknown or expired variable reference {id}; request scopes again"
            ))
        })
    }
}
