use crate::entry_point_inputs::{ComputeThreadInputs, FragmentThreadInputs, VertexThreadInputs};

/// Builtin input state for a single shader invocation.
#[derive(Clone, Debug)]
pub(crate) struct EvaluatorThread {
    pub(crate) compute_inputs: ComputeThreadInputs,
    pub(crate) vertex_inputs: VertexThreadInputs,
    pub(crate) fragment_inputs: FragmentThreadInputs,
}
