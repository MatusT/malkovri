use crate::{Primitive, Value};

/// Per-thread compute built-in inputs, computed by the evaluator from the
/// thread's position within the workgroup.
#[derive(Clone, Debug)]
pub(crate) struct ComputeThreadInputs {
    pub local_invocation_id: [u32; 3],
    pub local_invocation_index: u32,
    pub global_invocation_id: [u32; 3],
    pub workgroup_id: [u32; 3],
    pub subgroup_id: u32,
    pub subgroup_invocation_id: u32,
}

impl ComputeThreadInputs {
    /// Derive all compute built-in IDs from minimal inputs.
    pub(crate) fn new(
        local_invocation_id: [u32; 3],
        workgroup_size: [u32; 3],
        workgroup_id: [u32; 3],
        subgroup_size: u32,
    ) -> Self {
        let [wx, wy, _wz] = workgroup_size;
        let [x, y, z] = local_invocation_id;
        let local_invocation_index = x + y * wx + z * wx * wy;
        Self {
            local_invocation_id,
            local_invocation_index,
            global_invocation_id: [
                workgroup_id[0] * workgroup_size[0] + x,
                workgroup_id[1] * workgroup_size[1] + y,
                workgroup_id[2] * workgroup_size[2] + z,
            ],
            workgroup_id,
            subgroup_id: local_invocation_index / subgroup_size,
            subgroup_invocation_id: local_invocation_index % subgroup_size,
        }
    }
}

/// Per-thread vertex built-in inputs, computed by the evaluator from the
/// vertex/instance invocation index.
#[derive(Clone, Debug, Default)]
pub(crate) struct VertexThreadInputs {
    pub vertex_index: u32,
    pub instance_index: u32,
}

/// Per-thread fragment built-in inputs, computed by the evaluator from the
/// fragment's position in the render target.
#[derive(Clone, Debug, Default)]
pub(crate) struct FragmentThreadInputs {
    pub position: [f32; 4],
    pub front_facing: bool,
    pub sample_index: u32,
    pub sample_mask: u32,
    pub primitive_index: u32,
}

/// Constant globals that are the same across all threads, set by the user.
#[derive(Copy, Clone, Debug, Default, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GlobalConstants {
    // vertex
    #[serde(alias = "base_instance")]
    pub base_instance: u32,
    #[serde(alias = "base_vertex")]
    pub base_vertex: i32,
    #[serde(alias = "clip_distance")]
    pub clip_distance: [f32; 8],
    #[serde(alias = "cull_distance")]
    pub cull_distance: [f32; 8],
    #[serde(alias = "point_size")]
    pub point_size: f32,
    #[serde(alias = "draw_id")]
    pub draw_id: u32,

    // fragment
    #[serde(alias = "view_index")]
    pub view_index: i32,
    #[serde(alias = "frag_depth")]
    pub frag_depth: f32,
    #[serde(alias = "point_coord")]
    pub point_coord: [f32; 2],

    // compute
    #[serde(alias = "workgroup_size")]
    pub workgroup_size: [u32; 3],
    #[serde(alias = "num_workgroups")]
    pub num_workgroups: [u32; 3],
    #[serde(alias = "subgroup_size")]
    pub subgroup_size: u32,
    #[serde(alias = "num_subgroups")]
    pub num_subgroups: u32,
}

/// Inputs for exactly one invocation of the session's selected shader stage.
#[derive(Clone, Debug)]
pub(crate) enum InvocationInputs {
    Compute(ComputeThreadInputs),
    Vertex(VertexThreadInputs),
    Fragment(FragmentThreadInputs),
}

impl InvocationInputs {
    pub fn compute(&self) -> Option<&ComputeThreadInputs> {
        match self {
            Self::Compute(inputs) => Some(inputs),
            _ => None,
        }
    }

    pub fn builtin(&self, builtin: naga::BuiltIn, globals: &GlobalConstants) -> Value {
        use naga::BuiltIn as B;
        match (self, builtin) {
            (Self::Compute(inputs), B::GlobalInvocationId) => {
                Primitive::U32x3(inputs.global_invocation_id).into()
            }
            (Self::Compute(inputs), B::LocalInvocationId) => {
                Primitive::U32x3(inputs.local_invocation_id).into()
            }
            (Self::Compute(inputs), B::LocalInvocationIndex) => {
                Primitive::U32(inputs.local_invocation_index).into()
            }
            (Self::Compute(inputs), B::WorkGroupId) => Primitive::U32x3(inputs.workgroup_id).into(),
            (Self::Compute(inputs), B::SubgroupId) => Primitive::U32(inputs.subgroup_id).into(),
            (Self::Compute(inputs), B::SubgroupInvocationId) => {
                Primitive::U32(inputs.subgroup_invocation_id).into()
            }
            (Self::Vertex(inputs), B::VertexIndex) => Primitive::U32(inputs.vertex_index).into(),
            (Self::Vertex(inputs), B::InstanceIndex) => {
                Primitive::U32(inputs.instance_index).into()
            }
            (Self::Fragment(inputs), B::Position { .. }) => {
                Primitive::F32x4(inputs.position).into()
            }
            (Self::Fragment(inputs), B::FrontFacing) => {
                Primitive::U32(u32::from(inputs.front_facing)).into()
            }
            (Self::Fragment(inputs), B::SampleIndex) => Primitive::U32(inputs.sample_index).into(),
            (Self::Fragment(inputs), B::SampleMask) => Primitive::U32(inputs.sample_mask).into(),
            (Self::Fragment(inputs), B::PrimitiveIndex) => {
                Primitive::U32(inputs.primitive_index).into()
            }
            (_, B::BaseInstance) => Primitive::U32(globals.base_instance).into(),
            (_, B::BaseVertex) => Primitive::I32(globals.base_vertex).into(),
            (_, B::ClipDistance) => Value::Array(
                globals
                    .clip_distance
                    .iter()
                    .map(|&value| Primitive::F32(value).into())
                    .collect(),
            ),
            (_, B::CullDistance) => Value::Array(
                globals
                    .cull_distance
                    .iter()
                    .map(|&value| Primitive::F32(value).into())
                    .collect(),
            ),
            (_, B::PointSize) => Primitive::F32(globals.point_size).into(),
            (_, B::DrawID) => Primitive::U32(globals.draw_id).into(),
            (_, B::ViewIndex) => Primitive::I32(globals.view_index).into(),
            (_, B::FragDepth) => Primitive::F32(globals.frag_depth).into(),
            (_, B::PointCoord) => Primitive::F32x2(globals.point_coord).into(),
            (_, B::WorkGroupSize) => Primitive::U32x3(globals.workgroup_size).into(),
            (_, B::NumWorkGroups) => Primitive::U32x3(globals.num_workgroups).into(),
            (_, B::NumSubgroups) => Primitive::U32(globals.num_subgroups).into(),
            (_, B::SubgroupSize) => Primitive::U32(globals.subgroup_size).into(),
            _ => Value::Uninitialized,
        }
    }
}
