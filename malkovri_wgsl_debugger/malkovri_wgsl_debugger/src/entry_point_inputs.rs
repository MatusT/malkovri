use crate::{Primitive, Value};

/// Per-thread compute built-in inputs, computed by the evaluator from the
/// thread's position within the workgroup.
#[derive(Clone, Debug)]
pub(crate) struct ComputeThreadInputs {
    local_invocation_id: [u32; 3],
    local_invocation_index: u32,
    global_invocation_id: [u32; 3],
    workgroup_id: [u32; 3],
    subgroup_id: u32,
    subgroup_invocation_id: u32,
}

impl ComputeThreadInputs {
    pub(crate) fn subgroup_id(&self) -> u32 {
        self.subgroup_id
    }
    pub(crate) fn subgroup_invocation_id(&self) -> u32 {
        self.subgroup_invocation_id
    }

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
    vertex_index: u32,
    instance_index: u32,
}

/// Per-thread fragment built-in inputs, computed by the evaluator from the
/// fragment's position in the render target.
#[derive(Clone, Debug, Default)]
pub(crate) struct FragmentThreadInputs {
    position: [f32; 4],
    front_facing: bool,
    sample_index: u32,
    sample_mask: u32,
    primitive_index: u32,
}

/// Constant globals that are the same across all threads, set by the user.
#[derive(Copy, Clone, Debug, Default, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GlobalConstants {
    // vertex
    #[serde(alias = "base_instance")]
    base_instance: u32,
    #[serde(alias = "base_vertex")]
    base_vertex: i32,
    #[serde(alias = "clip_distance")]
    clip_distance: [f32; 8],
    #[serde(alias = "cull_distance")]
    cull_distance: [f32; 8],
    #[serde(alias = "point_size")]
    point_size: f32,
    #[serde(alias = "draw_id")]
    draw_id: u32,

    // fragment
    #[serde(alias = "view_index")]
    view_index: i32,
    #[serde(alias = "frag_depth")]
    frag_depth: f32,
    #[serde(alias = "point_coord")]
    point_coord: [f32; 2],

    // compute
    #[serde(alias = "workgroup_size")]
    workgroup_size: [u32; 3],
    #[serde(alias = "num_workgroups")]
    num_workgroups: [u32; 3],
    #[serde(alias = "subgroup_size")]
    subgroup_size: u32,
    #[serde(alias = "num_subgroups")]
    num_subgroups: u32,
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

impl GlobalConstants {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn base_instance(&self) -> u32 {
        self.base_instance
    }
    pub fn with_base_instance(mut self, value: u32) -> Self {
        self.base_instance = value;
        self
    }
    pub fn base_vertex(&self) -> i32 {
        self.base_vertex
    }
    pub fn with_base_vertex(mut self, value: i32) -> Self {
        self.base_vertex = value;
        self
    }
    pub fn clip_distance(&self) -> [f32; 8] {
        self.clip_distance
    }
    pub fn with_clip_distance(mut self, value: [f32; 8]) -> Self {
        self.clip_distance = value;
        self
    }
    pub fn cull_distance(&self) -> [f32; 8] {
        self.cull_distance
    }
    pub fn with_cull_distance(mut self, value: [f32; 8]) -> Self {
        self.cull_distance = value;
        self
    }
    pub fn point_size(&self) -> f32 {
        self.point_size
    }
    pub fn with_point_size(mut self, value: f32) -> Self {
        self.point_size = value;
        self
    }
    pub fn draw_id(&self) -> u32 {
        self.draw_id
    }
    pub fn with_draw_id(mut self, value: u32) -> Self {
        self.draw_id = value;
        self
    }
    pub fn view_index(&self) -> i32 {
        self.view_index
    }
    pub fn with_view_index(mut self, value: i32) -> Self {
        self.view_index = value;
        self
    }
    pub fn frag_depth(&self) -> f32 {
        self.frag_depth
    }
    pub fn with_frag_depth(mut self, value: f32) -> Self {
        self.frag_depth = value;
        self
    }
    pub fn point_coord(&self) -> [f32; 2] {
        self.point_coord
    }
    pub fn with_point_coord(mut self, value: [f32; 2]) -> Self {
        self.point_coord = value;
        self
    }
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }
    pub fn num_workgroups(&self) -> [u32; 3] {
        self.num_workgroups
    }
    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }
    pub fn num_subgroups(&self) -> u32 {
        self.num_subgroups
    }
    pub(crate) fn set_compute_configuration(
        &mut self,
        size: [u32; 3],
        count: [u32; 3],
        subgroup_size: u32,
    ) {
        self.workgroup_size = size;
        self.num_workgroups = count;
        self.subgroup_size = subgroup_size;
        self.num_subgroups = (size[0] * size[1] * size[2]).div_ceil(subgroup_size);
    }
}
