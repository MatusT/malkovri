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
