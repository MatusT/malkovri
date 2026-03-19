#[derive(Copy, Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct EntryPointInputs {
    // vertex
    pub base_instance: u32,
    pub base_vertex: i32,
    pub clip_distance: [f32; 8],
    pub cull_distance: [f32; 8],
    pub instance_index: u32,
    pub point_size: f32,
    pub vertex_index: u32,
    pub draw_id: u32,

    // fragment
    pub position: [f32; 4],
    pub view_index: i32,
    pub frag_depth: f32,
    pub point_coord: [f32; 2],
    pub front_facing: bool,
    pub primitive_index: u32,
    pub sample_index: u32,
    pub sample_mask: u32,

    // compute
    pub global_invocation_id: [u32; 3],
    pub local_invocation_id: [u32; 3],
    pub local_invocation_index: u32,
    pub workgroup_id: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub num_workgroups: [u32; 3],

    // subgroup
    pub num_subgroups: u32,
    pub subgroup_id: u32,
    pub subgroup_size: u32,
    pub subgroup_invocation_id: u32,
}
