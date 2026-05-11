var<workgroup> shared_value: u32;
var<private> observed: u32;

@compute @workgroup_size(2, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        shared_value = 40u;
    }
    workgroupBarrier();
    observed = shared_value + lid.x;
}
