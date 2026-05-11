var<private> sink: u32;

@compute @workgroup_size(2, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 1u) {
        sink = lid.x;
    }
    let after = lid.x;
}
