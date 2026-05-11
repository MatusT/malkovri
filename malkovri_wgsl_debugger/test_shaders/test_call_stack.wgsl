var<private> sink: u32;

fn helper(value: u32) -> u32 {
    sink = value * 2u;
    return sink;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let x = helper(lid.x + 1u);
    sink = x;
}
