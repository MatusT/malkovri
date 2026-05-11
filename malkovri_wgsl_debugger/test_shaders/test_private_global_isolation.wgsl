var<private> counter: u32;

@compute @workgroup_size(2, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    counter = 10u + lid.x;
    let stop_here = counter;
}
