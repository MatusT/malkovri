@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let outer: u32 = global_id.x;

    if true {
        // The debugger should stop on the `if` below while still inside this block.
        let stop_here: u32 = outer + 1u;

        if stop_here == 999u {}
    }
}
