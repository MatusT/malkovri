@compute @workgroup_size(1)
fn main() {
    var x: u32 = 1u;
    x = 5u;
    var y: u32 = x;

    // The debugger should stop on the `if` below, after y has been initialized.
    let stop_here: u32 = y;

    if stop_here == 999u {}
}
