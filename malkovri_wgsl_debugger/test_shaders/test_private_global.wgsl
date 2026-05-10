var<private> counter: u32 = 7u;

@compute @workgroup_size(1)
fn main() {
    counter = counter + 1u;
    let stop_here: u32 = counter;

    if stop_here == 999u {}
}
