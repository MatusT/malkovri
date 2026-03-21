@compute @workgroup_size(1)
fn main() {
    var acc: u32 = 0u;

    for (var i = 0u; i < 2u; i = i + 1u) {
        var snapshot: u32 = acc;
        acc = acc + 1u;

        if i == 1u {
            // The debugger should stop here on the second iteration, after
            // snapshot has been initialized from the pre-increment acc value.
            if snapshot == 999u {}
        }
    }
}
