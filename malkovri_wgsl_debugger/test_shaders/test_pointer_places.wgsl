struct Inner {
    value: u32,
    more: array<u32, 2>,
}

var<private> sink: u32;

fn bump(p: ptr<function, u32>) {
    *p = *p + 5u;
}

fn write_inner(p: ptr<function, Inner>) {
    (*p).value = 7u;
    (*p).more[1] = 9u;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var x: u32 = gid.x;
    bump(&x);

    var s: Inner;
    write_inner(&s);

    let y = x + s.value + s.more[0] + s.more[1];
    let stop_here = y;
    sink = stop_here;
}
