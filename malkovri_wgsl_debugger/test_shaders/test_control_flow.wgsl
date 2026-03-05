@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    var result = 0u;

    // For loop with nested if/else
    for (var i = 0u; i < 8u; i = i + 1u) {
        if i % 2u == 0u {
            if idx > 32u {
                result = result + i * 2u;
            } else {
                result = result + i;
            }
        } else {
            for (var j = 0u; j < i; j = j + 1u) {
                result = result + 1u;
            }
        }
    }

    // While loop with nested switch
    var count = idx % 16u;
    while count > 0u {
        switch count % 3u {
            case 0u: {
                count = count - 1u;
                result = result + 3u;
            }
            case 1u: {
                if count > 5u {
                    count = count - 2u;
                } else {
                    count = count - 1u;
                }
                result = result + 1u;
            }
            default: {
                count = count - 1u;
                result = result + 2u;
            }
        }
    }

    // Loop (infinite) with break and continue
    var x = idx % 10u;
    loop {
        if x >= 100u {
            break;
        }

        x = x * 2u + 1u;

        if x % 3u == 0u {
            continue;
        }

        // Nested loop inside loop
        var inner = 0u;
        loop {
            if inner >= x % 5u {
                break;
            }
            result = result + inner;
            inner = inner + 1u;
        }
    }

    // Nested for with continuing block
    for (var i = 0u; i < 4u; i = i + 1u) {
        var k = 0u;
        loop {
            if k >= 3u {
                break;
            }

            if i == k {
                k = k + 1u;
                continue;
            }

            result = result + i + k;

            continuing {
                k = k + 1u;
            }
        }
    }

    output[idx] = result;
}
