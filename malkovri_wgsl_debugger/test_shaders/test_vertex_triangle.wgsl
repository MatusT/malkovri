@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4f {
    // Predefined coordinates for a triangle in Normalized Device Coordinates (NDC)
    var pos = array<vec2f, 3>(
        vec2f(0.0, 0.5),   // Top center
        vec2f(-0.5, -0.5), // Bottom left
        vec2f(0.5, -0.5)   // Bottom right
    );

    // Return the position as a 4D vector (x, y, z, w)
    return vec4f(pos[in_vertex_index], 0.0, 1.0);
}
