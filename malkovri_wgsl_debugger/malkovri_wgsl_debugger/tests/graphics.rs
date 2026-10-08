use malkovri_wgsl_debugger::{
    DrawConfig, GlobalConstants, Primitive, RunResult, ShaderProgram, VertexAttribute,
    VertexConfig, VertexStepMode,
};
use std::collections::{BTreeMap, HashMap};

#[test]
fn vertex_struct_attributes_use_draw_relative_indices_and_absolute_builtins() {
    let program = ShaderProgram::new(r#"
struct Input { @location(0) point: vec2f, @builtin(vertex_index) vertex: u32 }
struct Output { @builtin(position) position: vec4f, @location(0) @interpolate(flat) id: u32 }
@vertex fn main(input: Input, @location(1) offset: f32, @builtin(instance_index) instance: u32) -> Output {
    return Output(vec4f(input.point + vec2f(offset), 0.0, 1.0), input.vertex + instance * 10u);
}"#).unwrap();
    let config = VertexConfig {
        draw: DrawConfig {
            vertex_count: 2,
            instance_count: 2,
            first_vertex: 7,
            first_instance: 3,
        },
        attributes: BTreeMap::from([
            (
                0,
                VertexAttribute {
                    step_mode: VertexStepMode::Vertex,
                    values: vec![
                        Primitive::F32x2([1., 2.]).into(),
                        Primitive::F32x2([3., 4.]).into(),
                    ],
                },
            ),
            (
                1,
                VertexAttribute {
                    step_mode: VertexStepMode::Instance,
                    values: vec![Primitive::F32(10.).into(), Primitive::F32(20.).into()],
                },
            ),
        ]),
    };
    let mut debugger = program
        .create_debugger(
            0,
            config.clone(),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap();
    assert_eq!(
        debugger.run_to_breakpoint(1, false, &[], None).unwrap(),
        RunResult::Finished
    );
    for (thread, xy, id) in [
        (1, [11., 12.], 37),
        (2, [13., 14.], 38),
        (3, [21., 22.], 47),
        (4, [23., 24.], 48),
    ] {
        let outputs = debugger.thread_shader_outputs(thread).unwrap();
        assert_eq!(
            outputs[0].value,
            Some(Primitive::F32x4([xy[0], xy[1], 0., 1.]).into())
        );
        assert_eq!(outputs[1].value, Some(Primitive::U32(id).into()));
    }
    for invalid in 0..4 {
        let mut bad = config.clone();
        match invalid {
            0 => {
                bad.attributes.remove(&0);
            }
            1 => {
                bad.attributes.get_mut(&0).unwrap().values.pop();
            }
            2 => {
                bad.attributes.get_mut(&1).unwrap().values[0] = Primitive::U32(10).into();
            }
            _ => {
                bad.attributes.insert(9, bad.attributes[&0].clone());
            }
        }
        assert!(
            program
                .create_debugger(0, bad, GlobalConstants::default(), HashMap::new())
                .is_err()
        );
    }
}

#[test]
fn location_json_decoding_rejects_lossy_integers_and_wrong_shapes() {
    let program = ShaderProgram::new("@vertex fn main(@location(0) a: u32, @location(1) b: vec2f) -> @builtin(position) vec4f { return vec4f(b, f32(a), 1.0); }").unwrap();
    let valid = BTreeMap::from([
        (0, serde_json::json!(4294967295u64)),
        (1, serde_json::json!([0.5, 1.0])),
    ]);
    let decoded = program.parse_location_values(0, &valid).unwrap();
    assert_eq!(decoded[&0], Primitive::U32(u32::MAX).into());
    for value in [
        serde_json::json!(-1),
        serde_json::json!(1.5),
        serde_json::json!(4294967296u64),
        serde_json::json!([1]),
    ] {
        let mut bad = valid.clone();
        bad.insert(0, value);
        assert!(program.parse_location_values(0, &bad).is_err());
    }
    for value in [
        serde_json::json!([1.]),
        serde_json::json!([1., 2., 3.]),
        serde_json::json!([1e100, 0.]),
    ] {
        let mut bad = valid.clone();
        bad.insert(1, value);
        assert!(program.parse_location_values(0, &bad).is_err());
    }
}

use malkovri_wgsl_debugger::Value;
use malkovri_wgsl_debugger::graphics::{PixelRange, RasterConfig, VertexOutput, Viewport};

fn full_square() -> Vec<VertexOutput> {
    // Screen-space corners (0,0), (4,0), (0,4), (4,4), with UV = screen position.
    let corners = [
        [-1., 1., 0.5, 1.],
        [1., 1., 0.5, 1.],
        [-1., -1., 0.5, 1.],
        [1., -1., 0.5, 1.],
    ];
    [0, 1, 2, 1, 3, 2]
        .into_iter()
        .map(|i| VertexOutput {
            position: corners[i],
            locations: BTreeMap::from([(
                0,
                Primitive::F32x2([(corners[i][0] + 1.) * 2., (1. - corners[i][1]) * 2.]).into(),
            )]),
            vertex_index: i as u32,
            instance_index: 0,
        })
        .collect()
}

#[test]
fn pixel_ranges_preserve_interpolation_and_generate_only_boundary_helpers() {
    let p = ShaderProgram::new("@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv,0.0,1.0); }").unwrap();
    let full = RasterConfig {
        viewport: Viewport {
            width: 4,
            height: 4,
        },
        pixel_range: None,
    };
    let vertices = full_square();
    let all = p.interpolate_fragments(0, &vertices, &full).unwrap();
    let count: u32 = all.iter().map(|q| q.selected.count_ones()).sum();
    assert_eq!(count, 16, "shared diagonal must be owned once");
    let range = RasterConfig {
        pixel_range: Some(PixelRange {
            from: [1, 1],
            to: [3, 3],
        }),
        ..full.clone()
    };
    let selected = p.interpolate_fragments(0, &vertices, &range).unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|q| q.selected.count_ones())
            .sum::<u32>(),
        4
    );
    for quad in &selected {
        assert_ne!(quad.selected, 0);
        let original = all
            .iter()
            .find(|q| q.origin == quad.origin && q.primitive_index == quad.primitive_index)
            .unwrap();
        assert_eq!(quad.coverage, original.coverage);
        for lane in 0..4 {
            assert_eq!(quad.inputs[lane].locations, original.inputs[lane].locations);
        }
    }
    let one = RasterConfig {
        pixel_range: Some(PixelRange {
            from: [1, 1],
            to: [2, 2],
        }),
        ..full.clone()
    };
    let quads = p.interpolate_fragments(0, &vertices, &one).unwrap();
    assert_eq!(quads.len(), 1);
    assert_eq!(quads[0].origin, [0, 0]);
    assert_eq!(quads[0].selected, 8);
    assert_eq!(
        quads[0].inputs[0].locations[&0],
        Primitive::F32x2([0.5, 0.5]).into()
    );
    for (from, to) in [([1, 1], [1, 2]), ([2, 1], [1, 2]), ([0, 0], [5, 4])] {
        assert!(
            p.interpolate_fragments(
                0,
                &vertices,
                &RasterConfig {
                    pixel_range: Some(PixelRange { from, to }),
                    ..full.clone()
                }
            )
            .is_err()
        );
    }
}

#[test]
fn perspective_linear_flat_and_reciprocal_w_use_correct_vertex_values() {
    let p = ShaderProgram::new(r#"@fragment fn main(@location(0) p: f32, @location(1) @interpolate(linear) l: f32, @location(2) @interpolate(flat) id: u32) -> @location(0) vec4f { return vec4f(p,l,f32(id),1.0); }"#).unwrap();
    let vertices = [[-1., 1., 0.5, 1.], [2., 2., 1., 2.], [-4., -4., 2., 4.]]
        .into_iter()
        .enumerate()
        .map(|(i, position)| VertexOutput {
            position,
            locations: BTreeMap::from([
                (0, Primitive::F32((i * 4) as f32).into()),
                (1, Primitive::F32((i * 4) as f32).into()),
                (2, Primitive::U32(9 + i as u32).into()),
            ]),
            vertex_index: i as u32,
            instance_index: 0,
        })
        .collect::<Vec<_>>();
    let quads = p
        .interpolate_fragments(
            0,
            &vertices,
            &RasterConfig {
                viewport: Viewport {
                    width: 4,
                    height: 4,
                },
                pixel_range: Some(PixelRange {
                    from: [0, 0],
                    to: [1, 1],
                }),
            },
        )
        .unwrap();
    let input = &quads[0].inputs[0]; // weights [0.75,0.125,0.125]
    let Value::Primitive(Primitive::F32(perspective)) = input.locations[&0] else {
        panic!()
    };
    assert!((perspective - (0.5 / 0.84375)).abs() < 1e-6);
    assert_eq!(input.locations[&1], Primitive::F32(1.5).into());
    assert_eq!(input.locations[&2], Primitive::U32(9).into());
    assert_eq!(input.position, [0.5, 0.5, 0.5, 0.84375]);
    assert!(!input.front_facing);
    let mut reversed = vertices.clone();
    reversed.swap(1, 2);
    let q = p
        .interpolate_fragments(
            0,
            &reversed,
            &RasterConfig {
                viewport: Viewport {
                    width: 4,
                    height: 4,
                },
                pixel_range: None,
            },
        )
        .unwrap();
    assert!(q[0].inputs[0].front_facing);
    assert_eq!(q[0].inputs[0].locations[&2], Primitive::U32(9).into());
}

#[test]
fn rasterizer_handles_empty_coverage_and_odd_viewport() {
    let p=ShaderProgram::new("@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv,0.0,1.0); }").unwrap();
    let config = RasterConfig {
        viewport: Viewport {
            width: 3,
            height: 3,
        },
        pixel_range: None,
    };
    let quads = p.interpolate_fragments(0, &full_square(), &config).unwrap();
    assert_eq!(
        quads.iter().map(|q| q.selected.count_ones()).sum::<u32>(),
        9
    );
    assert!(quads.iter().any(|q| q.origin == [2, 2] && q.selected == 1));
    let mut vertices = full_square();
    for v in &mut vertices {
        v.position[0] += 10.;
    }
    assert!(
        p.interpolate_fragments(0, &vertices, &config)
            .unwrap()
            .is_empty()
    );
    vertices[0].position[3] = 0.;
    assert!(p.interpolate_fragments(0, &vertices, &config).is_err());
}

#[test]
fn fragment_quads_resolve_struct_inputs_and_suppress_helper_and_discard_writes() {
    use malkovri_wgsl_debugger::{ExecutionConfig, ResourceBinding};
    let program = ShaderProgram::new(
        r#"
@group(0) @binding(0) var<storage, read_write> values: array<u32>;
struct In { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
fn maybe_discard(x: f32) { if x > 1.0 { discard; } }
@fragment fn main(input: In) -> @location(0) vec4f {
    let index = u32(input.pos.y) * 4u + u32(input.pos.x);
    values[index] = 1u;
    maybe_discard(input.pos.x);
    values[index] = 2u;
    return vec4f(input.uv, 0.0, 1.0);
}"#,
    )
    .unwrap();
    let config = RasterConfig {
        viewport: Viewport {
            width: 4,
            height: 4,
        },
        pixel_range: Some(PixelRange {
            from: [1, 1],
            to: [2, 2],
        }),
    };
    let quads = program
        .interpolate_fragments(0, &full_square(), &config)
        .unwrap();
    let mut debugger = program
        .create_debugger(
            0,
            ExecutionConfig::FragmentQuads(quads),
            GlobalConstants::default(),
            HashMap::from([(
                ResourceBinding {
                    group: 0,
                    binding: 0,
                },
                Value::Array(vec![Primitive::U32(0).into(); 16]),
            )]),
        )
        .unwrap();
    assert_eq!(
        debugger.run_to_breakpoint(1, false, &[], None).unwrap(),
        RunResult::Finished
    );
    assert_eq!(debugger.threads().len(), 4);
    for thread in 1..=4 {
        assert!(
            debugger
                .thread_shader_outputs(thread)
                .unwrap()
                .iter()
                .all(|output| output.value.is_none())
        );
    }
    let info = debugger.thread_fragment_info(4).unwrap().unwrap();
    assert!(info.discarded && info.helper);
    let globals = debugger.global_variables();
    let values = &globals
        .iter()
        .find(|v| v.name.as_deref() == Some("values"))
        .unwrap()
        .value;
    for i in 0..16 {
        assert_eq!(
            values.index_into(i),
            Primitive::U32(if i == 5 { 1 } else { 0 }).into()
        );
    }
}

#[test]
fn empty_fragment_selection_finishes_without_invocations() {
    let program =
        ShaderProgram::new("@fragment fn main() -> @location(0) vec4f { return vec4f(1.0); }")
            .unwrap();
    let mut debugger = program
        .create_debugger(
            0,
            malkovri_wgsl_debugger::ExecutionConfig::FragmentQuads(vec![]),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap();
    assert_eq!(
        debugger.run_to_breakpoint(1, false, &[], None).unwrap(),
        RunResult::Finished
    );
    assert!(debugger.threads().is_empty());
    assert!(debugger.current_location().is_none());
    assert!(debugger.entry_point_output().is_none());
    assert!(debugger.global_variables().is_empty());
    assert!(debugger.local_variables().is_empty());
    assert!(debugger.argument_variables().is_empty());
}

#[test]
fn quad_derivatives_use_shader_operands_across_calls_loops_and_helpers() {
    let program = ShaderProgram::new(
        r#"
fn gradient(uv: vec2f) -> vec4f {
    let v=uv.x*uv.y;
    return vec4f(dpdxFine(v),dpdyFine(v),dpdxCoarse(v),fwidth(v));
}
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
    if uv.x > 1.0 { discard; }
    var sum=vec4f(0.0);
    for(var i=0u;i<3u;i++) { sum += gradient(uv*f32(i+1u)); }
    return sum;
}"#,
    )
    .unwrap();
    let config = RasterConfig {
        viewport: Viewport {
            width: 4,
            height: 4,
        },
        pixel_range: Some(PixelRange {
            from: [0, 0],
            to: [1, 1],
        }),
    };
    let quads = program
        .interpolate_fragments(0, &full_square(), &config)
        .unwrap();
    let mut debugger = program
        .create_debugger(
            0,
            malkovri_wgsl_debugger::ExecutionConfig::FragmentQuads(quads),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap();
    assert_eq!(
        debugger.run_to_breakpoint(1, false, &[], None).unwrap(),
        RunResult::Finished
    );
    assert_eq!(
        debugger.thread_shader_outputs(1).unwrap()[0].value,
        Some(Primitive::F32x4([7., 7., 7., 14.]).into())
    );
    for id in 2..=4 {
        assert!(
            debugger.thread_shader_outputs(id).unwrap()[0]
                .value
                .is_none()
        );
    }
}

#[test]
fn nonuniform_derivatives_with_missing_lanes_fail_clearly() {
    let shader = "@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { if uv.x > 1.0 {return vec4f(0.0);} return vec4f(dpdx(uv),0.0,1.0); }";

    let program = ShaderProgram::new(&format!(
        "diagnostic(off, derivative_uniformity);\n{shader}"
    ))
    .unwrap();
    let quads = program
        .interpolate_fragments(
            0,
            &full_square(),
            &RasterConfig {
                viewport: Viewport {
                    width: 4,
                    height: 4,
                },
                pixel_range: Some(PixelRange {
                    from: [0, 0],
                    to: [1, 1],
                }),
            },
        )
        .unwrap();
    let mut debugger = program
        .create_debugger(
            0,
            malkovri_wgsl_debugger::ExecutionConfig::FragmentQuads(quads),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap();
    let error = debugger.run_to_breakpoint(1, false, &[], None).unwrap_err();
    assert!(error.to_string().contains("quad lane returned"), "{error}");
}

#[test]
fn selected_pixel_continue_advances_only_its_quad_and_honors_peer_breakpoints() {
    let program = ShaderProgram::new(
        r#"
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
    var value=uv;
    if uv.x > 1.0 {
        value += vec2f(1.0);
    }
    let gradient = dpdx(value);
    return vec4f(gradient,0.0,1.0);
}"#,
    )
    .unwrap();
    let quads = program
        .interpolate_fragments(
            0,
            &full_square(),
            &RasterConfig {
                viewport: Viewport {
                    width: 4,
                    height: 4,
                },
                pixel_range: None,
            },
        )
        .unwrap();
    let mut debugger = program
        .create_debugger(
            0,
            malkovri_wgsl_debugger::ExecutionConfig::FragmentQuads(quads),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap();
    let other_before = debugger.thread_current_location(5).unwrap().line;
    assert_eq!(
        debugger.run_to_breakpoint(1, true, &[5], None).unwrap(),
        RunResult::Breakpoint
    );
    assert_ne!(debugger.focused_thread_id(), 1);
    assert_eq!(
        debugger.thread_current_location(5).unwrap().line,
        other_before
    );
    assert_eq!(
        debugger.run_to_breakpoint(1, true, &[], None).unwrap(),
        RunResult::InvocationFinished
    );
    assert_eq!(
        debugger.thread_shader_outputs(1).unwrap()[0].value,
        Some(Primitive::F32x4([2., 1., 0., 1.]).into())
    );
    assert_eq!(
        debugger.thread_current_location(5).unwrap().line,
        other_before
    );
}

#[test]
fn linked_stages_retain_vertex_outputs_share_resources_and_reset_private_state() {
    use malkovri_wgsl_debugger::{
        ResourceBinding,
        graphics::{GraphicsSession, GraphicsSource},
    };
    let program = ShaderProgram::new(
        r#"
@group(0) @binding(0) var<storage, read_write> buffer: array<u32>;
var<private> marker:u32=3u;
struct Out { @builtin(position) pos:vec4f, @location(0) uv:vec2f }
@vertex fn vs(@builtin(vertex_index) vertex:u32) -> Out {
    let points=array<vec2f,3>(vec2f(-1.0,1.0),vec2f(1.0,1.0),vec2f(-1.0,-1.0));
    buffer[vertex]=vertex+5u;
    marker=99u;
    return Out(vec4f(points[vertex],0.5,1.0),points[vertex]);
}
@fragment fn fs(@location(0) uv:vec2f) -> @location(0) vec4f {
    return vec4f(uv,f32(buffer[0]),f32(marker));
}"#,
    )
    .unwrap();
    let raster = RasterConfig {
        viewport: Viewport {
            width: 4,
            height: 4,
        },
        pixel_range: Some(PixelRange {
            from: [0, 0],
            to: [1, 1],
        }),
    };
    let binding = ResourceBinding {
        group: 0,
        binding: 0,
    };
    let mut session = GraphicsSession::new(
        program.clone(),
        1,
        GraphicsSource::Vertex {
            entry: 0,
            config: VertexConfig {
                draw: DrawConfig {
                    vertex_count: 3,
                    ..Default::default()
                },
                ..Default::default()
            },
        },
        raster.clone(),
        GlobalConstants::default(),
        HashMap::from([(binding, Value::Array(vec![Primitive::U32(0).into(); 3]))]),
    )
    .unwrap();
    assert!(session.advance_stage().is_err());
    assert!(session.has_next_stage());
    assert_eq!(
        session
            .debugger_mut()
            .run_to_breakpoint(1, false, &[], None)
            .unwrap(),
        RunResult::Finished
    );
    let captured = session.debugger().vertex_outputs().unwrap();
    session.advance_stage().unwrap();
    assert!(!session.has_next_stage());
    assert_eq!(session.vertex_outputs().len(), 3);
    assert_eq!(session.debugger().threads()[0].id, 4);
    assert!(session.debugger().thread_shader_outputs(1).is_err());
    assert_eq!(
        session
            .debugger_mut()
            .run_to_breakpoint(4, false, &[], None)
            .unwrap(),
        RunResult::Finished
    );
    let output = session.debugger().thread_shader_outputs(4).unwrap()[0]
        .value
        .clone();
    assert_eq!(output, Some(Primitive::F32x4([-0.75, 0.75, 5., 3.]).into()));
    let mut supplied = GraphicsSession::new(
        program,
        1,
        GraphicsSource::Outputs(captured),
        raster,
        GlobalConstants::default(),
        HashMap::from([(binding, Value::Array(vec![Primitive::U32(5).into(); 3]))]),
    )
    .unwrap();
    supplied
        .debugger_mut()
        .run_to_breakpoint(1, false, &[], None)
        .unwrap();
    assert_eq!(
        supplied.debugger().thread_shader_outputs(1).unwrap()[0].value,
        output
    );
}
