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
