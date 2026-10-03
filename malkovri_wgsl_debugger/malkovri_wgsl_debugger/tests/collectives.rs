use std::collections::HashMap;

use malkovri_wgsl_debugger::{
    GlobalConstants, Primitive, ShaderProgram, StepResult, Value, WorkgroupConfig,
};

fn execute(source: &str, invocation_count: u32) -> Vec<HashMap<String, Value>> {
    let mut debugger = ShaderProgram::new(source)
        .unwrap()
        .create_debugger(
            0,
            WorkgroupConfig::new([invocation_count, 1, 1], [0, 0, 0], 4, [1, 1, 1]).unwrap(),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap();
    for _ in 0..1_000 {
        if debugger.step_all().unwrap() == StepResult::Finished {
            return debugger
                .threads()
                .into_iter()
                .map(|thread| {
                    debugger
                        .thread_global_variables(thread.id)
                        .unwrap()
                        .into_iter()
                        .map(|variable| (variable.name.unwrap(), variable.value))
                        .collect()
                })
                .collect();
        }
    }
    panic!("shader did not finish within 1,000 steps");
}

#[test]
fn subgroup_min_and_max_preserve_numeric_types_and_vector_components() {
    let globals = execute(
        r#"
var<private> min_float: vec3f;
var<private> max_float: vec3f;
var<private> min_signed: vec2i;
var<private> max_signed: vec2i;
var<private> min_unsigned: vec4u;
var<private> max_unsigned: vec4u;
@compute @workgroup_size(4)
fn main(@builtin(subgroup_invocation_id) lane: u32) {
    let floats = vec3f(f32(lane) - 2.0, 8.0 - f32(lane), f32(lane % 2u));
    let signed_values = vec2i(i32(lane) - 3, -i32(lane));
    let unsigned_values = vec4u(lane + 1u, 9u - lane, lane % 2u, 4u);
    min_float = subgroupMin(floats);
    max_float = subgroupMax(floats);
    min_signed = subgroupMin(signed_values);
    max_signed = subgroupMax(signed_values);
    min_unsigned = subgroupMin(unsigned_values);
    max_unsigned = subgroupMax(unsigned_values);
}
"#,
        4,
    );
    for values in globals {
        for (name, expected) in [
            ("min_float", Primitive::F32x3([-2.0, 5.0, 0.0])),
            ("max_float", Primitive::F32x3([1.0, 8.0, 1.0])),
            ("min_signed", Primitive::I32x2([-3, -3])),
            ("max_signed", Primitive::I32x2([0, 0])),
            ("min_unsigned", Primitive::U32x4([1, 6, 0, 4])),
            ("max_unsigned", Primitive::U32x4([4, 9, 1, 4])),
        ] {
            assert_eq!(values[name], Value::from(expected), "{name}");
        }
    }
}

#[test]
fn subgroup_scans_preserve_identities_and_reset_for_partial_subgroups() {
    let globals = execute(
        r#"
var<private> inclusive_add: vec2f;
var<private> exclusive_add: vec3i;
var<private> inclusive_mul: vec4u;
var<private> exclusive_mul: f32;
@compute @workgroup_size(6)
fn main(@builtin(subgroup_invocation_id) lane: u32) {
    let value = lane + 1u;
    inclusive_add = subgroupInclusiveAdd(vec2f(f32(value), 2.0));
    exclusive_add = subgroupExclusiveAdd(vec3i(i32(value), -i32(value), 1));
    inclusive_mul = subgroupInclusiveMul(vec4u(value, 2u, 1u, 3u));
    exclusive_mul = subgroupExclusiveMul(f32(value));
}
"#,
        6,
    );
    for (index, values) in globals.iter().enumerate() {
        let lane = index % 4;
        let previous_sum = [0, 1, 3, 6][lane];
        let inclusive_product = [1, 2, 6, 24][lane];
        let previous_product = [1.0, 1.0, 2.0, 6.0][lane];
        assert_eq!(
            values["inclusive_add"],
            Value::from(Primitive::F32x2([
                (previous_sum + lane as i32 + 1) as f32,
                2.0 * (lane + 1) as f32,
            ]))
        );
        assert_eq!(
            values["exclusive_add"],
            Value::from(Primitive::I32x3(
                [previous_sum, -previous_sum, lane as i32,]
            ))
        );
        assert_eq!(
            values["inclusive_mul"],
            Value::from(Primitive::U32x4([
                inclusive_product,
                2u32.pow(lane as u32 + 1),
                1,
                3u32.pow(lane as u32 + 1),
            ]))
        );
        assert_eq!(
            values["exclusive_mul"],
            Value::from(Primitive::F32(previous_product))
        );
    }
}
