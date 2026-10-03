use std::collections::HashMap;

use malkovri_wgsl_debugger::{
    GlobalConstants, Primitive, ShaderProgram, StepResult, Value, WorkgroupConfig,
};

fn execute(source: &str) -> HashMap<String, Value> {
    let mut debugger = ShaderProgram::new(source)
        .unwrap()
        .create_debugger(
            0,
            WorkgroupConfig::default(),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap();
    for _ in 0..1_000 {
        if debugger.step_all().unwrap() == StepResult::Finished {
            return debugger
                .global_variables()
                .into_iter()
                .map(|variable| {
                    let (name, value) = variable.into_parts();
                    (name.unwrap(), value)
                })
                .collect();
        }
    }
    panic!("shader did not finish within 1,000 steps");
}

fn u32_result(source: &str) -> u32 {
    match &execute(source)["result"] {
        Value::Primitive(Primitive::U32(value)) => *value,
        other => panic!("expected u32 result, got {other:?}"),
    }
}

#[test]
fn let_binding_keeps_the_value_loaded_at_its_declaration() {
    assert_eq!(
        u32_result(
            r#"
var<private> result: u32;
@compute @workgroup_size(1) fn main() {
    var x = 1u;
    let saved = x;
    x = 2u;
    result = saved;
}
"#
        ),
        1
    );
}

#[test]
fn emitted_expressions_are_refreshed_on_each_loop_iteration() {
    assert_eq!(
        u32_result(
            r#"
var<private> result: u32;
@compute @workgroup_size(1) fn main() {
    for (var i = 0u; i < 3u; i++) {
        var value = i;
        let saved = value;
        value = 100u;
        result += saved;
    }
}
"#
        ),
        3
    );
}

#[test]
fn pointer_binding_keeps_its_original_index() {
    assert_eq!(
        u32_result(
            r#"
var<private> result: u32;
@compute @workgroup_size(1) fn main() {
    var values = array<u32, 2>(10u, 20u);
    var index = 0u;
    let selected = &values[index];
    index = 1u;
    *selected = 42u;
    result = values[0];
}
"#
        ),
        42
    );
}

#[test]
fn load_before_function_call_keeps_its_value() {
    assert_eq!(
        u32_result(
            r#"
var<private> result: u32;
var<private> input: u32 = 1u;
fn change() -> u32 { input = 100u; return 2u; }
@compute @workgroup_size(1) fn main() {
    result = input + change();
}
"#
        ),
        3
    );
}

#[test]
fn switch_executes_the_body_for_every_selector_of_a_shared_case() {
    for selector in [1, 2, 3] {
        let source = format!(
            r#"
var<private> result: u32;
@compute @workgroup_size(1) fn main() {{
    var selector = {selector}u;
    switch selector {{
        case 1u, 2u: {{ result = 42u; }}
        default: {{ result = 99u; }}
    }}
}}
"#
        );
        assert_eq!(u32_result(&source), if selector == 3 { 99 } else { 42 });
    }
}

#[test]
fn select_uses_each_boolean_vector_component() {
    let globals = execute(
        r#"
var<private> result: vec2u;
@compute @workgroup_size(1) fn main() {
    var a = vec2u(10u, 20u);
    var b = vec2u(30u, 40u);
    result = select(a, b, a < vec2u(15u));
}
"#,
    );
    assert!(matches!(
        globals["result"],
        Value::Primitive(Primitive::U32x2([30, 20]))
    ));
}

#[test]
fn ldexp_handles_negative_and_large_positive_exponents() {
    let globals = execute(
        r#"
var<private> result: vec2f;
@compute @workgroup_size(1) fn main() {
    var value = vec2f(8.0, 1.0);
    var exponent = vec2i(-1, 40);
    result = ldexp(value, exponent);
}
"#,
    );
    assert!(matches!(
        globals["result"],
        Value::Primitive(Primitive::F32x2([4.0, 1099511627776.0]))
    ));
}

#[test]
fn missing_or_invalid_entry_point_is_an_error() {
    for (source, index) in [("", 0), ("@compute @workgroup_size(1) fn main() {}", 1)] {
        assert!(
            ShaderProgram::new(source)
                .unwrap()
                .create_debugger(
                    index,
                    WorkgroupConfig::default(),
                    GlobalConstants::default(),
                    HashMap::new()
                )
                .is_err()
        );
    }
}

#[test]
fn select_accepts_a_literal_boolean_mask() {
    let globals = execute(
        r#"
var<private> result: vec2f;
@compute @workgroup_size(1) fn main() {
    var a = vec2f(10.0, 20.0);
    var b = vec2f(30.0, 40.0);
    result = select(a, b, vec2<bool>(true, false));
}
"#,
    );
    assert!(matches!(
        globals["result"],
        Value::Primitive(Primitive::F32x2([30.0, 20.0]))
    ));
}
