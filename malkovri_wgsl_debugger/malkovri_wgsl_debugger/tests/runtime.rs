use std::collections::HashMap;

use malkovri_wgsl_debugger::{
    Debugger, GlobalConstants, Primitive, ShaderProgram, StepResult, Value, WorkgroupConfig,
};

fn debugger(source: &str, entry: usize) -> Debugger {
    ShaderProgram::new(source)
        .unwrap()
        .create_debugger(
            entry,
            WorkgroupConfig::default(),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap()
}

fn finish(debugger: &mut Debugger) -> u32 {
    for _ in 0..1_000 {
        if debugger.step_all().unwrap() == StepResult::Finished {
            return debugger
                .global_variables()
                .into_iter()
                .find_map(
                    |variable| match (variable.name.as_deref(), &variable.value) {
                        (Some("result"), Value::Primitive(Primitive::U32(value))) => Some(*value),
                        _ => None,
                    },
                )
                .unwrap();
        }
    }
    panic!("shader did not finish within 1,000 steps");
}

#[test]
fn nested_loops_switches_and_early_returns_preserve_control_flow() {
    let mut debugger = debugger(
        r#"
var<private> result: u32;
fn helper(value: u32) -> u32 {
    if value == 2u { return 10u; }
    return value;
}
@compute @workgroup_size(1) fn main() {
    for (var outer = 0u; outer < 3u; outer++) {
        var inner = 0u;
        loop {
            inner++;
            switch inner {
                case 1u: { continue; }
                case 2u, 3u: { result += helper(outer); }
                default: { break; }
            }
            if inner == 4u { break; }
        }
    }
}
"#,
        0,
    );
    assert_eq!(finish(&mut debugger), 22);
}

#[test]
fn continuing_blocks_run_after_continue_and_can_break_the_loop() {
    let mut debugger = debugger(
        r#"
var<private> result: u32;
@compute @workgroup_size(1) fn main() {
    var i = 0u;
    loop {
        if i < 2u { continue; }
        result += i;
        continuing {
            i++;
            break if i == 4u;
        }
    }
}
"#,
        0,
    );
    assert_eq!(finish(&mut debugger), 5);
}

#[test]
fn pointer_arguments_keep_aliasing_through_nested_calls() {
    let mut debugger = debugger(
        r#"
struct Item { values: array<array<u32, 2>, 2>, untouched: u32 }
var<private> result: u32;
fn update(first: ptr<function, u32>, second: ptr<function, u32>) {
    *first += 2u;
    *second += 3u;
}
fn forward(first: ptr<function, u32>, second: ptr<function, u32>) {
    if *first == 10u { update(first, second); }
}
@compute @workgroup_size(1) fn main() {
    var items: array<Item, 2>;
    items[1].values[0] = array<u32, 2>(10u, 20u);
    items[1].untouched = 7u;
    forward(&items[1].values[0][0], &items[1].values[0][0]);
    result = items[1].values[0][0] + items[1].values[0][1] + items[1].untouched;
}
"#,
        0,
    );
    assert_eq!(finish(&mut debugger), 42);
}

#[test]
fn vertex_triangle_returns_the_first_position_with_default_inputs() {
    let program =
        ShaderProgram::new(include_str!("../../test_shaders/test_vertex_triangle.wgsl")).unwrap();
    let entry = program.entry_points().next().unwrap();
    assert_eq!(entry.name, "vs_main");
    assert_eq!(entry.stage, malkovri_wgsl_debugger::ShaderStage::Vertex);

    // Explicit vertex inputs are not configurable yet; the default index is zero.
    let mut debugger = program
        .create_debugger(
            entry.index,
            WorkgroupConfig::default(),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap();
    assert_eq!(debugger.entry_point_output(), None);

    for _ in 0..1_000 {
        if debugger.step_all().unwrap() == StepResult::Finished {
            assert_eq!(
                debugger.entry_point_output(),
                Some(Primitive::F32x4([0.0, 0.5, 0.0, 1.0]).into())
            );
            return;
        }
    }
    panic!("vertex shader did not finish within 1,000 steps");
}

#[test]
fn vertex_and_fragment_entry_points_can_execute_in_separate_sessions() {
    let source = r#"
var<private> result: u32;
fn helper(value: u32) -> u32 {
    if value == 0u { return 7u; }
    return 9u;
}
@vertex fn vertex_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    result = helper(index);
    return vec4f(0.0, 0.0, 0.0, 1.0);
}
@fragment fn fragment_main() -> @location(0) vec4f {
    result = helper(1u);
    return vec4f(1.0);
}
"#;
    let program = ShaderProgram::new(source).unwrap();
    let make_session = |entry| {
        program
            .create_debugger(
                entry,
                WorkgroupConfig::default(),
                GlobalConstants::default(),
                HashMap::new(),
            )
            .unwrap()
    };
    let mut vertex = make_session(0);
    assert_eq!(finish(&mut vertex), 7);
    assert!(matches!(
        vertex.entry_point_output(),
        Some(Value::Primitive(Primitive::F32x4([0.0, 0.0, 0.0, 1.0])))
    ));
    // A later stage starts from the same program with fresh invocation memory.
    let mut fragment = make_session(1);
    assert_eq!(finish(&mut fragment), 9);
    assert!(matches!(
        fragment.entry_point_output(),
        Some(Value::Primitive(Primitive::F32x4([1.0, 1.0, 1.0, 1.0])))
    ));
    assert_eq!(finish(&mut vertex), 7);
}

#[test]
fn shared_program_sessions_keep_their_memory_independent() {
    let source =
        "var<private> result: u32; @compute @workgroup_size(1) fn main() { result += 1u; }";
    let program = ShaderProgram::new(source).unwrap();
    let make_session = || {
        program
            .create_debugger(
                0,
                WorkgroupConfig::default(),
                GlobalConstants::default(),
                HashMap::new(),
            )
            .unwrap()
    };
    let mut first = make_session();
    let mut second = make_session();
    assert_eq!(program.entry_points().next().unwrap().name, "main");
    assert_eq!(
        program.entry_points().next().unwrap().stage,
        malkovri_wgsl_debugger::ShaderStage::Compute
    );
    assert_eq!(program.source(), source);
    drop(program);
    // Sessions retain immutable program data even after the caller releases it.
    assert_eq!(finish(&mut first), 1);
    assert_eq!(finish(&mut second), 1);
}
