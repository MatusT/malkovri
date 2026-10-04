use std::collections::HashMap;

use malkovri_wgsl_debugger::{
    Debugger, DrawConfig, ExecutionConfig, GlobalConstants, Primitive, ShaderProgram, ShaderStage,
    StepResult, Value, WorkgroupConfig,
};

fn debugger(source: &str, entry: usize) -> Debugger {
    let program = ShaderProgram::new(source).unwrap();
    let config = match program.entry_points().nth(entry).unwrap().stage {
        ShaderStage::Compute => WorkgroupConfig::default().into(),
        ShaderStage::Vertex => DrawConfig::default().into(),
        ShaderStage::Fragment => ExecutionConfig::Fragment,
        _ => panic!("unsupported stage"),
    };
    program
        .create_debugger(entry, config, GlobalConstants::default(), HashMap::new())
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

    let mut debugger = program
        .create_debugger(
            entry.index,
            DrawConfig::default(),
            GlobalConstants::default(),
            HashMap::new(),
        )
        .unwrap();
    assert_eq!(debugger.entry_point_output(), None);
    let outputs = debugger.thread_shader_outputs(1).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].name, "@builtin(position)");
    assert_eq!(outputs[0].value, None);
    assert!(
        debugger
            .thread_entry_point_return_location(1)
            .unwrap()
            .is_none()
    );

    for _ in 0..1_000 {
        if debugger.step_all().unwrap() == StepResult::Finished {
            assert_eq!(
                debugger.entry_point_output(),
                Some(Primitive::F32x4([0.0, 0.5, 0.0, 1.0]).into())
            );
            assert_eq!(
                debugger.thread_shader_outputs(1).unwrap()[0].value,
                debugger.entry_point_output()
            );
            let location = debugger
                .thread_entry_point_return_location(1)
                .unwrap()
                .unwrap();
            assert_eq!(location.function_name.as_deref(), Some("vs_main"));
            assert!(
                debugger
                    .source()
                    .lines()
                    .nth(location.line as usize - 1)
                    .unwrap()
                    .contains("return vec4f")
            );
            assert!(debugger.call_stack().is_empty());
            return;
        }
    }
    panic!("vertex shader did not finish within 1,000 steps");
}

#[test]
fn shader_outputs_map_struct_members_and_retain_the_executed_entry_return() {
    let source = r#"
struct Output {
    @location(0) color: vec3f,
    @builtin(position) position: vec4f,
}
fn helper() -> Output {
    return Output(vec3f(1.0, 0.0, 0.5), vec4f(0.0, 0.5, 0.0, 1.0));
}
@vertex fn main(@builtin(vertex_index) index: u32) -> Output {
    if index == 0u {
        return helper();
    }
    return Output(vec3f(0.0), vec4f(0.0));
}
"#;
    let mut debugger = debugger(source, 0);
    let outputs = debugger.thread_shader_outputs(1).unwrap();
    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(|output| output.value.is_none()));
    assert_eq!(
        debugger.run_to_breakpoint(1, false, &[], None).unwrap(),
        malkovri_wgsl_debugger::RunResult::Finished
    );
    let outputs = debugger.thread_shader_outputs(1).unwrap();
    assert_eq!(outputs[0].name, "@location(0)");
    assert_eq!(
        outputs[0].value,
        Some(Primitive::F32x3([1.0, 0.0, 0.5]).into())
    );
    assert_eq!(outputs[1].name, "@builtin(position)");
    assert_eq!(
        outputs[1].value,
        Some(Primitive::F32x4([0.0, 0.5, 0.0, 1.0]).into())
    );
    let location = debugger
        .thread_entry_point_return_location(1)
        .unwrap()
        .unwrap();
    assert_eq!(
        source
            .lines()
            .nth(location.line as usize - 1)
            .unwrap()
            .trim(),
        "return helper();"
    );
    assert!(debugger.thread_shader_outputs(99).is_err());
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
    let make_session = |entry, config: ExecutionConfig| {
        program
            .create_debugger(entry, config, GlobalConstants::default(), HashMap::new())
            .unwrap()
    };
    let mut vertex = make_session(0, DrawConfig::default().into());
    assert_eq!(finish(&mut vertex), 7);
    assert!(matches!(
        vertex.entry_point_output(),
        Some(Value::Primitive(Primitive::F32x4([0.0, 0.0, 0.0, 1.0])))
    ));
    // A later stage starts from the same program with fresh invocation memory.
    let mut fragment = make_session(1, ExecutionConfig::Fragment);
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

#[test]
fn execution_configuration_must_match_the_selected_shader_stage() {
    let program =
        ShaderProgram::new("@vertex fn main() -> @builtin(position) vec4f { return vec4f(0.0); }")
            .unwrap();
    for config in [
        WorkgroupConfig::default().into(),
        ExecutionConfig::Fragment,
        DrawConfig {
            vertex_count: 0,
            ..Default::default()
        }
        .into(),
    ] {
        assert!(matches!(
            program.create_debugger(0, config, GlobalConstants::default(), HashMap::new()),
            Err(malkovri_wgsl_debugger::DebuggerError::InvalidConfig(_))
        ));
    }
    let program = ShaderProgram::new("@compute @workgroup_size(1) fn main() {}").unwrap();
    assert!(matches!(
        program.create_debugger(
            0,
            DrawConfig::default(),
            GlobalConstants::default(),
            HashMap::new()
        ),
        Err(malkovri_wgsl_debugger::DebuggerError::InvalidConfig(_))
    ));
}
