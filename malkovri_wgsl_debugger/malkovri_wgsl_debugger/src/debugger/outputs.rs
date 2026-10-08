use naga::{Binding, TypeInner, common::wgsl::TryToWgsl};

use crate::{EvaluatorError, Value};

use super::{DebugThreadId, Debugger, SourceLocation};

/// One declared shader output. Its value is available only after the entry point returns.
#[derive(Debug, Clone)]
pub struct ShaderOutput {
    /// WGSL interface annotation, such as `@builtin(position)` or `@location(0)`.
    pub name: String,
    /// Machine-readable binding, including interpolation and sampling qualifiers.
    pub binding: Binding,
    pub value: Option<Value>,
}

impl Debugger {
    /// Inspect output declarations and returned values without evaluating shader expressions.
    pub fn thread_shader_outputs(
        &self,
        thread_id: DebugThreadId,
    ) -> Result<Vec<ShaderOutput>, EvaluatorError> {
        let invocation = self.invocation_for_thread(thread_id)?;
        let module = self.program.module();
        let function = &module.entry_points[self.entry_point_index].function;
        let Some(result) = &function.result else {
            return Ok(Vec::new());
        };
        let value = invocation.entry_point_output();
        if let Some(binding) = &result.binding {
            return Ok(vec![ShaderOutput {
                name: output_name(binding),
                binding: binding.clone(),
                value: value.cloned(),
            }]);
        }
        let TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
            return Ok(Vec::new());
        };
        Ok(members
            .iter()
            .enumerate()
            .filter_map(|(index, member)| {
                member.binding.as_ref().map(|binding| ShaderOutput {
                    name: output_name(binding),
                    binding: binding.clone(),
                    value: value.map(|value| value.index_into(index)),
                })
            })
            .collect())
    }

    /// Source of the executed entry-point return, retained after the live stack is gone.
    pub fn thread_entry_point_return_location(
        &self,
        thread_id: DebugThreadId,
    ) -> Result<Option<SourceLocation>, EvaluatorError> {
        let invocation = self.invocation_for_thread(thread_id)?;
        let Some(statement) = invocation.entry_point_return() else {
            return Ok(None);
        };
        let Some((_, span)) = self
            .program
            .block(statement.block())
            .span_iter()
            .nth(statement.index())
        else {
            return Ok(None);
        };
        let location = span.location(self.source());
        Ok(Some(SourceLocation {
            line: location.line_number,
            column: location.line_position,
            function_name: Some(
                self.program.module().entry_points[self.entry_point_index]
                    .name
                    .clone(),
            ),
        }))
    }
}

fn output_name(binding: &Binding) -> String {
    match binding {
        Binding::BuiltIn(builtin) => format!("@builtin({})", builtin.to_wgsl_for_diagnostics()),
        Binding::Location { location, .. } => format!("@location({location})"),
    }
}

impl Debugger {
    /// Completed vertex outputs suitable for triangle interpolation.
    pub fn vertex_outputs(&self) -> Result<Vec<crate::graphics::VertexOutput>, String> {
        if self.program.module().entry_points[self.entry_point_index].stage
            != naga::ShaderStage::Vertex
        {
            return Err("expected a vertex debugger".into());
        }
        self.threads()
            .iter()
            .map(|thread| {
                if self.thread_state(thread.id).map_err(|e| e.to_string())?
                    != super::ThreadState::Finished
                {
                    return Err(format!("vertex thread {} has not returned", thread.id));
                }
                let mut position = None;
                let mut locations = std::collections::BTreeMap::new();
                for output in self
                    .thread_shader_outputs(thread.id)
                    .map_err(|e| e.to_string())?
                {
                    let value = output.value.ok_or_else(|| {
                        format!("vertex thread {} missing {}", thread.id, output.name)
                    })?;
                    match output.binding {
                        Binding::BuiltIn(naga::BuiltIn::Position { .. }) => {
                            let Value::Primitive(crate::Primitive::F32x4(v)) = value else {
                                return Err("vertex position must be vec4f".into());
                            };
                            position = Some(v);
                        }
                        Binding::Location { location, .. } => {
                            locations.insert(location, value);
                        }
                        _ => return Err("unsupported vertex builtin output".into()),
                    }
                }
                Ok(crate::graphics::VertexOutput {
                    position: position.ok_or("missing vertex position")?,
                    locations,
                    vertex_index: thread.global_invocation_id[0],
                    instance_index: thread.global_invocation_id[1],
                })
            })
            .collect()
    }
}
