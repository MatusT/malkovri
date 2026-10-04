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
