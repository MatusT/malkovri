use std::collections::HashSet;

use naga::{Handle, LocalVariable, Statement};

use crate::{error::EvaluatorError, value::Value};

use super::Evaluator;

impl Evaluator {
    /// Return all global variables with their names and current values.
    pub(crate) fn global_variable_values(&self) -> Vec<(Option<String>, Value)> {
        self.global_values
            .iter()
            .map(|(handle, value)| {
                let name = self.module.global_variables[*handle].name.clone();
                (name, value.read())
            })
            .collect()
    }

    /// Evaluate the current function arguments in declaration order.
    pub(crate) fn current_function_argument_values(
        &self,
    ) -> Result<Vec<(Option<String>, Value)>, EvaluatorError> {
        let func_idx = self.current_function_frame_index()?;
        let frame = self.current_function_frame()?;
        let function = self.resolve_function(&frame.function_ref);
        Ok(function
            .arguments
            .iter()
            .enumerate()
            .map(|(index, argument)| {
                (
                    argument.name.clone(),
                    self.evaluate_function_argument(index, func_idx),
                )
            })
            .collect())
    }

    pub(crate) fn local_variables_in_current_scope(
        &self,
    ) -> Result<HashSet<Handle<LocalVariable>>, EvaluatorError> {
        let frame = self.current_function_frame()?;
        let function = self.resolve_function(&frame.function_ref);
        let declaring_scopes = self
            .declaring_scopes
            .local_scopes(&frame.function_ref)
            .ok_or_else(|| {
                EvaluatorError::InternalError("missing local declaring scopes".into())
            })?;

        let current_block = self.current_frame()?;
        // The span of the current (innermost) execution scope.
        let current_scope = self.current_scope_range()?;

        // Current execution position for the "declared before" check.
        let current_pos = current_block
            .statements()
            .span_iter()
            .nth(current_block.current_statement_index())
            .and_then(|(_, sp)| sp.to_range())
            .map(|r| r.start);

        Ok(function
            .local_variables
            .iter()
            .filter_map(|(handle, _)| {
                let var_range = function.local_variables.get_span(handle).to_range()?;
                let var_end = var_range.end;

                // Skip variables whose declaration we haven't stepped past.
                if current_pos.is_some_and(|pos| pos < var_end) {
                    return None;
                }

                let declaring_scope = declaring_scopes.get(&handle)?;

                // Variable is visible iff the current scope is nested inside the declaring scope.
                (declaring_scope.contains(&current_scope.start)
                    && current_scope.end <= declaring_scope.end)
                    .then_some(handle)
            })
            .collect())
    }

    /// Evaluate all named expressions (WGSL `let` bindings) in the current function frame
    /// that are in scope at the current execution point.
    /// Returns `(name, value)` pairs in source order.
    pub(crate) fn named_expression_values(&self) -> Result<Vec<(String, Value)>, EvaluatorError> {
        let function_index = self.current_function_frame_index()?;
        let frame = self.current_function_frame()?;
        let function = self.current_function()?;
        let declaring_scopes = self
            .declaring_scopes
            .named_expression_scopes(&frame.function_ref)
            .ok_or_else(|| {
                EvaluatorError::InternalError("missing named expression scopes".into())
            })?;

        let current_scope = self.current_scope_range()?;

        let mut emitted = HashSet::new();
        for frame_idx in function_index..self.stack.len() {
            let frame = &self.stack[frame_idx];
            let limit = frame.current_statement_index();
            for (i, (stmt, _)) in frame.statements().span_iter().enumerate() {
                if i > limit {
                    break;
                }
                if let Statement::Emit(range) = stmt {
                    emitted.extend(range.clone());
                }
            }
        }

        Ok(function
            .named_expressions
            .iter()
            .filter(|(handle, _)| {
                if !emitted.contains(handle) {
                    return false;
                }
                let Some(declaring_scope) = declaring_scopes.get(handle) else {
                    return false;
                };
                declaring_scope.start <= current_scope.start
                    && current_scope.end <= declaring_scope.end
            })
            .map(|(handle, name)| (name.clone(), self.evaluate_expression(*handle)))
            .collect())
    }
}
