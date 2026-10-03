use std::{cell::RefCell, rc::Rc};

use naga::{Expression, GlobalVariable, Handle, LocalVariable};

use crate::{
    error::EvaluatorError,
    function_state::StackFrame,
    place::{Place, PlaceRoot, PlaceSegment},
    value::Value,
};

use super::{InvocationState, evaluate_global_expression};

#[derive(Clone, Debug)]
pub(super) enum GlobalValue {
    Private(Value),
    Shared(Rc<RefCell<Value>>),
}

impl GlobalValue {
    pub(super) fn read_path(&self, path: &[PlaceSegment]) -> Value {
        match self {
            GlobalValue::Private(value) => value.at_path(path),
            GlobalValue::Shared(value) => value.borrow().at_path(path),
        }
    }

    fn write_path(&mut self, path: &[PlaceSegment], value: Value) -> Result<(), EvaluatorError> {
        match self {
            GlobalValue::Private(slot) => slot.assign_path(path, value),
            GlobalValue::Shared(slot) => slot.borrow_mut().assign_path(path, value),
        }
        .map_err(EvaluatorError::InternalError)
    }
}

impl InvocationState {
    pub(crate) fn read_place(&self, place: &Place) -> Value {
        match place.root() {
            PlaceRoot::Local {
                function_frame_index,
                handle,
            } => self.read_local_value(*function_frame_index, *handle, place.path()),
            PlaceRoot::Global { handle } => self
                .global_values
                .get(handle)
                .map(|value| value.read_path(place.path()))
                .unwrap_or(Value::Uninitialized),
        }
    }

    fn read_local_value(
        &self,
        function_frame_index: usize,
        handle: Handle<LocalVariable>,
        path: &[PlaceSegment],
    ) -> Value {
        let Some(StackFrame::Function(frame)) = self.stack.get(function_frame_index) else {
            return Value::Uninitialized;
        };
        if let Some(value) = frame.local(handle) {
            return value.at_path(path);
        }
        self.local_initial_value(function_frame_index, handle)
            .at_path(path)
    }

    fn local_initial_value(
        &self,
        function_frame_index: usize,
        handle: Handle<LocalVariable>,
    ) -> Value {
        let Some(StackFrame::Function(frame)) = self.stack.get(function_frame_index) else {
            return Value::Uninitialized;
        };
        let function = self.resolve_function(&frame.function_id());
        let local = &function.local_variables[handle];
        match local.init {
            Some(expr) => self.eval_value(expr, function_frame_index),
            None => Value::zero(self.program.module(), local.ty),
        }
    }

    pub(crate) fn write_place(
        &mut self,
        place: &Place,
        value: Value,
    ) -> Result<(), EvaluatorError> {
        match place.root() {
            PlaceRoot::Local {
                function_frame_index,
                handle,
            } => {
                self.ensure_local_variable_value(*handle, *function_frame_index)?;
                let StackFrame::Function(frame) = &mut self.stack[*function_frame_index] else {
                    return Err(EvaluatorError::InternalError(
                        "expected function frame".into(),
                    ));
                };
                let slot = frame.local_mut(*handle).ok_or_else(|| {
                    EvaluatorError::InternalError(format!(
                        "local variable {handle:?} was not initialized"
                    ))
                })?;
                slot.assign_path(place.path(), value)
                    .map_err(EvaluatorError::InternalError)
            }
            PlaceRoot::Global { handle } => {
                self.ensure_global_variable_value(*handle)?;
                let slot = self.global_values.get_mut(handle).ok_or_else(|| {
                    EvaluatorError::InternalError(format!(
                        "global variable {handle:?} was not initialized"
                    ))
                })?;
                slot.write_path(place.path(), value)
            }
        }
    }

    pub(crate) fn set_current_expression_value(
        &mut self,
        handle: Handle<Expression>,
        value: Value,
    ) -> Result<(), EvaluatorError> {
        self.current_function_frame_mut()?
            .set_expression(handle, value.into());
        Ok(())
    }

    /// Index of the `Function` frame below `function_index` — the caller's frame, used for `CallResult`.
    pub(super) fn parent_function_frame_index(&self, function_index: usize) -> Option<usize> {
        self.stack[..function_index]
            .iter()
            .rposition(|sf| matches!(sf, StackFrame::Function(_)))
    }

    pub(super) fn ensure_local_variable_value(
        &mut self,
        handle: Handle<LocalVariable>,
        func_idx: usize,
    ) -> Result<(), EvaluatorError> {
        let (init, ty) = {
            let StackFrame::Function(frame) = &self.stack[func_idx] else {
                return Err(EvaluatorError::InternalError(
                    "expected function frame".into(),
                ));
            };
            if frame.local(handle).is_some() {
                return Ok(());
            }
            let function = self.resolve_function(&frame.function_id());
            let local = &function.local_variables[handle];
            (local.init, local.ty)
        };

        let value = match init {
            Some(expr) => self.eval_value(expr, func_idx),
            None => Value::zero(self.program.module(), ty),
        };

        let StackFrame::Function(frame) = &mut self.stack[func_idx] else {
            return Err(EvaluatorError::InternalError(
                "expected function frame".into(),
            ));
        };
        frame.set_local(handle, value);
        Ok(())
    }

    pub(super) fn ensure_global_variable_value(
        &mut self,
        handle: Handle<GlobalVariable>,
    ) -> Result<(), EvaluatorError> {
        if self.global_values.contains_key(&handle) {
            return Ok(());
        }

        let global = &self.program.module().global_variables[handle];
        let value = match global.init {
            Some(expr) => evaluate_global_expression(self.program.module(), expr),
            None => Value::zero(self.program.module(), global.ty),
        };
        self.global_values
            .insert(handle, GlobalValue::Private(value));
        Ok(())
    }
}
