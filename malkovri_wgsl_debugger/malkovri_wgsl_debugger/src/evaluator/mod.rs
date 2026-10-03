use crate::program::FunctionId;
mod binary;
mod cast;
mod expression;
mod math;
mod scopes;
mod statement;
mod step;
mod storage;

pub(crate) use expression::evaluate_global_expression;

use crate::{
    entry_point_inputs::{GlobalConstants, InvocationInputs},
    error::EvaluatorError,
    function_state::{ControlFlow, FrameContext, FunctionFrame, StackFrame},
    program::ShaderProgram,
    value::Value,
};

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use naga::{GlobalVariable, Handle};

#[derive(Clone, Debug)]
pub(crate) enum GlobalValue {
    Private(Value),
    Shared(Rc<RefCell<Value>>),
}

impl GlobalValue {
    fn read(&self) -> Value {
        match self {
            GlobalValue::Private(value) => value.clone(),
            GlobalValue::Shared(value) => value.borrow().clone(),
        }
    }

    fn write_path(
        &mut self,
        path: &[crate::place::PlaceSegment],
        value: Value,
    ) -> Result<(), EvaluatorError> {
        match self {
            GlobalValue::Private(slot) => slot.assign_path(path, value),
            GlobalValue::Shared(slot) => slot.borrow_mut().assign_path(path, value),
        }
        .map_err(EvaluatorError::InternalError)
    }
}

pub(crate) struct InvocationState {
    pub(crate) program: ShaderProgram,
    pub(crate) global_constants: GlobalConstants,
    pub(crate) global_values: HashMap<naga::Handle<GlobalVariable>, GlobalValue>,
    pub(crate) entry_point_output: Option<Value>,
    pub(crate) stack: Vec<StackFrame>,
    pub(crate) inputs: InvocationInputs,
}

impl InvocationState {
    pub(crate) fn new(
        program: ShaderProgram,
        entry_point_index: usize,
        global_constants: GlobalConstants,
        global_values: HashMap<naga::ResourceBinding, Rc<RefCell<Value>>>,
        shared_global_values: HashMap<Handle<GlobalVariable>, Rc<RefCell<Value>>>,
        inputs: InvocationInputs,
    ) -> Result<Self, EvaluatorError> {
        let module = program.module();
        let block = program.function_body(FunctionId::EntryPoint(entry_point_index));

        let mut global_values: HashMap<_, _> = global_values
            .iter()
            .map(|(k, v)| {
                let handle = module
                    .global_variables
                    .fetch_if(|h| h.binding.as_ref() == Some(k))
                    .ok_or_else(|| {
                        EvaluatorError::InternalError(format!(
                            "no global variable with binding {:?}",
                            k
                        ))
                    })?;
                Ok((handle, GlobalValue::Shared(v.clone())))
            })
            .collect::<Result<_, EvaluatorError>>()?;

        for (handle, global) in module.global_variables.iter() {
            if global_values.contains_key(&handle) || global.binding.is_some() {
                continue;
            }
            if let Some(shared) = shared_global_values.get(&handle) {
                global_values.insert(handle, GlobalValue::Shared(shared.clone()));
            } else {
                let value = match global.init {
                    Some(expr) => evaluate_global_expression(module, expr),
                    None => Value::zero(module, global.ty),
                };
                global_values.insert(handle, GlobalValue::Private(value));
            }
        }

        let evaluator = InvocationState {
            global_values,
            program,
            global_constants,
            entry_point_output: None,
            stack: vec![StackFrame::Function(Box::new(FunctionFrame {
                function_id: FunctionId::EntryPoint(entry_point_index),
                local_variables: HashMap::new(),
                evaluated_expressions: HashMap::new(),
                evaluated_function_arguments: Vec::new(),
                block,
                current_statement_index: 0,
                call_result_handle: None,
                control_flow: ControlFlow::None,
            }))],
            inputs,
        };

        Ok(evaluator)
    }

    /// Resolve a [`FunctionId`] to the actual `naga::Function` in the module.
    pub(crate) fn resolve_function(&self, fref: &FunctionId) -> &naga::Function {
        self.program.function(*fref)
    }

    /// Return a reference to the `naga::Function` for the current call frame.
    pub(crate) fn current_function(&self) -> Result<&naga::Function, EvaluatorError> {
        let frame = self.current_function_frame()?;
        Ok(self.resolve_function(&frame.function_id))
    }

    /// Index of the topmost `Function` frame, used to look up expressions and variables.
    pub(crate) fn current_function_frame_index(&self) -> Result<usize, EvaluatorError> {
        self.stack
            .iter()
            .rposition(|sf| matches!(sf, StackFrame::Function(_)))
            .ok_or_else(|| EvaluatorError::InternalError("no function frame on stack".into()))
    }

    /// Return a reference to the current function frame (the nearest `Function` variant on the
    /// stack).
    pub(crate) fn current_function_frame(&self) -> Result<&FunctionFrame, EvaluatorError> {
        let function_index = self.current_function_frame_index()?;
        match &self.stack[function_index] {
            StackFrame::Function(f) => Ok(f),
            _ => Err(EvaluatorError::InternalError(
                "expected function frame".into(),
            )),
        }
    }

    /// Return a mutable reference to the current function frame (the nearest `Function` variant on the
    /// stack).
    pub(crate) fn current_function_frame_mut(
        &mut self,
    ) -> Result<&mut FunctionFrame, EvaluatorError> {
        let function_index = self.current_function_frame_index()?;
        match &mut self.stack[function_index] {
            StackFrame::Function(f) => Ok(f),
            _ => Err(EvaluatorError::InternalError(
                "expected function frame".into(),
            )),
        }
    }

    /// Return a reference to the topmost stack frame (function or block).
    fn current_frame(&self) -> Result<&StackFrame, EvaluatorError> {
        self.stack
            .last()
            .ok_or_else(|| EvaluatorError::InternalError("stack is empty".into()))
    }

    /// Index of the topmost stack frame.
    fn current_frame_index(&self) -> Result<usize, EvaluatorError> {
        if self.stack.is_empty() {
            return Err(EvaluatorError::InternalError("stack is empty".into()));
        }
        Ok(self.stack.len() - 1)
    }

    pub(crate) fn frame_context(
        &self,
        function_index: usize,
    ) -> Result<FrameContext, EvaluatorError> {
        if !matches!(
            self.stack.get(function_index),
            Some(StackFrame::Function(_))
        ) {
            return Err(EvaluatorError::InternalError(
                "unknown function frame".into(),
            ));
        }
        let callee_index = self
            .stack
            .iter()
            .enumerate()
            .skip(function_index + 1)
            .find_map(|(index, frame)| matches!(frame, StackFrame::Function(_)).then_some(index));
        let block_index = callee_index.unwrap_or(self.stack.len()) - 1;
        // The caller's program counter points past the suspended call.
        let statement_index = self.stack[block_index]
            .current_statement_index()
            .saturating_sub(usize::from(callee_index.is_some()));
        Ok(FrameContext {
            function_index,
            block_index,
            statement_index,
        })
    }

    fn scope_range(&self, context: FrameContext) -> std::ops::Range<usize> {
        naga::Span::total_span(
            self.program
                .block(self.stack[context.block_index].block())
                .span_iter()
                .map(|(_, span)| *span),
        )
        .to_range()
        .unwrap_or(0..usize::MAX)
    }
}
