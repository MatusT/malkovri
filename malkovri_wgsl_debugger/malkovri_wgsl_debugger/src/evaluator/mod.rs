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
    debugger::WorkgroupConfig,
    declaring_scopes,
    entry_point_inputs::{
        ComputeThreadInputs, FragmentThreadInputs, GlobalConstants, VertexThreadInputs,
    },
    error::EvaluatorError,
    function_state::{ControlFlow, FunctionFrame, FunctionRef, StackFrame},
    thread::EvaluatorThread,
    value::Value,
};

use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use naga::{GlobalVariable, Handle, Module};

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

pub(crate) struct Evaluator {
    pub(crate) module: Arc<Module>,
    pub(crate) global_constants: GlobalConstants,
    pub(crate) global_values: HashMap<naga::Handle<GlobalVariable>, GlobalValue>,
    pub(crate) entry_point_output: Option<Value>,
    pub(crate) stack: Vec<StackFrame>,
    pub(crate) threads: HashMap<[u32; 3], EvaluatorThread>,
    /// Global invocation ID of the currently active thread.
    active_thread_gid: [u32; 3],
    declaring_scopes: declaring_scopes::ModuleScopes,
}

impl Evaluator {
    pub(crate) fn new(
        module: Arc<Module>,
        entry_point_index: usize,
        global_constants: GlobalConstants,
        global_values: HashMap<naga::ResourceBinding, Rc<RefCell<Value>>>,
        shared_global_values: HashMap<Handle<GlobalVariable>, Rc<RefCell<Value>>>,
        workgroup_config: WorkgroupConfig,
    ) -> Result<Self, EvaluatorError> {
        let statements = module.entry_points[entry_point_index].function.body.clone();

        let declaring_scopes = declaring_scopes::ModuleScopes::new(&module);

        let mut threads = HashMap::new();
        let mut first_gid = [0u32; 3];
        for x in 0..workgroup_config.workgroup_size[0] {
            for y in 0..workgroup_config.workgroup_size[1] {
                for z in 0..workgroup_config.workgroup_size[2] {
                    let compute_inputs = ComputeThreadInputs::new(
                        [x, y, z],
                        workgroup_config.workgroup_size,
                        workgroup_config.workgroup_id,
                        workgroup_config.subgroup_size,
                    );

                    let gid = compute_inputs.global_invocation_id;
                    if x == 0 && y == 0 && z == 0 {
                        first_gid = gid;
                    }
                    let thread = EvaluatorThread {
                        compute_inputs,
                        vertex_inputs: VertexThreadInputs::default(),
                        fragment_inputs: FragmentThreadInputs::default(),
                    };
                    threads.insert(gid, thread);
                }
            }
        }

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
                    Some(expr) => evaluate_global_expression(&module, expr),
                    None => Value::zero(&module, global.ty),
                };
                global_values.insert(handle, GlobalValue::Private(value));
            }
        }

        let evaluator = Evaluator {
            global_values,
            module,
            global_constants,
            entry_point_output: None,
            stack: vec![StackFrame::Function(Box::new(FunctionFrame {
                function_ref: FunctionRef::EntryPoint(entry_point_index),
                local_variables: HashMap::new(),
                evaluated_expressions: HashMap::new(),
                evaluated_function_arguments: Vec::new(),
                statements,
                current_statement_index: 0,
                call_result_handle: None,
                control_flow: ControlFlow::None,
            }))],
            declaring_scopes,
            threads,
            active_thread_gid: first_gid,
        };

        Ok(evaluator)
    }

    /// Return a reference to the currently active thread.
    pub(crate) fn active_thread(&self) -> &EvaluatorThread {
        &self.threads[&self.active_thread_gid]
    }

    pub(crate) fn set_active_thread_gid(&mut self, gid: [u32; 3]) -> Result<(), EvaluatorError> {
        if !self.threads.contains_key(&gid) {
            return Err(EvaluatorError::InternalError(format!(
                "unknown thread gid {gid:?}"
            )));
        }
        self.active_thread_gid = gid;
        Ok(())
    }

    /// Resolve a [`FunctionRef`] to the actual `naga::Function` in the module.
    pub(crate) fn resolve_function(&self, fref: &FunctionRef) -> &naga::Function {
        match fref {
            FunctionRef::EntryPoint(idx) => &self.module.entry_points[*idx].function,
            FunctionRef::Called(handle) => &self.module.functions[*handle],
        }
    }

    /// Return a reference to the `naga::Function` for the current call frame.
    pub(crate) fn current_function(&self) -> Result<&naga::Function, EvaluatorError> {
        let frame = self.current_function_frame()?;
        Ok(self.resolve_function(&frame.function_ref))
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

    fn current_scope_range(&self) -> Result<std::ops::Range<usize>, EvaluatorError> {
        let current_frame = self.current_frame()?;
        Ok(naga::Span::total_span(
            current_frame
                .statements()
                .span_iter()
                .map(|(_, span)| *span),
        )
        .to_range()
        .unwrap_or(0..usize::MAX))
    }

    /// Return the statements and current index of the top-of-stack frame.
    /// Unlike `current_function_frame`, this reflects the innermost active block,
    /// which may be a nested `if`/`loop`/`switch` block rather than the function body.
    pub(crate) fn current_active_block(&self) -> Result<(&naga::Block, usize), EvaluatorError> {
        let top = self.current_frame()?;
        Ok((top.statements(), top.current_statement_index()))
    }
}
