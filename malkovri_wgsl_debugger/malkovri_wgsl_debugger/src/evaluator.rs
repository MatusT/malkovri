use crate::{
    debugger::WorkgroupConfig,
    declaring_scopes,
    entry_point_inputs::EntryPointInputs,
    error::EvaluatorError,
    eval_expressions::evaluate_global_expression,
    function_state::{
        BlockFrame, BlockKind, ControlFlow, FunctionFrame, FunctionRef, NextStatement, StackFrame,
    },
    primitive::Primitive,
    thread::{EvaluatorThread, ThreadIdentification},
    value::Value,
};

use std::{cell::RefCell, collections::HashMap, collections::HashSet, rc::Rc, sync::Arc};

use naga::{AddressSpace, Expression, GlobalVariable, Handle, LocalVariable, Module, Statement};

pub(crate) struct Evaluator {
    pub(crate) module: Arc<Module>,
    pub(crate) entry_point_inputs: EntryPointInputs,
    pub(crate) global_values: HashMap<naga::Handle<GlobalVariable>, Value>,
    pub(crate) stack: Vec<StackFrame>,
    threads: HashMap<[u32; 3], EvaluatorThread>,
    declaring_scopes: declaring_scopes::ModuleScopes,
}

impl Evaluator {
    pub(crate) fn new(
        module: Arc<Module>,
        entry_point_index: usize,
        entry_point_inputs: EntryPointInputs,
        global_values: HashMap<naga::ResourceBinding, Value>,
        workgroup_config: WorkgroupConfig,
    ) -> Result<Self, EvaluatorError> {
        let statements = module.entry_points[entry_point_index].function.body.clone();

        let declaring_scopes = declaring_scopes::ModuleScopes::new(&module);

        let mut threads = HashMap::new();
        for x in 0..workgroup_config.size[0] {
            for y in 0..workgroup_config.size[1] {
                for z in 0..workgroup_config.size[2] {
                    let thread_id = ThreadIdentification::new(
                        [x, y, z],
                        workgroup_config.size,
                        workgroup_config.id,
                        workgroup_config.subgroup_size,
                    );

                    let gid = thread_id.global_invocation_id();
                    let thread = EvaluatorThread {
                        id: thread_id,
                        stack: vec![],
                        private_globals: module
                            .global_variables
                            .iter()
                            .filter(|(_, v)| v.space == AddressSpace::Private)
                            .map(|(h, v)| {
                                let value: Value = match v.init {
                                    Some(init_expr) => {
                                        evaluate_global_expression(&module, init_expr)
                                    }
                                    None => Value::from(&module.types[v.ty].inner),
                                };
                                (h, value)
                            })
                            .collect(),
                        status: crate::thread::ThreadStatus::Running,
                    };
                    threads.insert(gid, thread);
                }
            }
        }

        let mut evaluator = Evaluator {
            global_values: global_values
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
                    Ok((handle, v.clone()))
                })
                .collect::<Result<_, EvaluatorError>>()?,
            module,
            entry_point_inputs,
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
        };
        evaluator.initialize_local_variables()?;
        Ok(evaluator)
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

    /// Return all global variables with their names and current values.
    pub(crate) fn global_variable_values(&self) -> Vec<(Option<String>, Value)> {
        self.global_values
            .iter()
            .map(|(handle, value)| {
                let name = self.module.global_variables[*handle].name.clone();
                (name, value.clone())
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

    /// Index of the `Function` frame below `function_index` — the caller's frame, used for `CallResult`.
    fn parent_function_frame_index(&self, function_index: usize) -> Option<usize> {
        self.stack[..function_index]
            .iter()
            .rposition(|sf| matches!(sf, StackFrame::Function(_)))
    }

    /// Returns local variable handles that are lexically in scope at the
    /// current execution point: declared before the current position and
    /// owned by a block that contains the current scope.
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

// Core execution loop
impl Evaluator {
    /// Advance past any pending control-flow signals and exhausted frames until
    /// a live statement is ready to execute (or the stack is empty).
    fn advance_to_live_statement(&mut self) -> Result<bool, EvaluatorError> {
        loop {
            if self.stack.is_empty() {
                return Ok(false);
            }

            // Phase 1: Resolve pending control-flow signals.
            if self.resolve_control_flow_signal()? {
                continue;
            }

            // Phase 2: Pop exhausted frames.
            if self.pop_if_exhausted()? {
                continue;
            }

            return Ok(true);
        }
    }

    /// If the current function frame has a pending control-flow signal, apply it
    /// and return `true`. Otherwise return `false`.
    fn resolve_control_flow_signal(&mut self) -> Result<bool, EvaluatorError> {
        let function_index = self.current_function_frame_index()?;
        let StackFrame::Function(ref mut frame) = self.stack[function_index] else {
            return Err(EvaluatorError::InternalError(
                "expected function frame at function_index".into(),
            ));
        };
        let signal = std::mem::take(&mut frame.control_flow);
        match signal {
            ControlFlow::None => Ok(false),
            ControlFlow::Break => {
                self.apply_break(function_index);
                Ok(true)
            }
            ControlFlow::Continue => {
                self.apply_continue(function_index);
                Ok(true)
            }
            ControlFlow::Return(return_val) => {
                self.apply_return(function_index, return_val);
                Ok(true)
            }
        }
    }

    /// If the top frame is exhausted, handle it and return `true`.
    fn pop_if_exhausted(&mut self) -> Result<bool, EvaluatorError> {
        let is_exhausted = self.current_frame()?.is_exhausted();
        if is_exhausted {
            self.handle_exhausted_frame();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Execute the current statement and advance the program counter.
    /// Returns the *upcoming* statement that will execute on the next call,
    /// or `None` if execution has finished.
    pub(crate) fn step(&mut self) -> Result<Option<NextStatement>, EvaluatorError> {
        if !self.advance_to_live_statement()? {
            return Ok(None);
        }

        let caller_frame_index = self.current_frame_index()?;

        {
            let top = self.current_frame()?;
            let current_statement_index = top.current_statement_index();
            let current_statement = top
                .statements()
                .get(current_statement_index)
                .cloned()
                .ok_or_else(|| EvaluatorError::InternalError("invalid statement index".into()))?;

            self.handle_statement(current_statement)?;
        }

        self.stack[caller_frame_index].increment_statement_index();

        // Resolve any signals/exhaustion produced by the statement we just ran,
        // then return whatever is live next (or None if execution finished).
        self.advance_to_live_statement()?;

        Ok(self.peek_next_statement())
    }

    fn peek_next_statement(&self) -> Option<NextStatement> {
        let current_block = self.current_frame().ok()?;
        let current_statement_index = current_block.current_statement_index();
        let statement = current_block
            .statements()
            .get(current_statement_index)?
            .clone();

        Some(NextStatement { statement })
    }
}

// Control-flow signal handlers
impl Evaluator {
    /// Pop block frames above `function_index` until (and including) the nearest `Loop` or `Switch` frame.
    fn apply_break(&mut self, function_index: usize) {
        while self.stack.len() > function_index + 1 {
            let is_target = matches!(
                self.current_frame().ok(),
                Some(StackFrame::Block(BlockFrame {
                    kind: BlockKind::Loop { .. } | BlockKind::Switch,
                    ..
                }))
            );
            self.stack.pop();
            if is_target {
                break;
            }
        }
    }

    /// Pop block frames above `function_index` until the nearest `Loop` frame (keep it), then switch
    /// it to its `continuing` block.
    fn apply_continue(&mut self, function_index: usize) {
        while self.stack.len() > function_index + 1 {
            if matches!(
                self.current_frame().ok(),
                Some(StackFrame::Block(BlockFrame {
                    kind: BlockKind::Loop { .. },
                    ..
                }))
            ) {
                break;
            }
            self.stack.pop();
        }

        // Switch the loop frame to its continuing block.
        if let Ok(top) = self.current_frame_index()
            && top > function_index
            && let StackFrame::Block(ref mut block_frame) = self.stack[top]
        {
            block_frame.switch_to_continuing();
        }
    }

    /// Store the return value in the parent function frame, then truncate the stack
    /// to remove the returning function and everything above it.
    fn apply_return(&mut self, function_index: usize, value: Option<Value>) {
        // Read the result handle from the callee before truncating the stack.
        let call_result_handle = match &self.stack[function_index] {
            StackFrame::Function(frame) => frame.call_result_handle,
            StackFrame::Block(_) => None,
        };
        // Store the return value in the parent frame's expression cache, keyed
        // by the CallResult expression handle so that each call's result is
        // independently retrievable even when multiple calls appear in sequence.
        if let (Some(handle), Some(return_val)) = (call_result_handle, value)
            && let Some(parent_function_index) = self.parent_function_frame_index(function_index)
            && let StackFrame::Function(ref mut parent_frame) = self.stack[parent_function_index]
        {
            parent_frame
                .evaluated_expressions
                .insert(handle, return_val);
        }
        self.stack.truncate(function_index);
    }
}

// Exhausted-frame handler
impl Evaluator {
    fn handle_exhausted_frame(&mut self) {
        let Ok(top) = self.current_frame_index() else {
            return;
        };
        match &self.stack[top] {
            StackFrame::Function(_) => {
                self.stack.pop();
            }
            StackFrame::Block(block_frame) => match &block_frame.kind {
                BlockKind::Plain | BlockKind::Switch => {
                    self.stack.pop();
                }
                BlockKind::Loop {
                    in_continuing: false,
                    ..
                } => {
                    if let StackFrame::Block(ref mut bf) = self.stack[top] {
                        bf.switch_to_continuing();
                    }
                }
                BlockKind::Loop {
                    in_continuing: true,
                    break_if,
                    ..
                } => {
                    let should_break = if let Some(expr) = break_if {
                        self.evaluate_expression(*expr).is_truthy()
                    } else {
                        false
                    };

                    if should_break {
                        self.stack.pop();
                    } else if let StackFrame::Block(ref mut bf) = self.stack[top] {
                        bf.restart_body();
                    }
                }
            },
        }
    }
}

// Statement dispatch
impl Evaluator {
    fn handle_statement(&mut self, statement: Statement) -> Result<(), EvaluatorError> {
        match statement {
            Statement::Emit(_) => {}
            Statement::Call {
                function: function_handle,
                arguments,
                result,
            } => {
                self.handle_call(function_handle, arguments, result)?;
            }
            Statement::Store { pointer, value } => {
                self.handle_store(pointer, value)?;
            }
            Statement::Return { value } => {
                let return_value = value.map(|v| self.evaluate_expression(v));
                self.current_function_frame_mut()?.control_flow = ControlFlow::Return(return_value);
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                self.handle_if(condition, accept, reject);
            }
            Statement::Block(block) => {
                self.stack.push(StackFrame::Block(BlockFrame {
                    statements: block,
                    current_statement_index: 0,
                    kind: BlockKind::Plain,
                }));
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                self.stack.push(StackFrame::Block(BlockFrame {
                    statements: body,
                    current_statement_index: 0,
                    kind: BlockKind::Loop {
                        other_block: continuing,
                        break_if,
                        in_continuing: false,
                    },
                }));
            }
            Statement::Switch { selector, cases } => {
                self.handle_switch(selector, cases)?;
            }
            Statement::Break => {
                self.current_function_frame_mut()?.control_flow = ControlFlow::Break;
            }
            Statement::Continue => {
                self.current_function_frame_mut()?.control_flow = ControlFlow::Continue;
            }
            Statement::Kill
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_)
            | Statement::ImageStore { .. }
            | Statement::Atomic { .. }
            | Statement::ImageAtomic { .. }
            | Statement::RayQuery { .. }
            | Statement::WorkGroupUniformLoad { .. }
            | Statement::SubgroupBallot { .. }
            | Statement::SubgroupGather { .. }
            | Statement::SubgroupCollectiveOperation { .. } => {}
        }
        Ok(())
    }

    fn handle_call(
        &mut self,
        function_handle: naga::Handle<naga::Function>,
        arguments: Vec<Handle<Expression>>,
        call_result_handle: Option<Handle<Expression>>,
    ) -> Result<(), EvaluatorError> {
        let evaluated_function_arguments = arguments
            .iter()
            .map(|&arg| self.evaluate_expression(arg))
            .collect();

        let statements = self.module.functions[function_handle].body.clone();

        self.stack
            .push(StackFrame::Function(Box::new(FunctionFrame {
                function_ref: FunctionRef::Called(function_handle),
                local_variables: HashMap::new(),
                evaluated_expressions: HashMap::new(),
                evaluated_function_arguments,
                statements,
                current_statement_index: 0,
                call_result_handle,
                control_flow: ControlFlow::None,
            })));

        self.initialize_local_variables()
    }

    fn initialize_local_variables(&mut self) -> Result<(), EvaluatorError> {
        let mut insert_variables: Vec<(Handle<LocalVariable>, Value)> = Vec::new();
        {
            let function = self.current_function()?;
            let local_vars: Vec<_> = function.local_variables.iter().collect();
            for (handle, local_var) in local_vars {
                let value = match &local_var.init {
                    Some(init_expr) => self.evaluate_expression(*init_expr),
                    None => {
                        let ty = &self.module.types[local_var.ty];
                        Value::from(&ty.inner)
                    }
                };
                insert_variables.push((handle, value));
            }
        }

        let frame = self.current_function_frame_mut()?;
        for (handle, value) in insert_variables {
            frame
                .local_variables
                .insert(handle, Value::Pointer(Rc::new(RefCell::new(value))));
        }
        Ok(())
    }

    fn handle_if(
        &mut self,
        condition: Handle<Expression>,
        accept: naga::Block,
        reject: naga::Block,
    ) {
        let condition_result = self.evaluate_expression(condition);

        let branch = if condition_result.is_truthy() {
            accept
        } else {
            reject
        };

        if !branch.is_empty() {
            self.stack.push(StackFrame::Block(BlockFrame {
                statements: branch,
                current_statement_index: 0,
                kind: BlockKind::Plain,
            }));
        }
    }

    fn handle_switch(
        &mut self,
        selector: Handle<Expression>,
        cases: Vec<naga::SwitchCase>,
    ) -> Result<(), EvaluatorError> {
        let selector_val = self.evaluate_expression(selector);

        let selector_i32 = match selector_val {
            Value::Primitive(Primitive::I32(v)) => v,
            Value::Primitive(Primitive::U32(v)) => v as i32,
            _ => {
                return Err(EvaluatorError::InternalError(format!(
                    "switch selector is not an integer: {:?}",
                    selector_val
                )));
            }
        };

        // Find the matching case, fall back to Default.
        let body = cases
            .iter()
            .find(|c| matches!(&c.value, naga::SwitchValue::I32(v) if *v == selector_i32))
            .or_else(|| {
                cases.iter().find(
                    |c| matches!(&c.value, naga::SwitchValue::U32(v) if *v == selector_i32 as u32),
                )
            })
            .or_else(|| {
                cases
                    .iter()
                    .find(|c| matches!(&c.value, naga::SwitchValue::Default))
            })
            .map(|c| c.body.clone());

        if let Some(body) = body
            && !body.is_empty()
        {
            self.stack.push(StackFrame::Block(BlockFrame {
                statements: body,
                current_statement_index: 0,
                kind: BlockKind::Switch,
            }));
        }
        Ok(())
    }

    fn handle_store(
        &mut self,
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        let evaluated_value = self.evaluate_expression(value);
        let evaluated_pointer = self.evaluate_expression(pointer);

        match evaluated_pointer {
            Value::Pointer(inner) => {
                *inner.borrow_mut() = evaluated_value;
                Ok(())
            }
            _ => Err(EvaluatorError::StoreToNonPointer),
        }
    }
}
