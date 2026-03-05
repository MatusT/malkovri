use crate::{
    function_state::{
        BlockFrame, BlockKind, ControlFlow, FunctionFrame, NextStatement, StackFrame,
    },
    primitive::Primitive,
    value::Value,
};

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use naga::{Expression, GlobalVariable, Handle, LocalVariable, Module, Statement};

#[derive(Copy, Clone, Debug, Default)]
pub struct EntryPointInputs {
    pub global_invocation_id: [u32; 3],
}

pub struct Evaluator {
    pub(crate) module: Module,
    #[allow(dead_code)]
    pub(crate) entry_point: naga::ir::EntryPoint,
    pub(crate) entry_point_inputs: EntryPointInputs,
    pub(crate) global_values: HashMap<naga::Handle<GlobalVariable>, Value>,
    pub(crate) stack: Vec<StackFrame>,
}

impl Evaluator {
    pub fn new(
        module: &Module,
        entry_point_index: usize,
        entry_point_inputs: EntryPointInputs,
        global_values: HashMap<naga::ResourceBinding, Value>,
    ) -> Self {
        let entry_fn = module.entry_points[entry_point_index].function.clone();
        let statements = entry_fn.body.clone();

        let mut evaluator = Evaluator {
            module: module.clone(),
            entry_point: module.entry_points[entry_point_index].clone(),
            entry_point_inputs,
            global_values: global_values
                .iter()
                .map(|(k, v)| {
                    (
                        module
                            .global_variables
                            .fetch_if(|h| h.binding.unwrap() == *k)
                            .unwrap(),
                        v.clone(),
                    )
                })
                .collect(),
            stack: vec![StackFrame::Function(Box::new(FunctionFrame {
                function: entry_fn,
                function_handle: None,
                local_variables: HashMap::new(),
                evaluated_expressions: HashMap::new(),
                evaluated_function_arguments: Vec::new(),
                statements,
                current_statement_index: 0,
                call_result_handle: None,
                control_flow: ControlFlow::None,
            }))],
        };
        evaluator.initialize_local_variables();
        evaluator
    }

    /// Return a reference to the current function frame (the nearest `Function` variant on the
    /// stack), or `None` if the stack is empty.
    pub fn current_function_frame(&self) -> Option<&FunctionFrame> {
        let function_index = self.current_function_frame_index()?;
        match &self.stack[function_index] {
            StackFrame::Function(f) => Some(f),
            StackFrame::Block(_) => None,
        }
    }

    /// Return the statements and current index of the top-of-stack frame.
    /// Unlike `current_function_frame`, this reflects the innermost active block,
    /// which may be a nested `if`/`loop`/`switch` block rather than the function body.
    pub fn current_active_block(&self) -> Option<(&naga::Block, usize)> {
        let top = self.stack.last()?;
        Some((top.statements(), top.current_statement_index()))
    }

    /// Index of the topmost `Function` frame, used to look up expressions and variables.
    pub(crate) fn current_function_frame_index(&self) -> Option<usize> {
        self.stack
            .iter()
            .rposition(|sf| matches!(sf, StackFrame::Function(_)))
    }

    /// Index of the `Function` frame below `function_index` — the caller's frame, used for `CallResult`.
    fn parent_function_frame_index(&self, function_index: usize) -> Option<usize> {
        self.stack[..function_index]
            .iter()
            .rposition(|sf| matches!(sf, StackFrame::Function(_)))
    }

    /// Evaluate all named expressions (WGSL `let` bindings) in the current function frame.
    /// Returns `(name, value)` pairs in source order.
    pub fn named_expression_values(&self) -> Vec<(String, Value)> {
        let func_idx = match self.current_function_frame_index() {
            Some(i) => i,
            None => return vec![],
        };
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return vec![];
        };
        frame
            .function
            .named_expressions
            .iter()
            .map(|(handle, name)| (name.clone(), self.evaluate_expression(*handle)))
            .collect()
    }

    // -------------------------------------------------------------------------
    // Core execution loop
    // -------------------------------------------------------------------------

    /// Advance past any pending control-flow signals and exhausted frames until
    /// a live statement is ready to execute (or the stack is empty).
    fn advance_to_live_statement(&mut self) -> bool {
        loop {
            if self.stack.is_empty() {
                return false;
            }

            // Phase 1: Resolve pending control-flow signals.
            if self.resolve_control_flow_signal() {
                continue;
            }

            // Phase 2: Pop exhausted frames.
            if self.pop_if_exhausted() {
                continue;
            }

            return true;
        }
    }

    /// If the current function frame has a pending control-flow signal, apply it
    /// and return `true`. Otherwise return `false`.
    fn resolve_control_flow_signal(&mut self) -> bool {
        let function_index = match self.current_function_frame_index() {
            Some(i) => i,
            None => return false,
        };
        let StackFrame::Function(ref mut frame) = self.stack[function_index] else {
            unreachable!("current_function_frame_index always returns a Function frame");
        };
        let signal = std::mem::take(&mut frame.control_flow);
        match signal {
            ControlFlow::None => false,
            ControlFlow::Break => {
                self.apply_break(function_index);
                true
            }
            ControlFlow::Continue => {
                self.apply_continue(function_index);
                true
            }
            ControlFlow::Return(return_val) => {
                self.apply_return(function_index, return_val);
                true
            }
        }
    }

    /// If the top frame is exhausted, handle it and return `true`.
    fn pop_if_exhausted(&mut self) -> bool {
        let top = self.stack.len() - 1;
        if self.stack[top].is_exhausted() {
            self.handle_exhausted_frame(top);
            true
        } else {
            false
        }
    }

    /// Execute the current statement and advance the program counter.
    /// Returns the *upcoming* statement that will execute on the next call,
    /// or `None` if execution has finished.
    pub fn step(&mut self) -> Option<NextStatement> {
        if !self.advance_to_live_statement() {
            return None;
        }

        let top = self.stack.len() - 1;
        let stmt_idx = self.stack[top].current_statement_index();
        let stmt = self.stack[top].statements()[stmt_idx].clone();
        let function_index = self.current_function_frame_index().unwrap();

        self.handle_statement(stmt, function_index);
        self.stack[top].increment_statement_index();

        // Resolve any signals/exhaustion produced by the statement we just ran,
        // then return whatever is live next (or None if execution finished).
        self.advance_to_live_statement();
        self.peek_next_statement()
    }

    fn peek_next_statement(&self) -> Option<NextStatement> {
        let function_index = self.current_function_frame_index()?;
        let StackFrame::Function(ref frame) = self.stack[function_index] else {
            return None;
        };
        let top = self.stack.len() - 1;
        let stmt_idx = self.stack[top].current_statement_index();
        let stmt = self.stack[top].statements().get(stmt_idx)?.clone();
        Some(NextStatement {
            function: frame.function.clone(),
            statement: stmt,
            statement_index: stmt_idx,
        })
    }

    // -------------------------------------------------------------------------
    // Control-flow signal handlers
    // -------------------------------------------------------------------------

    /// Pop block frames above `function_index` until (and including) the nearest `Loop` or `Switch` frame.
    fn apply_break(&mut self, function_index: usize) {
        while self.stack.len() > function_index + 1 {
            let top = self.stack.len() - 1;
            let is_target = matches!(
                &self.stack[top],
                StackFrame::Block(BlockFrame {
                    kind: BlockKind::Loop { .. } | BlockKind::Switch,
                    ..
                })
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
            let top = self.stack.len() - 1;
            if matches!(
                &self.stack[top],
                StackFrame::Block(BlockFrame {
                    kind: BlockKind::Loop { .. },
                    ..
                })
            ) {
                break;
            }
            self.stack.pop();
        }

        // Switch the loop frame to its continuing block.
        let top = self.stack.len() - 1;
        if top > function_index {
            if let StackFrame::Block(ref mut block_frame) = self.stack[top] {
                block_frame.switch_to_continuing();
            }
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

    // -------------------------------------------------------------------------
    // Exhausted-frame handler
    // -------------------------------------------------------------------------

    fn handle_exhausted_frame(&mut self, top: usize) {
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

    // -------------------------------------------------------------------------
    // Statement dispatch
    // -------------------------------------------------------------------------

    fn handle_statement(&mut self, statement: Statement, function_index: usize) {
        match statement {
            Statement::Emit(_) => {
                // Expressions are evaluated lazily on demand.
            }
            Statement::Call {
                function: function_handle,
                arguments,
                result,
            } => {
                self.handle_call(function_handle, arguments, result);
            }
            Statement::Store { pointer, value } => {
                self.handle_store(pointer, value);
            }
            Statement::Return { value } => {
                let return_value = value.map(|v| self.evaluate_expression(v));
                if let StackFrame::Function(ref mut frame) = self.stack[function_index] {
                    frame.control_flow = ControlFlow::Return(return_value);
                }
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
                    statements: body.clone(),
                    current_statement_index: 0,
                    kind: BlockKind::Loop {
                        body,
                        continuing,
                        break_if,
                        in_continuing: false,
                    },
                }));
            }
            Statement::Switch { selector, cases } => {
                self.handle_switch(selector, cases);
            }
            Statement::Break => {
                if let StackFrame::Function(ref mut frame) = self.stack[function_index] {
                    frame.control_flow = ControlFlow::Break;
                }
            }
            Statement::Continue => {
                if let StackFrame::Function(ref mut frame) = self.stack[function_index] {
                    frame.control_flow = ControlFlow::Continue;
                }
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
    }

    fn handle_call(
        &mut self,
        function_handle: naga::Handle<naga::Function>,
        arguments: Vec<Handle<Expression>>,
        call_result_handle: Option<Handle<Expression>>,
    ) {
        let evaluated_function_arguments = arguments
            .iter()
            .map(|&arg| self.evaluate_expression(arg))
            .collect();

        let callee = self.module.functions[function_handle].clone();
        let statements = callee.body.clone();

        self.stack
            .push(StackFrame::Function(Box::new(FunctionFrame {
                function: callee,
                function_handle: Some(function_handle),
                local_variables: HashMap::new(),
                evaluated_expressions: HashMap::new(),
                evaluated_function_arguments,
                statements,
                current_statement_index: 0,
                call_result_handle,
                control_flow: ControlFlow::None,
            })));

        self.initialize_local_variables();
    }

    fn initialize_local_variables(&mut self) {
        let func_idx = match self.current_function_frame_index() {
            Some(i) => i,
            None => return,
        };

        let mut insert_variables: Vec<(Handle<LocalVariable>, Value)> = Vec::new();

        let local_vars: Vec<_> = match &self.stack[func_idx] {
            StackFrame::Function(frame) => frame.function.local_variables.iter().collect(),
            StackFrame::Block(_) => return,
        };

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

        for (handle, value) in insert_variables {
            if let StackFrame::Function(ref mut frame) = self.stack[func_idx] {
                frame
                    .local_variables
                    .insert(handle, Value::Pointer(Rc::new(RefCell::new(value))));
            }
        }
    }

    fn handle_store(&mut self, pointer: Handle<Expression>, value: Handle<Expression>) {
        let evaluated_value = self.evaluate_expression(value);
        let evaluated_pointer = self.evaluate_expression(pointer);

        match evaluated_pointer {
            Value::Pointer(inner) => {
                *inner.borrow_mut() = evaluated_value;
            }
            _ => {
                eprintln!("Store to non-pointer value: {:?}", evaluated_pointer);
            }
        }
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

    fn handle_switch(&mut self, selector: Handle<Expression>, cases: Vec<naga::SwitchCase>) {
        let selector_val = self.evaluate_expression(selector);

        let selector_i32 = match selector_val {
            Value::Primitive(Primitive::I32(v)) => v,
            Value::Primitive(Primitive::U32(v)) => v as i32,
            _ => return,
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
    }
}
