use crate::{
    function_state::{
        BlockFrame, BlockKind, ControlFlow, FunctionFrame, NextStatement, StackFrame,
    },
    primitive::Primitive,
    value::Value,
};

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use naga::{
    Expression, GlobalVariable, Handle, Literal, LocalVariable, MathFunction, Module,
    RelationalFunction, Statement, SwizzleComponent, Type, TypeInner, UnaryOperator, VectorSize,
};

#[derive(Copy, Clone, Debug, Default)]
pub struct EntryPointInputs {
    pub global_invocation_id: [u32; 3],
}

pub struct Evaluator {
    module: Module,
    #[allow(dead_code)]
    entry_point: naga::ir::EntryPoint,
    entry_point_inputs: EntryPointInputs,
    global_values: HashMap<naga::Handle<GlobalVariable>, Value>,
    stack: Vec<StackFrame>,
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
        evaluator.initialize_local_variables(0, None);
        evaluator
    }

    /// Return a reference to the current function frame (the nearest `Function` variant on the
    /// stack), or `None` if the stack is empty.
    pub fn current_function_frame(&self) -> Option<&FunctionFrame> {
        let func_idx = self.current_function_frame_index()?;
        match &self.stack[func_idx] {
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
    fn current_function_frame_index(&self) -> Option<usize> {
        self.stack
            .iter()
            .rposition(|sf| matches!(sf, StackFrame::Function(_)))
    }

    /// Index of the `Function` frame below `func_idx` — the caller's frame, used for `CallResult`.
    fn parent_function_frame_index(&self, func_idx: usize) -> Option<usize> {
        self.stack[..func_idx]
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
        let parent_func_idx = self.parent_function_frame_index(func_idx);
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return vec![];
        };
        frame
            .function
            .named_expressions
            .iter()
            .map(|(handle, name)| {
                (
                    name.clone(),
                    self.evaluate_expression(*handle, func_idx, parent_func_idx),
                )
            })
            .collect()
    }

    // -------------------------------------------------------------------------
    // Core execution loop
    // -------------------------------------------------------------------------

    /// Advance past any pending control-flow signals and exhausted frames until
    /// a live statement is ready to execute (or the stack is empty).
    /// Returns `true` if a live statement exists at the top of the stack.
    fn advance_to_live_statement(&mut self) -> bool {
        loop {
            if self.stack.is_empty() {
                return false;
            }

            // Handle any control-flow signal on the current function frame.
            let func_idx = match self.current_function_frame_index() {
                Some(i) => i,
                None => return false,
            };
            let signal = match &self.stack[func_idx] {
                StackFrame::Function(frame) => frame.control_flow.clone(),
                StackFrame::Block(_) => ControlFlow::None,
            };
            match signal {
                ControlFlow::None => {}
                ControlFlow::Break => {
                    self.apply_break(func_idx);
                    continue;
                }
                ControlFlow::Continue => {
                    self.apply_continue(func_idx);
                    continue;
                }
                ControlFlow::Return(return_val) => {
                    self.apply_return(func_idx, return_val);
                    continue;
                }
            }

            // Pop exhausted frames.
            let top = self.stack.len() - 1;
            if self.stack[top].is_exhausted() {
                self.handle_exhausted_frame(top);
                continue;
            }

            return true;
        }
    }

    pub fn next_statement(&mut self) -> Option<NextStatement> {
        if !self.advance_to_live_statement() {
            return None;
        }

        let top = self.stack.len() - 1;
        let stmt_idx = self.stack[top].current_statement_index();
        let stmt = self.stack[top].statements()[stmt_idx].clone();
        let func_idx = self.current_function_frame_index().unwrap();
        let parent_func_idx = self.parent_function_frame_index(func_idx);

        self.handle_statement(stmt, func_idx, parent_func_idx);
        self.stack[top].increment_statement_index();

        // Resolve any signals/exhaustion produced by the statement we just ran,
        // then return whatever is live next (or None if execution finished).
        self.advance_to_live_statement();
        self.peek_next_statement()
    }

    fn peek_next_statement(&self) -> Option<NextStatement> {
        let func_idx = self.current_function_frame_index()?;
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
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

    /// Pop block frames above `func_idx` until (and including) the nearest `Loop` or `Switch` frame.
    fn apply_break(&mut self, func_idx: usize) {
        while self.stack.len() > func_idx + 1 {
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
        if let StackFrame::Function(ref mut frame) = self.stack[func_idx] {
            frame.control_flow = ControlFlow::None;
        }
    }

    /// Pop block frames above `func_idx` until the nearest `Loop` frame (keep it), then switch
    /// it to its `continuing` block.
    fn apply_continue(&mut self, func_idx: usize) {
        while self.stack.len() > func_idx + 1 {
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
        if top > func_idx {
            let cont = match &self.stack[top] {
                StackFrame::Block(block_frame) => match &block_frame.kind {
                    BlockKind::Loop { continuing, .. } => Some(continuing.clone()),
                    _ => None,
                },
                _ => None,
            };
            if let (Some(cont), StackFrame::Block(block_frame)) = (cont, &mut self.stack[top])
                && let BlockKind::Loop {
                    ref mut in_continuing,
                    ..
                } = block_frame.kind
                {
                    block_frame.statements = cont;
                    block_frame.current_statement_index = 0;
                    *in_continuing = true;
                }
        }

        if let StackFrame::Function(ref mut frame) = self.stack[func_idx] {
            frame.control_flow = ControlFlow::None;
        }
    }

    /// Store the return value in the parent function frame, then truncate the stack
    /// to remove the returning function and everything above it.
    fn apply_return(&mut self, func_idx: usize, value: Option<Value>) {
        // Read the result handle from the callee before truncating the stack.
        let call_result_handle = match &self.stack[func_idx] {
            StackFrame::Function(frame) => frame.call_result_handle,
            StackFrame::Block(_) => None,
        };
        // Store the return value in the parent frame's expression cache, keyed
        // by the CallResult expression handle so that each call's result is
        // independently retrievable even when multiple calls appear in sequence.
        if let (Some(handle), Some(return_val)) = (call_result_handle, value)
            && let Some(parent_func_idx) = self.parent_function_frame_index(func_idx)
            && let StackFrame::Function(ref mut parent_frame) = self.stack[parent_func_idx]
        {
            parent_frame.evaluated_expressions.insert(handle, return_val);
        }
        self.stack.truncate(func_idx);
    }

    // -------------------------------------------------------------------------
    // Exhausted-frame handler
    // -------------------------------------------------------------------------

    fn handle_exhausted_frame(&mut self, top: usize) {
        enum Action {
            Pop,
            SwitchToContinuing(naga::Block),
            CheckBreakIf {
                break_if: Option<Handle<Expression>>,
                body: naga::Block,
            },
        }

        let action = match &self.stack[top] {
            StackFrame::Function(_) => Action::Pop,
            StackFrame::Block(block_frame) => match &block_frame.kind {
                BlockKind::Plain | BlockKind::Switch => Action::Pop,
                BlockKind::Loop {
                    in_continuing: false,
                    continuing,
                    ..
                } => Action::SwitchToContinuing(continuing.clone()),
                BlockKind::Loop {
                    in_continuing: true,
                    break_if,
                    body,
                    ..
                } => Action::CheckBreakIf {
                    break_if: *break_if,
                    body: body.clone(),
                },
            },
        };

        match action {
            Action::Pop => {
                self.stack.pop();
            }
            Action::SwitchToContinuing(cont) => {
                if let StackFrame::Block(ref mut block_frame) = self.stack[top]
                    && let BlockKind::Loop {
                        ref mut in_continuing,
                        ..
                    } = block_frame.kind
                    {
                        block_frame.statements = cont;
                        block_frame.current_statement_index = 0;
                        *in_continuing = true;
                    }
            }
            Action::CheckBreakIf { break_if, body } => {
                let should_break = if let Some(expr) = break_if {
                    let func_idx = self.current_function_frame_index().unwrap();
                    let parent_func_idx = self.parent_function_frame_index(func_idx);
                    let v = self.evaluate_expression(expr, func_idx, parent_func_idx);
                    matches!(v, Value::Primitive(Primitive::U32(v)) if v != 0)
                } else {
                    false // infinite loop — repeat body
                };

                if should_break {
                    self.stack.pop();
                } else if let StackFrame::Block(ref mut block_frame) = self.stack[top]
                    && let BlockKind::Loop {
                        ref mut in_continuing,
                        ..
                    } = block_frame.kind
                    {
                        block_frame.statements = body;
                        block_frame.current_statement_index = 0;
                        *in_continuing = false;
                    }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Statement dispatch
    // -------------------------------------------------------------------------

    fn handle_statement(
        &mut self,
        statement: Statement,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) {
        match statement {
            Statement::Emit(_) => {
                // Expressions are evaluated lazily on demand.
            }
            Statement::Call {
                function: function_handle,
                arguments,
                result,
            } => {
                self.handle_call(function_handle, arguments, result, func_idx, parent_func_idx);
            }
            Statement::Store { pointer, value } => {
                self.handle_store(pointer, value, func_idx, parent_func_idx);
            }
            Statement::Return { value } => {
                let return_value =
                    value.map(|v| self.evaluate_expression(v, func_idx, parent_func_idx));
                if let StackFrame::Function(ref mut frame) = self.stack[func_idx] {
                    frame.control_flow = ControlFlow::Return(return_value);
                }
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                self.handle_if(condition, accept, reject, func_idx, parent_func_idx);
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
                self.handle_switch(selector, cases, func_idx, parent_func_idx);
            }
            Statement::Break => {
                if let StackFrame::Function(ref mut frame) = self.stack[func_idx] {
                    frame.control_flow = ControlFlow::Break;
                }
            }
            Statement::Continue => {
                if let StackFrame::Function(ref mut frame) = self.stack[func_idx] {
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
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) {
        let evaluated_function_arguments = arguments
            .iter()
            .map(|&arg| self.evaluate_expression(arg, func_idx, parent_func_idx))
            .collect();

        let callee = self.module.functions[function_handle].clone();
        let statements = callee.body.clone();

        self.stack.push(StackFrame::Function(Box::new(FunctionFrame {
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

        let new_func_idx = self.stack.len() - 1;
        self.initialize_local_variables(new_func_idx, parent_func_idx);
    }

    fn initialize_local_variables(&mut self, func_idx: usize, parent_func_idx: Option<usize>) {
        let mut insert_variables: Vec<(Handle<LocalVariable>, Value)> = Vec::new();

        let local_vars: Vec<_> = match &self.stack[func_idx] {
            StackFrame::Function(frame) => frame.function.local_variables.iter().collect(),
            StackFrame::Block(_) => return,
        };

        for (handle, local_var) in local_vars {
            let value = match &local_var.init {
                Some(init_expr) => self.evaluate_expression(*init_expr, func_idx, parent_func_idx),
                None => {
                    let ty = &self.module.types[local_var.ty];
                    Value::from(&ty.inner)
                }
            };
            insert_variables.push((handle, value));
        }

        for (handle, value) in insert_variables {
            if let StackFrame::Function(ref mut frame) = self.stack[func_idx] {
                frame.local_variables
                    .insert(handle, Value::Pointer(Rc::new(RefCell::new(value))));
            }
        }
    }

    fn handle_store(
        &mut self,
        pointer: Handle<Expression>,
        value: Handle<Expression>,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) {
        let evaluated_value = self.evaluate_expression(value, func_idx, parent_func_idx);
        let evaluated_pointer = self.evaluate_expression(pointer, func_idx, parent_func_idx);

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
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) {
        let condition_result = self.evaluate_expression(condition, func_idx, parent_func_idx);
        let branch = if matches!(condition_result, Value::Primitive(Primitive::U32(v)) if v != 0) {
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
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) {
        let selector_val = self.evaluate_expression(selector, func_idx, parent_func_idx);

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
                cases
                    .iter()
                    .find(|c| matches!(&c.value, naga::SwitchValue::U32(v) if *v == selector_i32 as u32))
            })
            .or_else(|| {
                cases
                    .iter()
                    .find(|c| matches!(&c.value, naga::SwitchValue::Default))
            })
            .map(|c| c.body.clone());

        if let Some(body) = body
            && !body.is_empty() {
                self.stack.push(StackFrame::Block(BlockFrame {
                    statements: body,
                    current_statement_index: 0,
                    kind: BlockKind::Switch,
                }));
            }
    }

    // -------------------------------------------------------------------------
    // Expression evaluation
    // -------------------------------------------------------------------------

    pub fn evaluate_expression(
        &self,
        expression_handle: Handle<Expression>,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return Value::Uninitialized;
        };
        let expression = &frame.function.expressions[expression_handle];

        match expression {
            Expression::Literal(literal) => self.evaluate_literal(literal),
            Expression::Constant(handle) => {
                self.evaluate_global_expression(self.module.constants[*handle].init)
            }
            Expression::Override(handle) => match self.module.overrides[*handle].init {
                Some(init) => self.evaluate_global_expression(init),
                None => Value::Uninitialized,
            },
            Expression::ZeroValue(ty) => Value::from(&self.module.types[*ty].inner),
            Expression::Compose { ty, components } => {
                self.evaluate_compose(*ty, components, func_idx, parent_func_idx)
            }
            Expression::Splat { size, value } => {
                self.evaluate_splat(*size, *value, func_idx, parent_func_idx)
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => self.evaluate_swizzle(*size, *vector, *pattern, func_idx, parent_func_idx),
            Expression::Load { pointer } => self.evaluate_load(*pointer, func_idx, parent_func_idx),
            Expression::AccessIndex { base, index } => {
                self.evaluate_access_index(*base, *index, func_idx, parent_func_idx)
            }
            Expression::FunctionArgument(index) => {
                self.evaluate_function_argument(*index as usize, func_idx)
            }
            Expression::LocalVariable(handle) => {
                self.evaluate_local_variable(*handle, func_idx, parent_func_idx)
            }
            Expression::Binary { op, left, right } => {
                self.evaluate_binary(*op, *left, *right, func_idx, parent_func_idx)
            }
            Expression::Unary { op, expr } => {
                let val = self.evaluate_expression(*expr, func_idx, parent_func_idx).leaf_value();
                self.evaluate_unary(*op, val)
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => self.evaluate_select(*condition, *accept, *reject, func_idx, parent_func_idx),
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let val = self.evaluate_expression(*expr, func_idx, parent_func_idx);
                self.evaluate_as(val, *kind, *convert)
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self.evaluate_math(*fun, *arg, *arg1, *arg2, *arg3, func_idx, parent_func_idx),
            Expression::Relational { fun, argument } => {
                let val = self.evaluate_expression(*argument, func_idx, parent_func_idx).leaf_value();
                self.evaluate_relational(*fun, val)
            }
            Expression::ArrayLength(expr) => {
                let argument = self.evaluate_expression(*expr, func_idx, parent_func_idx).leaf_value();
                match argument {
                    Value::Array(elements) => Primitive::U32(elements.len() as u32).into(),
                    _ => panic!("ArrayLength applied to non-array value: {:?}", argument),
                }
            }
            Expression::GlobalVariable(handle) => self.evaluate_global_variable(*handle),
            Expression::Access { base, index } => {
                self.evaluate_access(*base, *index, func_idx, parent_func_idx)
            }
            Expression::CallResult(_) => {
                // The return value was stored in evaluated_expressions by apply_return,
                // keyed by expression_handle (the Handle<Expression> for this CallResult).
                frame
                    .evaluated_expressions
                    .get(&expression_handle)
                    .cloned()
                    .unwrap_or(Value::Uninitialized)
            }
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_literal(&self, literal: &Literal) -> Value {
        match literal {
            Literal::F32(v) => Primitive::F32(*v).into(),
            Literal::F64(v) => Primitive::F64(*v).into(),
            Literal::I32(v) => Primitive::I32(*v).into(),
            Literal::I64(v) => Primitive::I64(*v).into(),
            Literal::U32(v) => Primitive::U32(*v).into(),
            Literal::U64(v) => Primitive::U64(*v).into(),
            Literal::Bool(v) => Primitive::U32(if *v { 1 } else { 0 }).into(),
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_load(
        &self,
        pointer: Handle<Expression>,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        // Check expression cache first.
        let cached = match &self.stack[func_idx] {
            StackFrame::Function(frame) => frame.evaluated_expressions.get(&pointer).cloned(),
            StackFrame::Block(_) => None,
        };
        if let Some(value) = cached {
            return value;
        }
        match self.evaluate_expression(pointer, func_idx, parent_func_idx) {
            Value::Pointer(inner) => inner.borrow().clone(),
            // Access into a global storage array evaluates directly to a value
            // (not a pointer) because global_values stores plain Value::Array.
            // Return it as-is rather than falling through to Uninitialized.
            Value::Uninitialized => Value::Uninitialized,
            value => value,
        }
    }

    fn evaluate_access_index(
        &self,
        base: Handle<Expression>,
        index: u32,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        let value = self.evaluate_expression(base, func_idx, parent_func_idx).leaf_value();
        value.index_into(index as usize)
    }

    fn evaluate_function_argument(&self, index: usize, func_idx: usize) -> Value {
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return Value::Uninitialized;
        };
        let function_argument = &frame.function.arguments[index];

        if let Some(binding) = &function_argument.binding {
            match binding {
                naga::ir::Binding::BuiltIn(built_in) => match built_in {
                    naga::ir::BuiltIn::GlobalInvocationId => {
                        Primitive::U32x3(self.entry_point_inputs.global_invocation_id).into()
                    }
                    _ => Value::Uninitialized,
                },
                naga::ir::Binding::Location { .. } => Value::Uninitialized,
            }
        } else {
            frame.evaluated_function_arguments[index].clone()
        }
    }

    fn evaluate_local_variable(
        &self,
        handle: Handle<LocalVariable>,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        // Check if already initialized.
        let (cached, init_expr) = match &self.stack[func_idx] {
            StackFrame::Function(frame) => {
                let cached = frame.local_variables.get(&handle).cloned();
                let init = frame.function.local_variables[handle].init;
                (cached, init)
            }
            StackFrame::Block(_) => return Value::Uninitialized,
        };

        if let Some(value) = cached {
            return value;
        }

        Value::Pointer(Rc::new(RefCell::new(self.evaluate_expression(
            init_expr.unwrap(),
            func_idx,
            parent_func_idx,
        ))))
    }

    fn evaluate_binary(
        &self,
        op: naga::BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        use naga::BinaryOperator::*;

        let l = self.evaluate_expression(left, func_idx, parent_func_idx).leaf_value();
        let r = self.evaluate_expression(right, func_idx, parent_func_idx).leaf_value();

        match op {
            // Arithmetic — Value trait impls handle all scalars, vectors, and scalar×vector
            Add => l + r,
            Subtract => l - r,
            Multiply => l * r,
            Divide => l / r,
            Modulo => l % r,

            // Bitwise — Value trait impls handle I32/U32 scalars and vectors
            And => l & r,
            InclusiveOr => l | r,
            ExclusiveOr => l ^ r,

            // Logical (booleans as U32 0/1)
            LogicalAnd => match (l, r) {
                (Value::Primitive(Primitive::U32(l)), Value::Primitive(Primitive::U32(r))) => {
                    Primitive::U32(u32::from(l != 0 && r != 0)).into()
                }
                _ => Value::Uninitialized,
            },
            LogicalOr => match (l, r) {
                (Value::Primitive(Primitive::U32(l)), Value::Primitive(Primitive::U32(r))) => {
                    Primitive::U32(u32::from(l != 0 || r != 0)).into()
                }
                _ => Value::Uninitialized,
            },

            // Shifts — all cases (cross-type I32 << U32 etc.) handled by Primitive::Shl/Shr
            ShiftLeft => l << r,
            ShiftRight => l >> r,

            // Comparisons
            cmp => {
                fn scalar_cmp<T: PartialOrd>(op: naga::BinaryOperator, a: T, b: T) -> u32 {
                    use naga::BinaryOperator::*;
                    u32::from(match op {
                        Equal => a == b,
                        NotEqual => a != b,
                        Less => a < b,
                        LessEqual => a <= b,
                        Greater => a > b,
                        _ => a >= b,
                    })
                }
                fn cmp_slices<T: PartialOrd + Copy>(
                    op: naga::BinaryOperator,
                    a: &[T],
                    b: &[T],
                ) -> Option<Vec<u32>> {
                    if a.len() != b.len() {
                        return None;
                    }
                    Some(
                        a.iter()
                            .zip(b)
                            .map(|(x, y)| scalar_cmp(op, *x, *y))
                            .collect(),
                    )
                }
                match (l.as_primitive(), r.as_primitive()) {
                    (Some(lp), Some(rp)) => {
                        if let (Some(a), Some(b)) = (lp.as_f32_slice(), rp.as_f32_slice())
                            && let Some(res) = cmp_slices(cmp, a, b)
                        {
                            return Value::from(Primitive::from(res.as_slice()));
                        }
                        if let (Some(a), Some(b)) = (lp.as_i32_slice(), rp.as_i32_slice())
                            && let Some(res) = cmp_slices(cmp, a, b)
                        {
                            return Value::from(Primitive::from(res.as_slice()));
                        }
                        if let (Some(a), Some(b)) = (lp.as_u32_slice(), rp.as_u32_slice())
                            && let Some(res) = cmp_slices(cmp, a, b)
                        {
                            return Value::from(Primitive::from(res.as_slice()));
                        }
                        match (lp, rp) {
                            (Primitive::F64(a), Primitive::F64(b)) => {
                                Primitive::U32(scalar_cmp(cmp, *a, *b)).into()
                            }
                            (Primitive::I64(a), Primitive::I64(b)) => {
                                Primitive::U32(scalar_cmp(cmp, *a, *b)).into()
                            }
                            (Primitive::U64(a), Primitive::U64(b)) => {
                                Primitive::U32(scalar_cmp(cmp, *a, *b)).into()
                            }
                            _ => Value::Uninitialized,
                        }
                    }
                    _ => Value::Uninitialized,
                }
            }
        }
    }

    fn evaluate_global_variable(&self, handle: Handle<GlobalVariable>) -> Value {
        self.global_values
            .get(&handle)
            .cloned()
            .unwrap_or(Value::Uninitialized)
    }

    fn evaluate_access(
        &self,
        base: Handle<Expression>,
        index: Handle<Expression>,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        let base_value = self.evaluate_expression(base, func_idx, parent_func_idx).leaf_value();
        let index_value = self.evaluate_expression(index, func_idx, parent_func_idx).leaf_value();

        let index: usize = match index_value {
            Value::Primitive(Primitive::U32(i)) => i as usize,
            Value::Primitive(Primitive::I32(i)) => i.max(0) as usize,
            _ => return Value::Uninitialized,
        };

        base_value.index_into(index)
    }

    /// Evaluate an expression from the module's global_expressions arena (used for constants/overrides).
    fn evaluate_global_expression(&self, expr_handle: Handle<Expression>) -> Value {
        let expression = &self.module.global_expressions[expr_handle];
        match expression {
            Expression::Literal(literal) => self.evaluate_literal(literal),
            Expression::ZeroValue(ty) => Value::from(&self.module.types[*ty].inner),
            Expression::Constant(handle) => {
                self.evaluate_global_expression(self.module.constants[*handle].init)
            }
            Expression::Compose { ty, components } => {
                let ty_inner = &self.module.types[*ty].inner;
                let vals: Vec<Value> = components
                    .iter()
                    .map(|c| self.evaluate_global_expression(*c))
                    .collect();
                self.assemble_compose(ty_inner, &vals)
            }
            Expression::Splat { size, value } => {
                let val = self.evaluate_global_expression(*value).leaf_value();
                self.splat_value(*size, val)
            }
            _ => Value::Uninitialized,
        }
    }

    /// Assemble a composite value from evaluated components, guided by the target type.
    fn assemble_compose(&self, ty_inner: &TypeInner, vals: &[Value]) -> Value {
        use naga::ScalarKind;
        match ty_inner {
            TypeInner::Array { .. } => Value::Array(vals.to_vec()),
            TypeInner::Struct { members, .. } => {
                let fields = members
                    .iter()
                    .zip(vals.iter())
                    .map(|(m, v)| (m.name.clone().unwrap_or_default(), v.clone()))
                    .collect();
                Value::Struct(fields)
            }
            TypeInner::Vector { size, scalar } => {
                let expected_len = match size {
                    VectorSize::Bi => 2,
                    VectorSize::Tri => 3,
                    VectorSize::Quad => 4,
                };
                // Helper: collect components, truncate to expected_len, and build a Primitive.
                macro_rules! compose_vec {
                    ($collect:expr) => {{
                        let comps = $collect(vals);
                        if comps.len() >= expected_len {
                            Value::from(Primitive::from(&comps[..expected_len]))
                        } else {
                            Value::Uninitialized
                        }
                    }};
                }
                match (scalar.kind, scalar.width) {
                    (ScalarKind::Float, 4) => compose_vec!(Value::collect_f32_components),
                    (ScalarKind::Sint, 4) => compose_vec!(Value::collect_i32_components),
                    (ScalarKind::Uint, 4) => compose_vec!(Value::collect_u32_components),
                    _ => Value::Uninitialized,
                }
            }
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_compose(
        &self,
        ty: naga::Handle<Type>,
        components: &[Handle<Expression>],
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        let ty_inner = &self.module.types[ty].inner;
        let vals: Vec<Value> = components
            .iter()
            .map(|c| self.evaluate_expression(*c, func_idx, parent_func_idx).leaf_value())
            .collect();
        self.assemble_compose(ty_inner, &vals)
    }

    /// Splat a scalar value into a vector of the given size.
    fn splat_value(&self, size: VectorSize, val: Value) -> Value {
        match (size, val) {
            (VectorSize::Bi, Value::Primitive(Primitive::F32(v))) => {
                Primitive::F32x2([v; 2]).into()
            }
            (VectorSize::Tri, Value::Primitive(Primitive::F32(v))) => {
                Primitive::F32x3([v; 3]).into()
            }
            (VectorSize::Quad, Value::Primitive(Primitive::F32(v))) => {
                Primitive::F32x4([v; 4]).into()
            }
            (VectorSize::Bi, Value::Primitive(Primitive::I32(v))) => {
                Primitive::I32x2([v; 2]).into()
            }
            (VectorSize::Tri, Value::Primitive(Primitive::I32(v))) => {
                Primitive::I32x3([v; 3]).into()
            }
            (VectorSize::Quad, Value::Primitive(Primitive::I32(v))) => {
                Primitive::I32x4([v; 4]).into()
            }
            (VectorSize::Bi, Value::Primitive(Primitive::U32(v))) => {
                Primitive::U32x2([v; 2]).into()
            }
            (VectorSize::Tri, Value::Primitive(Primitive::U32(v))) => {
                Primitive::U32x3([v; 3]).into()
            }
            (VectorSize::Quad, Value::Primitive(Primitive::U32(v))) => {
                Primitive::U32x4([v; 4]).into()
            }
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_splat(
        &self,
        size: VectorSize,
        value: Handle<Expression>,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        let val = self.evaluate_expression(value, func_idx, parent_func_idx).leaf_value();
        self.splat_value(size, val)
    }

    fn evaluate_swizzle(
        &self,
        size: VectorSize,
        vector: Handle<Expression>,
        pattern: [SwizzleComponent; 4],
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        let vec_val = self.evaluate_expression(vector, func_idx, parent_func_idx).leaf_value();

        let count = match size {
            VectorSize::Bi => 2,
            VectorSize::Tri => 3,
            VectorSize::Quad => 4,
        };

        let components: Vec<Value> = (0..count)
            .map(|i| vec_val.extract_component(pattern[i] as usize))
            .collect();

        // Reconstruct using collect + from_*_slice based on first component's type
        match components[0].as_primitive() {
            Some(p) if p.as_f32_slice().is_some() => Value::from(Primitive::from(
                Value::collect_f32_components(&components).as_slice(),
            )),
            Some(p) if p.as_i32_slice().is_some() => Value::from(Primitive::from(
                Value::collect_i32_components(&components).as_slice(),
            )),
            Some(p) if p.as_u32_slice().is_some() => Value::from(Primitive::from(
                Value::collect_u32_components(&components).as_slice(),
            )),
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_unary(&self, op: UnaryOperator, val: Value) -> Value {
        match op {
            UnaryOperator::Negate => match val {
                Value::Primitive(Primitive::F64(v)) => Primitive::F64(-v).into(),
                Value::Primitive(Primitive::I64(v)) => Primitive::I64(v.wrapping_neg()).into(),
                _ => val.map_numeric(|f| -f, |i| i.wrapping_neg(), |_| 0),
            },
            UnaryOperator::LogicalNot => match val {
                Value::Primitive(Primitive::U32(v)) => Primitive::U32(u32::from(v == 0)).into(),
                _ => Value::Uninitialized,
            },
            UnaryOperator::BitwiseNot => match val {
                Value::Primitive(Primitive::I64(v)) => Primitive::I64(!v).into(),
                Value::Primitive(Primitive::U64(v)) => Primitive::U64(!v).into(),
                _ => val.map_numeric(|_| 0.0, |i| !i, |u| !u),
            },
        }
    }

    fn evaluate_select(
        &self,
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        let cond = self.evaluate_expression(condition, func_idx, parent_func_idx).leaf_value();

        let is_true = match cond {
            Value::Primitive(Primitive::U32(v)) => v != 0,
            _ => false,
        };

        if is_true {
            self.evaluate_expression(accept, func_idx, parent_func_idx)
        } else {
            self.evaluate_expression(reject, func_idx, parent_func_idx)
        }
    }

    /// Cast/bitcast a scalar or vector value to a different scalar kind.
    fn evaluate_as(
        &self,
        val: Value,
        kind: naga::ScalarKind,
        convert: Option<naga::Bytes>,
    ) -> Value {
        use Primitive::*;
        use naga::ScalarKind::*;

        let val = val.leaf_value();
        let p = match val {
            Value::Primitive(ref p) => p.clone(),
            _ => return Value::Uninitialized,
        };

        match convert {
            Some(4) => match (kind, &p) {
                // Scalar → F32
                (Float, F32(v)) => F32(*v).into(),
                (Float, F64(v)) => F32(*v as f32).into(),
                (Float, I64(v)) => F32(*v as f32).into(),
                (Float, U64(v)) => F32(*v as f32).into(),
                // Scalar → I32
                (Sint, I32(v)) => I32(*v).into(),
                (Sint, F64(v)) => I32(*v as i32).into(),
                (Sint, I64(v)) => I32(*v as i32).into(),
                (Sint, U64(v)) => I32(*v as i32).into(),
                // Scalar → U32
                (Uint, U32(v)) => U32(*v).into(),
                (Uint, F64(v)) => U32(*v as u32).into(),
                (Uint, I64(v)) => U32(*v as u32).into(),
                (Uint, U64(v)) => U32(*v as u32).into(),
                // To Bool (U32 0/1)
                (Bool, U32(v)) => U32(u32::from(*v != 0)).into(),
                (Bool, I32(v)) => U32(u32::from(*v != 0)).into(),
                (Bool, F32(v)) => U32(u32::from(*v != 0.0)).into(),
                // Scalar/Vector → F32xN (from i32 or u32 sources, or identity for f32)
                (Float, _) => p
                    .map_i32_to_f32(|v| v as f32)
                    .or_else(|| p.map_u32_to_f32(|v| v as f32))
                    .map(Value::from)
                    .unwrap_or(Value::Uninitialized),
                // Scalar/Vector → I32xN (from f32 or u32 sources, or identity for i32)
                (Sint, _) => p
                    .map_f32_to_i32(|v| v as i32)
                    .or_else(|| p.map_u32_to_i32(|v| v as i32))
                    .map(Value::from)
                    .unwrap_or(Value::Uninitialized),
                // Scalar/Vector → U32xN (from f32 or i32 sources, or identity for u32)
                (Uint, _) => p
                    .map_f32_to_u32(|v| v as u32)
                    .or_else(|| p.map_i32_to_u32(|v| v as u32))
                    .map(Value::from)
                    .unwrap_or(Value::Uninitialized),
                _ => Value::Uninitialized,
            },
            Some(8) => match (kind, p) {
                (Float, F32(v)) => F64(v as f64).into(),
                (Float, F64(v)) => F64(v).into(),
                (Float, I32(v)) => F64(v as f64).into(),
                (Float, U32(v)) => F64(v as f64).into(),
                (Float, I64(v)) => F64(v as f64).into(),
                (Float, U64(v)) => F64(v as f64).into(),
                (Sint, I32(v)) => I64(v as i64).into(),
                (Sint, U32(v)) => I64(v as i64).into(),
                (Sint, I64(v)) => I64(v).into(),
                (Sint, U64(v)) => I64(v as i64).into(),
                (Sint, F32(v)) => I64(v as i64).into(),
                (Sint, F64(v)) => I64(v as i64).into(),
                (Uint, I32(v)) => U64(v as u64).into(),
                (Uint, U32(v)) => U64(v as u64).into(),
                (Uint, I64(v)) => U64(v as u64).into(),
                (Uint, U64(v)) => U64(v).into(),
                (Uint, F32(v)) => U64(v as u64).into(),
                (Uint, F64(v)) => U64(v as u64).into(),
                _ => Value::Uninitialized,
            },
            // Bitcast — reinterpret bits (scalars handled explicitly, vectors via cross-type maps)
            None => match (kind, &p) {
                (Float, I32(v)) => F32(f32::from_bits(*v as u32)).into(),
                (Float, U32(v)) => F32(f32::from_bits(*v)).into(),
                (Sint, F32(v)) => I32(v.to_bits() as i32).into(),
                (Sint, U32(v)) => I32(*v as i32).into(),
                (Uint, F32(v)) => U32(v.to_bits()).into(),
                (Uint, I32(v)) => U32(*v as u32).into(),
                (Float, _) => p
                    .map_i32_to_f32(|v| f32::from_bits(v as u32))
                    .or_else(|| p.map_u32_to_f32(f32::from_bits))
                    .map(Value::from)
                    .unwrap_or(Value::Uninitialized),
                (Sint, _) => p
                    .map_f32_to_i32(|v| v.to_bits() as i32)
                    .or_else(|| p.map_u32_to_i32(|v| v as i32))
                    .map(Value::from)
                    .unwrap_or(Value::Uninitialized),
                (Uint, _) => p
                    .map_f32_to_u32(|v| v.to_bits())
                    .or_else(|| p.map_i32_to_u32(|v| v as u32))
                    .map(Value::from)
                    .unwrap_or(Value::Uninitialized),
                _ => Value::Uninitialized,
            },
            _ => Value::Uninitialized,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_math(
        &self,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        arg2: Option<Handle<Expression>>,
        arg3: Option<Handle<Expression>>,
        func_idx: usize,
        parent_func_idx: Option<usize>,
    ) -> Value {
        let a = self.evaluate_expression(arg, func_idx, parent_func_idx).leaf_value();
        let b = arg1.map(|h| self.evaluate_expression(h, func_idx, parent_func_idx).leaf_value());
        let c = arg2.map(|h| self.evaluate_expression(h, func_idx, parent_func_idx).leaf_value());
        let _d = arg3.map(|h| self.evaluate_expression(h, func_idx, parent_func_idx).leaf_value());

        match fun {
            // --- Comparison ---
            MathFunction::Abs => self.math_unary_float_or_int(
                a,
                f32::abs,
                f64::abs,
                |v: i32| v.wrapping_abs(),
                |v: u32| v,
            ),
            MathFunction::Min => self.math_binary_float_or_int(
                a,
                b.unwrap_or(Value::Uninitialized),
                f32::min,
                f64::min,
                i32::min,
                u32::min,
            ),
            MathFunction::Max => self.math_binary_float_or_int(
                a,
                b.unwrap_or(Value::Uninitialized),
                f32::max,
                f64::max,
                i32::max,
                u32::max,
            ),
            MathFunction::Clamp => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                let c_val = c.unwrap_or(Value::Uninitialized);
                self.math_clamp(a, b_val, c_val)
            }
            MathFunction::Saturate => self.math_unary_f32(a, |v| v.clamp(0.0, 1.0)),

            // --- Trigonometry ---
            MathFunction::Cos => self.math_unary_f32(a, f32::cos),
            MathFunction::Cosh => self.math_unary_f32(a, f32::cosh),
            MathFunction::Sin => self.math_unary_f32(a, f32::sin),
            MathFunction::Sinh => self.math_unary_f32(a, f32::sinh),
            MathFunction::Tan => self.math_unary_f32(a, f32::tan),
            MathFunction::Tanh => self.math_unary_f32(a, f32::tanh),
            MathFunction::Acos => self.math_unary_f32(a, f32::acos),
            MathFunction::Asin => self.math_unary_f32(a, f32::asin),
            MathFunction::Atan => self.math_unary_f32(a, f32::atan),
            MathFunction::Atan2 => {
                self.math_binary_f32(a, b.unwrap_or(Value::Uninitialized), f32::atan2)
            }
            MathFunction::Asinh => self.math_unary_f32(a, f32::asinh),
            MathFunction::Acosh => self.math_unary_f32(a, f32::acosh),
            MathFunction::Atanh => self.math_unary_f32(a, f32::atanh),
            MathFunction::Radians => self.math_unary_f32(a, |v| v.to_radians()),
            MathFunction::Degrees => self.math_unary_f32(a, |v| v.to_degrees()),

            // --- Decomposition ---
            MathFunction::Ceil => self.math_unary_f32(a, f32::ceil),
            MathFunction::Floor => self.math_unary_f32(a, f32::floor),
            MathFunction::Round => self.math_unary_f32(a, f32::round),
            MathFunction::Fract => self.math_unary_f32(a, f32::fract),
            MathFunction::Trunc => self.math_unary_f32(a, f32::trunc),
            MathFunction::Ldexp => {
                let base = a.as_primitive().and_then(|p| p.as_f32_slice()).unwrap();
                let b_val = b.unwrap();
                let exponent = b_val.as_primitive().and_then(|p| p.as_i32_slice()).unwrap();

                let floats: Vec<f32> = base
                    .iter()
                    .zip(exponent)
                    .map(|(base, exp)| base * i32::pow(2, *exp as u32) as f32)
                    .collect();

                Value::Primitive(Primitive::from(floats.as_slice()))
            }

            // --- Exponent ---
            MathFunction::Exp => self.math_unary_f32(a, f32::exp),
            MathFunction::Exp2 => self.math_unary_f32(a, f32::exp2),
            MathFunction::Log => self.math_unary_f32(a, f32::ln),
            MathFunction::Log2 => self.math_unary_f32(a, f32::log2),
            MathFunction::Pow => {
                self.math_binary_f32(a, b.unwrap_or(Value::Uninitialized), f32::powf)
            }

            // --- Geometry ---
            MathFunction::Dot => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                use Primitive::*;
                match (a, b_val) {
                    (Value::Primitive(F32x2([a0, a1])), Value::Primitive(F32x2([b0, b1]))) => {
                        Primitive::F32(a0 * b0 + a1 * b1).into()
                    }
                    (
                        Value::Primitive(F32x3([a0, a1, a2])),
                        Value::Primitive(F32x3([b0, b1, b2])),
                    ) => Primitive::F32(a0 * b0 + a1 * b1 + a2 * b2).into(),
                    (
                        Value::Primitive(F32x4([a0, a1, a2, a3])),
                        Value::Primitive(F32x4([b0, b1, b2, b3])),
                    ) => Primitive::F32(a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3).into(),
                    (Value::Primitive(I32x2([a0, a1])), Value::Primitive(I32x2([b0, b1]))) => {
                        Primitive::I32(a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1))).into()
                    }
                    (
                        Value::Primitive(I32x3([a0, a1, a2])),
                        Value::Primitive(I32x3([b0, b1, b2])),
                    ) => Primitive::I32(
                        a0.wrapping_mul(b0)
                            .wrapping_add(a1.wrapping_mul(b1))
                            .wrapping_add(a2.wrapping_mul(b2)),
                    )
                    .into(),
                    (
                        Value::Primitive(I32x4([a0, a1, a2, a3])),
                        Value::Primitive(I32x4([b0, b1, b2, b3])),
                    ) => Primitive::I32(
                        a0.wrapping_mul(b0)
                            .wrapping_add(a1.wrapping_mul(b1))
                            .wrapping_add(a2.wrapping_mul(b2))
                            .wrapping_add(a3.wrapping_mul(b3)),
                    )
                    .into(),
                    (Value::Primitive(U32x2([a0, a1])), Value::Primitive(U32x2([b0, b1]))) => {
                        Primitive::U32(a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1))).into()
                    }
                    (
                        Value::Primitive(U32x3([a0, a1, a2])),
                        Value::Primitive(U32x3([b0, b1, b2])),
                    ) => Primitive::U32(
                        a0.wrapping_mul(b0)
                            .wrapping_add(a1.wrapping_mul(b1))
                            .wrapping_add(a2.wrapping_mul(b2)),
                    )
                    .into(),
                    (
                        Value::Primitive(U32x4([a0, a1, a2, a3])),
                        Value::Primitive(U32x4([b0, b1, b2, b3])),
                    ) => Primitive::U32(
                        a0.wrapping_mul(b0)
                            .wrapping_add(a1.wrapping_mul(b1))
                            .wrapping_add(a2.wrapping_mul(b2))
                            .wrapping_add(a3.wrapping_mul(b3)),
                    )
                    .into(),
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Cross => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                match (a, b_val) {
                    (
                        Value::Primitive(Primitive::F32x3([a0, a1, a2])),
                        Value::Primitive(Primitive::F32x3([b0, b1, b2])),
                    ) => {
                        Primitive::F32x3([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])
                            .into()
                    }
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Distance => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                match (a, b_val) {
                    (Value::Primitive(Primitive::F32(a)), Value::Primitive(Primitive::F32(b))) => {
                        Primitive::F32((a - b).abs()).into()
                    }
                    (
                        Value::Primitive(Primitive::F32x2([a0, a1])),
                        Value::Primitive(Primitive::F32x2([b0, b1])),
                    ) => Primitive::F32(((a0 - b0).powi(2) + (a1 - b1).powi(2)).sqrt()).into(),
                    (
                        Value::Primitive(Primitive::F32x3([a0, a1, a2])),
                        Value::Primitive(Primitive::F32x3([b0, b1, b2])),
                    ) => Primitive::F32(
                        ((a0 - b0).powi(2) + (a1 - b1).powi(2) + (a2 - b2).powi(2)).sqrt(),
                    )
                    .into(),
                    (
                        Value::Primitive(Primitive::F32x4([a0, a1, a2, a3])),
                        Value::Primitive(Primitive::F32x4([b0, b1, b2, b3])),
                    ) => Primitive::F32(
                        ((a0 - b0).powi(2)
                            + (a1 - b1).powi(2)
                            + (a2 - b2).powi(2)
                            + (a3 - b3).powi(2))
                        .sqrt(),
                    )
                    .into(),
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Length => match a {
                Value::Primitive(Primitive::F32(v)) => Primitive::F32(v.abs()).into(),
                Value::Primitive(Primitive::F32x2([x, y])) => {
                    Primitive::F32((x * x + y * y).sqrt()).into()
                }
                Value::Primitive(Primitive::F32x3([x, y, z])) => {
                    Primitive::F32((x * x + y * y + z * z).sqrt()).into()
                }
                Value::Primitive(Primitive::F32x4([x, y, z, w])) => {
                    Primitive::F32((x * x + y * y + z * z + w * w).sqrt()).into()
                }
                _ => Value::Uninitialized,
            },
            MathFunction::Normalize => match a {
                Value::Primitive(Primitive::F32x2([x, y])) => {
                    let len = (x * x + y * y).sqrt();
                    if len != 0.0 {
                        Primitive::F32x2([x / len, y / len]).into()
                    } else {
                        Primitive::F32x2([0.0; 2]).into()
                    }
                }
                Value::Primitive(Primitive::F32x3([x, y, z])) => {
                    let len = (x * x + y * y + z * z).sqrt();
                    if len != 0.0 {
                        Primitive::F32x3([x / len, y / len, z / len]).into()
                    } else {
                        Primitive::F32x3([0.0; 3]).into()
                    }
                }
                Value::Primitive(Primitive::F32x4([x, y, z, w])) => {
                    let len = (x * x + y * y + z * z + w * w).sqrt();
                    if len != 0.0 {
                        Primitive::F32x4([x / len, y / len, z / len, w / len]).into()
                    } else {
                        Primitive::F32x4([0.0; 4]).into()
                    }
                }
                _ => Value::Uninitialized,
            },
            MathFunction::FaceForward => {
                let dot_val = if let (Some(e2), Some(e3)) = (arg1, arg2) {
                    self.evaluate_math(MathFunction::Dot, e2, Some(e3), None, None, func_idx, parent_func_idx)
                } else {
                    Value::Uninitialized
                };
                let dot_negative = match dot_val {
                    Value::Primitive(Primitive::F32(v)) => v < 0.0,
                    _ => false,
                };
                if dot_negative {
                    a
                } else {
                    self.evaluate_unary(UnaryOperator::Negate, a)
                }
            }
            MathFunction::Reflect => {
                // reflect(e1, e2) = e1 - 2 * dot(e2, e1) * e2
                let b_val = b.unwrap_or(Value::Uninitialized);
                let dot = match (&a, &b_val) {
                    (
                        Value::Primitive(Primitive::F32x2([a0, a1])),
                        Value::Primitive(Primitive::F32x2([b0, b1])),
                    ) => b0 * a0 + b1 * a1,
                    (
                        Value::Primitive(Primitive::F32x3([a0, a1, a2])),
                        Value::Primitive(Primitive::F32x3([b0, b1, b2])),
                    ) => b0 * a0 + b1 * a1 + b2 * a2,
                    (
                        Value::Primitive(Primitive::F32x4([a0, a1, a2, a3])),
                        Value::Primitive(Primitive::F32x4([b0, b1, b2, b3])),
                    ) => b0 * a0 + b1 * a1 + b2 * a2 + b3 * a3,
                    _ => return Value::Uninitialized,
                };
                let factor = 2.0 * dot;
                let scaled_n = b_val.map_f32(|x| x * factor);
                a.zip_map_f32(scaled_n, |e, s| e - s)
            }

            // --- Computational ---
            MathFunction::Sign => self.math_unary_float_or_int(
                a,
                |v: f32| v.signum(),
                |v: f64| v.signum(),
                |v: i32| v.signum(),
                |v: u32| if v > 0 { 1 } else { 0 },
            ),
            MathFunction::Fma => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                let c_val = c.unwrap_or(Value::Uninitialized);
                use Primitive::*;
                match (a, b_val, c_val) {
                    (
                        Value::Primitive(F32(a)),
                        Value::Primitive(F32(b)),
                        Value::Primitive(F32(c)),
                    ) => Primitive::F32(a.mul_add(b, c)).into(),
                    (
                        Value::Primitive(F32x2([a0, a1])),
                        Value::Primitive(F32x2([b0, b1])),
                        Value::Primitive(F32x2([c0, c1])),
                    ) => Primitive::F32x2([a0.mul_add(b0, c0), a1.mul_add(b1, c1)]).into(),
                    (
                        Value::Primitive(F32x3([a0, a1, a2])),
                        Value::Primitive(F32x3([b0, b1, b2])),
                        Value::Primitive(F32x3([c0, c1, c2])),
                    ) => Primitive::F32x3([
                        a0.mul_add(b0, c0),
                        a1.mul_add(b1, c1),
                        a2.mul_add(b2, c2),
                    ])
                    .into(),
                    (
                        Value::Primitive(F32x4([a0, a1, a2, a3])),
                        Value::Primitive(F32x4([b0, b1, b2, b3])),
                        Value::Primitive(F32x4([c0, c1, c2, c3])),
                    ) => Primitive::F32x4([
                        a0.mul_add(b0, c0),
                        a1.mul_add(b1, c1),
                        a2.mul_add(b2, c2),
                        a3.mul_add(b3, c3),
                    ])
                    .into(),
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Mix => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                let c_val = c.unwrap_or(Value::Uninitialized);
                use Primitive::*;
                match (a, b_val, c_val) {
                    (
                        Value::Primitive(F32(a)),
                        Value::Primitive(F32(b)),
                        Value::Primitive(F32(t)),
                    ) => Primitive::F32(a * (1.0 - t) + b * t).into(),
                    (
                        Value::Primitive(F32x2([a0, a1])),
                        Value::Primitive(F32x2([b0, b1])),
                        Value::Primitive(F32x2([t0, t1])),
                    ) => Primitive::F32x2([a0 * (1.0 - t0) + b0 * t0, a1 * (1.0 - t1) + b1 * t1])
                        .into(),
                    (
                        Value::Primitive(F32x3([a0, a1, a2])),
                        Value::Primitive(F32x3([b0, b1, b2])),
                        Value::Primitive(F32x3([t0, t1, t2])),
                    ) => Primitive::F32x3([
                        a0 * (1.0 - t0) + b0 * t0,
                        a1 * (1.0 - t1) + b1 * t1,
                        a2 * (1.0 - t2) + b2 * t2,
                    ])
                    .into(),
                    (
                        Value::Primitive(F32x4([a0, a1, a2, a3])),
                        Value::Primitive(F32x4([b0, b1, b2, b3])),
                        Value::Primitive(F32x4([t0, t1, t2, t3])),
                    ) => Primitive::F32x4([
                        a0 * (1.0 - t0) + b0 * t0,
                        a1 * (1.0 - t1) + b1 * t1,
                        a2 * (1.0 - t2) + b2 * t2,
                        a3 * (1.0 - t3) + b3 * t3,
                    ])
                    .into(),
                    // mix with scalar t
                    (
                        Value::Primitive(F32x2([a0, a1])),
                        Value::Primitive(F32x2([b0, b1])),
                        Value::Primitive(F32(t)),
                    ) => {
                        Primitive::F32x2([a0 * (1.0 - t) + b0 * t, a1 * (1.0 - t) + b1 * t]).into()
                    }
                    (
                        Value::Primitive(F32x3([a0, a1, a2])),
                        Value::Primitive(F32x3([b0, b1, b2])),
                        Value::Primitive(F32(t)),
                    ) => Primitive::F32x3([
                        a0 * (1.0 - t) + b0 * t,
                        a1 * (1.0 - t) + b1 * t,
                        a2 * (1.0 - t) + b2 * t,
                    ])
                    .into(),
                    (
                        Value::Primitive(F32x4([a0, a1, a2, a3])),
                        Value::Primitive(F32x4([b0, b1, b2, b3])),
                        Value::Primitive(F32(t)),
                    ) => Primitive::F32x4([
                        a0 * (1.0 - t) + b0 * t,
                        a1 * (1.0 - t) + b1 * t,
                        a2 * (1.0 - t) + b2 * t,
                        a3 * (1.0 - t) + b3 * t,
                    ])
                    .into(),
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Step => {
                // step(edge, x) = if x < edge { 0.0 } else { 1.0 }
                self.math_binary_f32(a, b.unwrap_or(Value::Uninitialized), |edge, x| {
                    if x < edge { 0.0 } else { 1.0 }
                })
            }
            MathFunction::SmoothStep => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                let c_val = c.unwrap_or(Value::Uninitialized);
                match (a, b_val, c_val) {
                    (
                        Value::Primitive(Primitive::F32(low)),
                        Value::Primitive(Primitive::F32(high)),
                        Value::Primitive(Primitive::F32(x)),
                    ) => {
                        let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
                        Primitive::F32(t * t * (3.0 - 2.0 * t)).into()
                    }
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Sqrt => self.math_unary_f32(a, f32::sqrt),
            MathFunction::InverseSqrt => self.math_unary_f32(a, |v| 1.0 / v.sqrt()),
            MathFunction::QuantizeToF16 => {
                // Approximate: round to f16 precision
                self.math_unary_f32(a, |v| {
                    let bits = v.to_bits();
                    // Simple truncation of mantissa to 10 bits
                    let truncated = bits & 0xFFFFE000;
                    f32::from_bits(truncated)
                })
            }

            // --- Bits ---
            MathFunction::CountTrailingZeros => self.math_unary_int(
                a,
                |v: i32| v.trailing_zeros() as i32,
                |v: u32| v.trailing_zeros(),
            ),
            MathFunction::CountLeadingZeros => self.math_unary_int(
                a,
                |v: i32| v.leading_zeros() as i32,
                |v: u32| v.leading_zeros(),
            ),
            MathFunction::CountOneBits => {
                self.math_unary_int(a, |v: i32| v.count_ones() as i32, |v: u32| v.count_ones())
            }
            MathFunction::ReverseBits => {
                self.math_unary_int(a, |v: i32| v.reverse_bits(), |v: u32| v.reverse_bits())
            }
            MathFunction::FirstTrailingBit => self.math_unary_int(
                a,
                |v: i32| {
                    if v == 0 {
                        -1i32
                    } else {
                        v.trailing_zeros() as i32
                    }
                },
                |v: u32| {
                    if v == 0 {
                        0xFFFFFFFF
                    } else {
                        v.trailing_zeros()
                    }
                },
            ),
            MathFunction::FirstLeadingBit => self.math_unary_int(
                a,
                |v: i32| {
                    if v == 0 || v == -1 {
                        -1i32
                    } else if v > 0 {
                        31 - v.leading_zeros() as i32
                    } else {
                        31 - (!v).leading_zeros() as i32
                    }
                },
                |v: u32| {
                    if v == 0 {
                        0xFFFFFFFF
                    } else {
                        31 - v.leading_zeros()
                    }
                },
            ),
            MathFunction::ExtractBits => {
                let offset = match b.unwrap_or(Value::Uninitialized) {
                    Value::Primitive(Primitive::U32(v)) => v,
                    _ => return Value::Uninitialized,
                };
                let count = match c.unwrap_or(Value::Uninitialized) {
                    Value::Primitive(Primitive::U32(v)) => v,
                    _ => return Value::Uninitialized,
                };
                match a {
                    Value::Primitive(Primitive::I32(v)) => {
                        if count == 0 {
                            Primitive::I32(0).into()
                        } else {
                            let shifted = (v as u32).wrapping_shr(offset);
                            let mask = if count >= 32 {
                                u32::MAX
                            } else {
                                (1u32 << count) - 1
                            };
                            let extracted = shifted & mask;
                            let sign_bit = 1u32 << (count - 1);
                            let sign_extended = if extracted & sign_bit != 0 {
                                extracted | !mask
                            } else {
                                extracted
                            };
                            Primitive::I32(sign_extended as i32).into()
                        }
                    }
                    Value::Primitive(Primitive::U32(v)) => {
                        if count == 0 {
                            Primitive::U32(0).into()
                        } else {
                            let shifted = v.wrapping_shr(offset);
                            let mask = if count >= 32 {
                                u32::MAX
                            } else {
                                (1u32 << count) - 1
                            };
                            Primitive::U32(shifted & mask).into()
                        }
                    }
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::InsertBits => {
                let newbits = match b.unwrap_or(Value::Uninitialized) {
                    Value::Primitive(Primitive::I32(v)) => v as u32,
                    Value::Primitive(Primitive::U32(v)) => v,
                    _ => return Value::Uninitialized,
                };
                let offset = match c.unwrap_or(Value::Uninitialized) {
                    Value::Primitive(Primitive::U32(v)) => v,
                    _ => return Value::Uninitialized,
                };
                let count = match _d.unwrap_or(Value::Uninitialized) {
                    Value::Primitive(Primitive::U32(v)) => v,
                    _ => return Value::Uninitialized,
                };
                match a {
                    Value::Primitive(Primitive::I32(v)) => {
                        let mask = if count >= 32 {
                            u32::MAX
                        } else {
                            ((1u32 << count) - 1) << offset
                        };
                        Primitive::I32((((v as u32) & !mask) | ((newbits << offset) & mask)) as i32)
                            .into()
                    }
                    Value::Primitive(Primitive::U32(v)) => {
                        let mask = if count >= 32 {
                            u32::MAX
                        } else {
                            ((1u32 << count) - 1) << offset
                        };
                        Primitive::U32((v & !mask) | ((newbits << offset) & mask)).into()
                    }
                    _ => Value::Uninitialized,
                }
            }

            // Unsupported / GPU-specific functions return Uninitialized
            _ => Value::Uninitialized,
        }
    }

    // --- Math helper methods ---

    fn math_unary_f32(&self, val: Value, f: impl Fn(f32) -> f32) -> Value {
        val.map_f32(f)
    }

    fn math_binary_f32(&self, a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Value {
        a.zip_map_f32(b, f)
    }

    fn math_unary_float_or_int(
        &self,
        val: Value,
        ff32: impl Fn(f32) -> f32,
        ff64: impl Fn(f64) -> f64,
        fi32: impl Fn(i32) -> i32,
        fu32: impl Fn(u32) -> u32,
    ) -> Value {
        match val {
            Value::Primitive(Primitive::F64(v)) => Primitive::F64(ff64(v)).into(),
            _ => val.map_numeric(ff32, fi32, fu32),
        }
    }

    fn math_binary_float_or_int(
        &self,
        a: Value,
        b: Value,
        ff32: impl Fn(f32, f32) -> f32,
        ff64: impl Fn(f64, f64) -> f64,
        fi32: impl Fn(i32, i32) -> i32,
        fu32: impl Fn(u32, u32) -> u32,
    ) -> Value {
        match (&a, &b) {
            (Value::Primitive(Primitive::F64(av)), Value::Primitive(Primitive::F64(bv))) => {
                Primitive::F64(ff64(*av, *bv)).into()
            }
            _ => a.zip_map_numeric(b, ff32, fi32, fu32),
        }
    }

    fn math_unary_int(
        &self,
        val: Value,
        fi32: impl Fn(i32) -> i32,
        fu32: impl Fn(u32) -> u32,
    ) -> Value {
        val.map_numeric(|_| 0.0, fi32, fu32)
    }

    fn math_clamp(&self, val: Value, min_val: Value, max_val: Value) -> Value {
        val.zip3_map_numeric(
            min_val,
            max_val,
            |v, lo, hi| v.clamp(lo, hi),
            |v, lo, hi| v.clamp(lo, hi),
            |v, lo, hi| v.clamp(lo, hi),
        )
    }

    fn evaluate_relational(&self, fun: RelationalFunction, val: Value) -> Value {
        match fun {
            RelationalFunction::All => {
                if let Some(comps) = val.as_primitive().and_then(|p| p.as_u32_slice()) {
                    Primitive::U32(u32::from(comps.iter().all(|&v| v != 0))).into()
                } else {
                    Value::Uninitialized
                }
            }
            RelationalFunction::Any => {
                if let Some(comps) = val.as_primitive().and_then(|p| p.as_u32_slice()) {
                    Primitive::U32(u32::from(comps.iter().any(|&v| v != 0))).into()
                } else {
                    Value::Uninitialized
                }
            }
            RelationalFunction::IsNan => {
                if let Some(comps) = val.as_primitive().and_then(|p| p.as_f32_slice()) {
                    let result: Vec<u32> = comps.iter().map(|v| u32::from(v.is_nan())).collect();
                    Value::from(Primitive::from(result.as_slice()))
                } else {
                    Value::Uninitialized
                }
            }
            RelationalFunction::IsInf => {
                if let Some(comps) = val.as_primitive().and_then(|p| p.as_f32_slice()) {
                    let result: Vec<u32> =
                        comps.iter().map(|v| u32::from(v.is_infinite())).collect();
                    Value::from(Primitive::from(result.as_slice()))
                } else {
                    Value::Uninitialized
                }
            }
        }
    }
}
