use crate::{
    function_state::{FunctionState, NextStatement},
    value::{Value, ScalarComponents},
};

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use naga::{
    Expression, GlobalVariable, Literal, LocalVariable, MathFunction, Module,
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
    stack: Vec<FunctionState>,
}

impl Evaluator {
    pub fn new(
        module: &Module,
        entry_point_index: usize,
        entry_point_inputs: EntryPointInputs,
        global_values: HashMap<naga::ResourceBinding, Value>,
    ) -> Self {
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
            stack: vec![FunctionState {
                function: module.entry_points[entry_point_index].function.clone(),
                function_handle: None,
                local_variables: HashMap::new(),
                evaluated_expressions: HashMap::new(),
                evaluated_function_arguments: Vec::new(),
                current_statement_index: 0,
                returns: None,
            }],
        };
        evaluator.initialize_local_variables(0, None);
        evaluator
    }

    pub fn peek(&self) -> &FunctionState {
        &self.stack[self.stack.len() - 1]
    }

    /// Evaluate all named expressions (WGSL `let` bindings) in the current frame.
    /// Returns `(name, value)` pairs in source order.
    pub fn named_expression_values(&self) -> Vec<(String, Value)> {
        let state_index = self.stack.len() - 1;
        let parent_state_index = if state_index > 0 { Some(state_index - 1) } else { None };
        let state = &self.stack[state_index];
        state
            .function
            .named_expressions
            .iter()
            .map(|(handle, name)| {
                (name.clone(), self.evaluate_expression(*handle, state_index, parent_state_index))
            })
            .collect()
    }

    pub fn next_statement(&mut self) -> Option<NextStatement> {
        let current_state_index = self.stack.len() - 1;
        let parent_state_index = if current_state_index == 0 {
            None
        } else {
            Some(current_state_index - 1)
        };
        let current_statement_index = self.stack[current_state_index].current_statement_index;
        let current_state_statements_len = self.stack[current_state_index].function.body.len();

        if current_statement_index < current_state_statements_len {
            let current_statement = self.stack[current_state_index]
                .function
                .body
                .get(current_statement_index)
                .cloned()
                .unwrap();

            self.handle_statement(
                current_statement,
                current_state_index,
                parent_state_index,
            );

            self.stack[current_state_index].current_statement_index += 1;

            // Pop stack if we've finished the current function
            if self.stack[current_state_index].current_statement_index
                >= self.stack[current_state_index].function.body.len()
            {
                self.stack.pop();
            }

            // Return the next statement if available
            if let Some(state) = self.stack.last() {
                return Some(NextStatement {
                    function: state.function.clone(),
                    statement: state
                        .function
                        .body
                        .get(state.current_statement_index)
                        .cloned()
                        .unwrap(),
                    statement_index: state.current_statement_index,
                });
            }

            None
        } else {
            None
        }
    }

    fn handle_statement(
        &mut self,
        statement: Statement,
        current_state_index: usize,
        parent_state_index: Option<usize>,
    ) {
        match statement {
            Statement::Emit(expression_handles) => {
                self.handle_emit(expression_handles, current_state_index, parent_state_index);
            }
            Statement::Call {
                function: function_handle,
                arguments,
                result: _result,
            } => {
                self.handle_call(
                    function_handle,
                    arguments,
                    current_state_index,
                    parent_state_index,
                );
            }
            Statement::Store { pointer, value } => {
                self.handle_store(pointer, value, current_state_index, parent_state_index);
            }
            Statement::Return { value } => {
                if let Some(value) = value {
                    self.handle_return(value, current_state_index, parent_state_index);
                }
            }
            Statement::Block(_)
            | Statement::If { .. }
            | Statement::Switch { .. }
            | Statement::Loop { .. }
            | Statement::Break
            | Statement::Continue
            | Statement::Kill
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

    fn handle_emit(
        &mut self,
        _expression_handles: naga::Range<naga::Expression>,
        _state_index: usize,
        _parent_state_index: Option<usize>,
    ) {
        // Expressions are evaluated lazily on demand, so there is nothing to do here.
    }

    fn handle_call(
        &mut self,
        function_handle: naga::Handle<naga::Function>,
        arguments: Vec<naga::Handle<Expression>>,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) {
        let evaluated_function_arguments = arguments
            .iter()
            .map(|&arg| self.evaluate_expression(arg, state_index, parent_state_index))
            .collect();

        let new_function_state = FunctionState {
            function: self.module.functions[function_handle].clone(),
            function_handle: Some(function_handle),
            local_variables: HashMap::new(),
            evaluated_expressions: HashMap::new(),
            evaluated_function_arguments,
            current_statement_index: 0,
            returns: None,
        };
        self.stack.push(new_function_state);

        let new_state_index = self.stack.len() - 1;
        self.initialize_local_variables(new_state_index, parent_state_index);
    }

    fn initialize_local_variables(&mut self, state_index: usize, parent_state_index: Option<usize>) {
        let mut insert_variables = Vec::new();

        for local_variable in self.stack[state_index].function.local_variables.iter() {
            let evaluated_local_variable = match &local_variable.1.init {
                Some(init_expr) => {
                    self.evaluate_expression(*init_expr, state_index, parent_state_index)
                }
                None => {
                    let ty = &self.module.types[local_variable.1.ty];
                    Value::from(&ty.inner)
                }
            };

            insert_variables.push((local_variable.0, evaluated_local_variable));
        }

        for (handle, value) in insert_variables {
            self.stack[state_index]
                .local_variables
                .insert(handle, Value::Pointer(Rc::new(RefCell::new(value))));
        }
    }

    fn handle_store(
        &mut self,
        pointer: naga::Handle<Expression>,
        value: naga::Handle<Expression>,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) {
        let evaluated_value = self.evaluate_expression(value, state_index, parent_state_index);
        let evaluated_pointer = self.evaluate_expression(pointer, state_index, parent_state_index);

        match evaluated_pointer {
            Value::Pointer(inner) => {
                *inner.borrow_mut() = evaluated_value;
            }
            _ => {
                eprintln!("Store to non-pointer value: {:?}", evaluated_pointer);
            }
        }
    }

    fn handle_return(
        &mut self,
        value: naga::Handle<Expression>,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) {
        let return_value = self.evaluate_expression(value, state_index, parent_state_index);

        if let Some(parent_state_index) = parent_state_index {
            self.stack[parent_state_index].returns = Some(return_value);
        }
    }

    pub fn evaluate_expression(
        &self,
        expression_handle: naga::Handle<Expression>,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        let state = &self.stack[state_index];
        let expression = &state.function.expressions[expression_handle];

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
                self.evaluate_compose(*ty, components, state_index, parent_state_index)
            }
            Expression::Splat { size, value } => {
                self.evaluate_splat(*size, *value, state_index, parent_state_index)
            }
            Expression::Swizzle { size, vector, pattern } => {
                self.evaluate_swizzle(*size, *vector, pattern.clone(), state_index, parent_state_index)
            }
            Expression::Load { pointer } => {
                self.evaluate_load(*pointer, state, state_index, parent_state_index)
            }
            Expression::AccessIndex { base, index } => {
                self.evaluate_access_index(*base, *index, state_index, parent_state_index)
            }
            Expression::FunctionArgument(index) => {
                self.evaluate_function_argument(*index as usize, state)
            }
            Expression::LocalVariable(handle) => {
                self.evaluate_local_variable(*handle, state, state_index, parent_state_index)
            }
            Expression::Binary { op, left, right } => {
                self.evaluate_binary(*op, *left, *right, state_index, parent_state_index)
            }
            Expression::Unary { op, expr } => {
                let val = self
                    .evaluate_expression(*expr, state_index, parent_state_index)
                    .leaf_value();
                self.evaluate_unary(*op, val)
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => self.evaluate_select(
                *condition,
                *accept,
                *reject,
                state_index,
                parent_state_index,
            ),
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let val = self.evaluate_expression(*expr, state_index, parent_state_index);
                self.evaluate_as(val, *kind, *convert)
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self.evaluate_math(
                *fun,
                *arg,
                *arg1,
                *arg2,
                *arg3,
                state_index,
                parent_state_index,
            ),
            Expression::Relational { fun, argument } => {
                let val = self
                    .evaluate_expression(*argument, state_index, parent_state_index)
                    .leaf_value();
                self.evaluate_relational(*fun, val)
            }
            Expression::ArrayLength(expr) => {
                match self.evaluate_expression(*expr, state_index, parent_state_index) {
                    Value::Pointer(inner) => match &*inner.borrow() {
                        Value::Array(elements) => Value::U32(elements.len() as u32),
                        _ => Value::Uninitialized,
                    },
                    _ => Value::Uninitialized,
                }
            }
            Expression::GlobalVariable(handle) => self.evaluate_global_variable(*handle),
            Expression::Access { base, index } => {
                self.evaluate_access(*base, *index, state_index, parent_state_index)
            }
            Expression::CallResult(_handle) => state.returns.clone().unwrap_or(Value::Uninitialized),
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_literal(&self, literal: &Literal) -> Value {
        match literal {
            Literal::F32(v) => Value::F32(*v),
            Literal::F64(v) => Value::F64(*v),
            Literal::I32(v) => Value::I32(*v),
            Literal::I64(v) => Value::I64(*v),
            Literal::U32(v) => Value::U32(*v),
            Literal::U64(v) => Value::U64(*v),
            Literal::Bool(v) => Value::U32(if *v { 1 } else { 0 }),
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_load(
        &self,
        pointer: naga::Handle<Expression>,
        state: &FunctionState,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        if let Some(value) = state.evaluated_expressions.get(&pointer).cloned() {
            value
        } else {
            match self.evaluate_expression(pointer, state_index, parent_state_index) {
                Value::Pointer(inner) => inner.borrow().clone(),
                _ => Value::Uninitialized,
            }
        }
    }

    fn evaluate_access_index(
        &self,
        base: naga::Handle<Expression>,
        index: u32,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        let value = self.evaluate_expression(base, state_index, parent_state_index).leaf_value();
        let index = index as usize;

        match value {
            Value::Array(elements) => elements
                .get(index)
                .map(|e| Value::Pointer(e.clone()))
                .unwrap_or(Value::Uninitialized),
            Value::U32x2(arr) => Value::U32(arr[index]),
            Value::U32x3(arr) => Value::U32(arr[index]),
            Value::U32x4(arr) => Value::U32(arr[index]),
            Value::I32x2(arr) => Value::I32(arr[index]),
            Value::I32x3(arr) => Value::I32(arr[index]),
            Value::I32x4(arr) => Value::I32(arr[index]),
            Value::F32x2(arr) => Value::F32(arr[index]),
            Value::F32x3(arr) => Value::F32(arr[index]),
            Value::F32x4(arr) => Value::F32(arr[index]),
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_function_argument(&self, index: usize, state: &FunctionState) -> Value {
        let function_argument = &state.function.arguments[index];

        if let Some(binding) = &function_argument.binding {
            match binding {
                naga::ir::Binding::BuiltIn(built_in) => match built_in {
                    naga::ir::BuiltIn::GlobalInvocationId => {
                        Value::U32x3(self.entry_point_inputs.global_invocation_id)
                    }
                    _ => Value::Uninitialized,
                },
                naga::ir::Binding::Location { .. } => Value::Uninitialized,
            }
        } else {
            state.evaluated_function_arguments[index].clone()
        }
    }

    fn evaluate_local_variable(
        &self,
        handle: naga::Handle<LocalVariable>,
        state: &FunctionState,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        if let Some(value) = state.local_variables.get(&handle).cloned() {
            return value;
        }

        let local_variable = &state.function.local_variables[handle];

        Value::Pointer(Rc::new(RefCell::new(self.evaluate_expression(
            local_variable.init.unwrap(),
            state_index,
            parent_state_index,
        ))))
    }

    fn evaluate_binary(
        &self,
        op: naga::BinaryOperator,
        left: naga::Handle<Expression>,
        right: naga::Handle<Expression>,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        use naga::BinaryOperator::*;

        let l = self.evaluate_expression(left, state_index, parent_state_index).leaf_value();
        let r = self.evaluate_expression(right, state_index, parent_state_index).leaf_value();

        match (op, l, r) {
            // I32
            (Add,          Value::I32(l), Value::I32(r)) => Value::I32(l.wrapping_add(r)),
            (Subtract,     Value::I32(l), Value::I32(r)) => Value::I32(l.wrapping_sub(r)),
            (Multiply,     Value::I32(l), Value::I32(r)) => Value::I32(l.wrapping_mul(r)),
            (Divide,       Value::I32(l), Value::I32(r)) => Value::I32(l.checked_div(r).unwrap_or(0)),
            (Modulo,       Value::I32(l), Value::I32(r)) => Value::I32(l.checked_rem(r).unwrap_or(0)),
            (And,          Value::I32(l), Value::I32(r)) => Value::I32(l & r),
            (InclusiveOr,  Value::I32(l), Value::I32(r)) => Value::I32(l | r),
            (ExclusiveOr,  Value::I32(l), Value::I32(r)) => Value::I32(l ^ r),
            (ShiftLeft,    Value::I32(l), Value::U32(r)) => Value::I32(l.wrapping_shl(r)),
            (ShiftRight,   Value::I32(l), Value::U32(r)) => Value::I32(l.wrapping_shr(r)),
            (Equal,        Value::I32(l), Value::I32(r)) => Value::U32(u32::from(l == r)),
            (NotEqual,     Value::I32(l), Value::I32(r)) => Value::U32(u32::from(l != r)),
            (Less,         Value::I32(l), Value::I32(r)) => Value::U32(u32::from(l < r)),
            (LessEqual,    Value::I32(l), Value::I32(r)) => Value::U32(u32::from(l <= r)),
            (Greater,      Value::I32(l), Value::I32(r)) => Value::U32(u32::from(l > r)),
            (GreaterEqual, Value::I32(l), Value::I32(r)) => Value::U32(u32::from(l >= r)),

            // U32
            (Add,          Value::U32(l), Value::U32(r)) => Value::U32(l.wrapping_add(r)),
            (Subtract,     Value::U32(l), Value::U32(r)) => Value::U32(l.wrapping_sub(r)),
            (Multiply,     Value::U32(l), Value::U32(r)) => Value::U32(l.wrapping_mul(r)),
            (Divide,       Value::U32(l), Value::U32(r)) => Value::U32(l.checked_div(r).unwrap_or(0)),
            (Modulo,       Value::U32(l), Value::U32(r)) => Value::U32(l.checked_rem(r).unwrap_or(0)),
            (And,          Value::U32(l), Value::U32(r)) => Value::U32(l & r),
            (InclusiveOr,  Value::U32(l), Value::U32(r)) => Value::U32(l | r),
            (ExclusiveOr,  Value::U32(l), Value::U32(r)) => Value::U32(l ^ r),
            (ShiftLeft,    Value::U32(l), Value::U32(r)) => Value::U32(l.wrapping_shl(r)),
            (ShiftRight,   Value::U32(l), Value::U32(r)) => Value::U32(l.wrapping_shr(r)),
            (Equal,        Value::U32(l), Value::U32(r)) => Value::U32(u32::from(l == r)),
            (NotEqual,     Value::U32(l), Value::U32(r)) => Value::U32(u32::from(l != r)),
            (Less,         Value::U32(l), Value::U32(r)) => Value::U32(u32::from(l < r)),
            (LessEqual,    Value::U32(l), Value::U32(r)) => Value::U32(u32::from(l <= r)),
            (Greater,      Value::U32(l), Value::U32(r)) => Value::U32(u32::from(l > r)),
            (GreaterEqual, Value::U32(l), Value::U32(r)) => Value::U32(u32::from(l >= r)),

            // F32
            (Add,          Value::F32(l), Value::F32(r)) => Value::F32(l + r),
            (Subtract,     Value::F32(l), Value::F32(r)) => Value::F32(l - r),
            (Multiply,     Value::F32(l), Value::F32(r)) => Value::F32(l * r),
            (Divide,       Value::F32(l), Value::F32(r)) => Value::F32(l / r),
            (Modulo,       Value::F32(l), Value::F32(r)) => Value::F32(l % r),
            (Equal,        Value::F32(l), Value::F32(r)) => Value::U32(u32::from(l == r)),
            (NotEqual,     Value::F32(l), Value::F32(r)) => Value::U32(u32::from(l != r)),
            (Less,         Value::F32(l), Value::F32(r)) => Value::U32(u32::from(l < r)),
            (LessEqual,    Value::F32(l), Value::F32(r)) => Value::U32(u32::from(l <= r)),
            (Greater,      Value::F32(l), Value::F32(r)) => Value::U32(u32::from(l > r)),
            (GreaterEqual, Value::F32(l), Value::F32(r)) => Value::U32(u32::from(l >= r)),

            // F32 vector OP scalar
            (Add,      v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), Value::F32(s)) => v.map_f32(|x| x + s),
            (Subtract, v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), Value::F32(s)) => v.map_f32(|x| x - s),
            (Multiply, v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), Value::F32(s)) => v.map_f32(|x| x * s),
            (Divide,   v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), Value::F32(s)) => v.map_f32(|x| x / s),
            (Modulo,   v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), Value::F32(s)) => v.map_f32(|x| x % s),

            // F32 scalar OP vector
            (Add,      Value::F32(s), v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => v.map_f32(|x| s + x),
            (Subtract, Value::F32(s), v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => v.map_f32(|x| s - x),
            (Multiply, Value::F32(s), v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => v.map_f32(|x| s * x),
            (Divide,   Value::F32(s), v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => v.map_f32(|x| s / x),
            (Modulo,   Value::F32(s), v @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => v.map_f32(|x| s % x),

            // I32 vector OP scalar (wrapping)
            (Add,      v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::I32(s)) => v.map_i32(|x| x.wrapping_add(s)),
            (Subtract, v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::I32(s)) => v.map_i32(|x| x.wrapping_sub(s)),
            (Multiply, v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::I32(s)) => v.map_i32(|x| x.wrapping_mul(s)),
            (Divide,   v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::I32(s)) => v.map_i32(|x| x.checked_div(s).unwrap_or(0)),
            (Modulo,   v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::I32(s)) => v.map_i32(|x| x.checked_rem(s).unwrap_or(0)),

            // I32 scalar OP vector (wrapping)
            (Add,      Value::I32(s), v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => v.map_i32(|x| s.wrapping_add(x)),
            (Subtract, Value::I32(s), v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => v.map_i32(|x| s.wrapping_sub(x)),
            (Multiply, Value::I32(s), v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => v.map_i32(|x| s.wrapping_mul(x)),
            (Divide,   Value::I32(s), v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => v.map_i32(|x| s.checked_div(x).unwrap_or(0)),
            (Modulo,   Value::I32(s), v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => v.map_i32(|x| s.checked_rem(x).unwrap_or(0)),

            // U32 vector OP scalar (wrapping)
            (Add,      v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x.wrapping_add(s)),
            (Subtract, v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x.wrapping_sub(s)),
            (Multiply, v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x.wrapping_mul(s)),
            (Divide,   v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x.checked_div(s).unwrap_or(0)),
            (Modulo,   v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x.checked_rem(s).unwrap_or(0)),

            // U32 scalar OP vector (wrapping)
            (Add,      Value::U32(s), v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => v.map_u32(|x| s.wrapping_add(x)),
            (Subtract, Value::U32(s), v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => v.map_u32(|x| s.wrapping_sub(x)),
            (Multiply, Value::U32(s), v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => v.map_u32(|x| s.wrapping_mul(x)),
            (Divide,   Value::U32(s), v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => v.map_u32(|x| s.checked_div(x).unwrap_or(0)),
            (Modulo,   Value::U32(s), v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => v.map_u32(|x| s.checked_rem(x).unwrap_or(0)),

            // F64 scalar
            (Add,          Value::F64(l), Value::F64(r)) => Value::F64(l + r),
            (Subtract,     Value::F64(l), Value::F64(r)) => Value::F64(l - r),
            (Multiply,     Value::F64(l), Value::F64(r)) => Value::F64(l * r),
            (Divide,       Value::F64(l), Value::F64(r)) => Value::F64(l / r),
            (Modulo,       Value::F64(l), Value::F64(r)) => Value::F64(l % r),
            (Equal,        Value::F64(l), Value::F64(r)) => Value::U32(u32::from(l == r)),
            (NotEqual,     Value::F64(l), Value::F64(r)) => Value::U32(u32::from(l != r)),
            (Less,         Value::F64(l), Value::F64(r)) => Value::U32(u32::from(l < r)),
            (LessEqual,    Value::F64(l), Value::F64(r)) => Value::U32(u32::from(l <= r)),
            (Greater,      Value::F64(l), Value::F64(r)) => Value::U32(u32::from(l > r)),
            (GreaterEqual, Value::F64(l), Value::F64(r)) => Value::U32(u32::from(l >= r)),

            // I64 scalar
            (Add,          Value::I64(l), Value::I64(r)) => Value::I64(l.wrapping_add(r)),
            (Subtract,     Value::I64(l), Value::I64(r)) => Value::I64(l.wrapping_sub(r)),
            (Multiply,     Value::I64(l), Value::I64(r)) => Value::I64(l.wrapping_mul(r)),
            (Divide,       Value::I64(l), Value::I64(r)) => Value::I64(l.checked_div(r).unwrap_or(0)),
            (Modulo,       Value::I64(l), Value::I64(r)) => Value::I64(l.checked_rem(r).unwrap_or(0)),
            (And,          Value::I64(l), Value::I64(r)) => Value::I64(l & r),
            (InclusiveOr,  Value::I64(l), Value::I64(r)) => Value::I64(l | r),
            (ExclusiveOr,  Value::I64(l), Value::I64(r)) => Value::I64(l ^ r),
            (ShiftLeft,    Value::I64(l), Value::U32(r)) => Value::I64(l.wrapping_shl(r)),
            (ShiftRight,   Value::I64(l), Value::U32(r)) => Value::I64(l.wrapping_shr(r)),
            (Equal,        Value::I64(l), Value::I64(r)) => Value::U32(u32::from(l == r)),
            (NotEqual,     Value::I64(l), Value::I64(r)) => Value::U32(u32::from(l != r)),
            (Less,         Value::I64(l), Value::I64(r)) => Value::U32(u32::from(l < r)),
            (LessEqual,    Value::I64(l), Value::I64(r)) => Value::U32(u32::from(l <= r)),
            (Greater,      Value::I64(l), Value::I64(r)) => Value::U32(u32::from(l > r)),
            (GreaterEqual, Value::I64(l), Value::I64(r)) => Value::U32(u32::from(l >= r)),

            // U64 scalar
            (Add,          Value::U64(l), Value::U64(r)) => Value::U64(l.wrapping_add(r)),
            (Subtract,     Value::U64(l), Value::U64(r)) => Value::U64(l.wrapping_sub(r)),
            (Multiply,     Value::U64(l), Value::U64(r)) => Value::U64(l.wrapping_mul(r)),
            (Divide,       Value::U64(l), Value::U64(r)) => Value::U64(l.checked_div(r).unwrap_or(0)),
            (Modulo,       Value::U64(l), Value::U64(r)) => Value::U64(l.checked_rem(r).unwrap_or(0)),
            (And,          Value::U64(l), Value::U64(r)) => Value::U64(l & r),
            (InclusiveOr,  Value::U64(l), Value::U64(r)) => Value::U64(l | r),
            (ExclusiveOr,  Value::U64(l), Value::U64(r)) => Value::U64(l ^ r),
            (ShiftLeft,    Value::U64(l), Value::U32(r)) => Value::U64(l.wrapping_shl(r)),
            (ShiftRight,   Value::U64(l), Value::U32(r)) => Value::U64(l.wrapping_shr(r)),
            (Equal,        Value::U64(l), Value::U64(r)) => Value::U32(u32::from(l == r)),
            (NotEqual,     Value::U64(l), Value::U64(r)) => Value::U32(u32::from(l != r)),
            (Less,         Value::U64(l), Value::U64(r)) => Value::U32(u32::from(l < r)),
            (LessEqual,    Value::U64(l), Value::U64(r)) => Value::U32(u32::from(l <= r)),
            (Greater,      Value::U64(l), Value::U64(r)) => Value::U32(u32::from(l > r)),
            (GreaterEqual, Value::U64(l), Value::U64(r)) => Value::U32(u32::from(l >= r)),

            // Bool (U32 0/1) logical ops
            (LogicalAnd, Value::U32(l), Value::U32(r)) => Value::U32(u32::from(l != 0 && r != 0)),
            (LogicalOr,  Value::U32(l), Value::U32(r)) => Value::U32(u32::from(l != 0 || r != 0)),

            // I32 vector OP scalar: missing bitwise and shifts
            (And,         v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::I32(s)) => v.map_i32(|x| x & s),
            (InclusiveOr, v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::I32(s)) => v.map_i32(|x| x | s),
            (ExclusiveOr, v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::I32(s)) => v.map_i32(|x| x ^ s),
            (ShiftLeft,   v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::U32(s)) => v.map_i32(|x| x.wrapping_shl(s)),
            (ShiftRight,  v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), Value::U32(s)) => v.map_i32(|x| x.wrapping_shr(s)),

            // I32 scalar OP vector: missing bitwise
            (And,         Value::I32(s), v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => v.map_i32(|x| s & x),
            (InclusiveOr, Value::I32(s), v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => v.map_i32(|x| s | x),
            (ExclusiveOr, Value::I32(s), v @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => v.map_i32(|x| s ^ x),

            // U32 vector OP scalar: missing bitwise and shifts
            (And,         v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x & s),
            (InclusiveOr, v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x | s),
            (ExclusiveOr, v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x ^ s),
            (ShiftLeft,   v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x.wrapping_shl(s)),
            (ShiftRight,  v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), Value::U32(s)) => v.map_u32(|x| x.wrapping_shr(s)),

            // U32 scalar OP vector: missing bitwise
            (And,         Value::U32(s), v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => v.map_u32(|x| s & x),
            (InclusiveOr, Value::U32(s), v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => v.map_u32(|x| s | x),
            (ExclusiveOr, Value::U32(s), v @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => v.map_u32(|x| s ^ x),

            // F32 vector OP F32 vector
            (Add,          l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_map_f32(r, |a, b| a + b),
            (Subtract,     l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_map_f32(r, |a, b| a - b),
            (Multiply,     l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_map_f32(r, |a, b| a * b),
            (Divide,       l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_map_f32(r, |a, b| a / b),
            (Modulo,       l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_map_f32(r, |a, b| a % b),
            (Equal,        l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_cmp_f32(r, |a, b| a == b),
            (NotEqual,     l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_cmp_f32(r, |a, b| a != b),
            (Less,         l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_cmp_f32(r, |a, b| a < b),
            (LessEqual,    l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_cmp_f32(r, |a, b| a <= b),
            (Greater,      l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_cmp_f32(r, |a, b| a > b),
            (GreaterEqual, l @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_)), r @ (Value::F32x2(_) | Value::F32x3(_) | Value::F32x4(_))) => l.zip_cmp_f32(r, |a, b| a >= b),

            // I32 vector OP I32 vector
            (Add,          l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_map_i32(r, |a, b| a.wrapping_add(b)),
            (Subtract,     l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_map_i32(r, |a, b| a.wrapping_sub(b)),
            (Multiply,     l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_map_i32(r, |a, b| a.wrapping_mul(b)),
            (Divide,       l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_map_i32(r, |a, b| a.checked_div(b).unwrap_or(0)),
            (Modulo,       l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_map_i32(r, |a, b| a.checked_rem(b).unwrap_or(0)),
            (And,          l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_map_i32(r, |a, b| a & b),
            (InclusiveOr,  l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_map_i32(r, |a, b| a | b),
            (ExclusiveOr,  l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_map_i32(r, |a, b| a ^ b),
            (Equal,        l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_cmp_i32(r, |a, b| a == b),
            (NotEqual,     l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_cmp_i32(r, |a, b| a != b),
            (Less,         l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_cmp_i32(r, |a, b| a < b),
            (LessEqual,    l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_cmp_i32(r, |a, b| a <= b),
            (Greater,      l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_cmp_i32(r, |a, b| a > b),
            (GreaterEqual, l @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_)), r @ (Value::I32x2(_) | Value::I32x3(_) | Value::I32x4(_))) => l.zip_cmp_i32(r, |a, b| a >= b),

            // I32 vector OP U32 vector: shifts
            (ShiftLeft,  Value::I32x2([l0, l1]), Value::U32x2([r0, r1])) => Value::I32x2([l0.wrapping_shl(r0), l1.wrapping_shl(r1)]),
            (ShiftLeft,  Value::I32x3([l0, l1, l2]), Value::U32x3([r0, r1, r2])) => Value::I32x3([l0.wrapping_shl(r0), l1.wrapping_shl(r1), l2.wrapping_shl(r2)]),
            (ShiftLeft,  Value::I32x4([l0, l1, l2, l3]), Value::U32x4([r0, r1, r2, r3])) => Value::I32x4([l0.wrapping_shl(r0), l1.wrapping_shl(r1), l2.wrapping_shl(r2), l3.wrapping_shl(r3)]),
            (ShiftRight, Value::I32x2([l0, l1]), Value::U32x2([r0, r1])) => Value::I32x2([l0.wrapping_shr(r0), l1.wrapping_shr(r1)]),
            (ShiftRight, Value::I32x3([l0, l1, l2]), Value::U32x3([r0, r1, r2])) => Value::I32x3([l0.wrapping_shr(r0), l1.wrapping_shr(r1), l2.wrapping_shr(r2)]),
            (ShiftRight, Value::I32x4([l0, l1, l2, l3]), Value::U32x4([r0, r1, r2, r3])) => Value::I32x4([l0.wrapping_shr(r0), l1.wrapping_shr(r1), l2.wrapping_shr(r2), l3.wrapping_shr(r3)]),

            // U32 vector OP U32 vector
            (Add,          l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a.wrapping_add(b)),
            (Subtract,     l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a.wrapping_sub(b)),
            (Multiply,     l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a.wrapping_mul(b)),
            (Divide,       l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a.checked_div(b).unwrap_or(0)),
            (Modulo,       l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a.checked_rem(b).unwrap_or(0)),
            (And,          l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a & b),
            (InclusiveOr,  l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a | b),
            (ExclusiveOr,  l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a ^ b),
            (ShiftLeft,    l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a.wrapping_shl(b)),
            (ShiftRight,   l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_map_u32(r, |a, b| a.wrapping_shr(b)),
            (Equal,        l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_cmp_u32(r, |a, b| a == b),
            (NotEqual,     l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_cmp_u32(r, |a, b| a != b),
            (Less,         l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_cmp_u32(r, |a, b| a < b),
            (LessEqual,    l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_cmp_u32(r, |a, b| a <= b),
            (Greater,      l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_cmp_u32(r, |a, b| a > b),
            (GreaterEqual, l @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_)), r @ (Value::U32x2(_) | Value::U32x3(_) | Value::U32x4(_))) => l.zip_cmp_u32(r, |a, b| a >= b),

            _ => Value::Uninitialized,
        }
    }

    fn evaluate_global_variable(&self, handle: naga::Handle<GlobalVariable>) -> Value {
        self.global_values
            .get(&handle)
            .cloned()
            .unwrap_or(Value::Uninitialized)
    }

    fn evaluate_access(
        &self,
        base: naga::Handle<Expression>,
        index: naga::Handle<Expression>,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        let base_value = self
            .evaluate_expression(base, state_index, parent_state_index)
            .leaf_value();
        let index_value = self
            .evaluate_expression(index, state_index, parent_state_index)
            .leaf_value();

        let index: usize = match index_value {
            Value::U32(i) => i as usize,
            Value::I32(i) => i.max(0) as usize,
            _ => return Value::Uninitialized,
        };

        match base_value {
            Value::Array(elements) => elements
                .get(index)
                .map(|e| Value::Pointer(e.clone()))
                .unwrap_or(Value::Uninitialized),
            Value::U32x2(arr) => Value::U32(arr[index]),
            Value::U32x3(arr) => Value::U32(arr[index]),
            Value::U32x4(arr) => Value::U32(arr[index]),
            Value::I32x2(arr) => Value::I32(arr[index]),
            Value::I32x3(arr) => Value::I32(arr[index]),
            Value::I32x4(arr) => Value::I32(arr[index]),
            Value::F32x2(arr) => Value::F32(arr[index]),
            Value::F32x3(arr) => Value::F32(arr[index]),
            Value::F32x4(arr) => Value::F32(arr[index]),
            _ => Value::Uninitialized,
        }
    }

    /// Evaluate an expression from the module's global_expressions arena (used for constants/overrides).
    fn evaluate_global_expression(&self, expr_handle: naga::Handle<Expression>) -> Value {
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
        use naga::{Scalar, ScalarKind};
        let expected_len = match ty_inner {
            TypeInner::Vector { size, .. } => match size {
                VectorSize::Bi => 2,
                VectorSize::Tri => 3,
                VectorSize::Quad => 4,
            },
            TypeInner::Array { .. } => {
                return Value::Array(vals.iter().map(|v| Rc::new(RefCell::new(v.clone()))).collect());
            }
            _ => return Value::Uninitialized,
        };
        match ty_inner {
            TypeInner::Vector { scalar: Scalar { kind: ScalarKind::Float, width: 4 }, .. } => {
                let comps = Value::collect_f32_components(vals);
                if comps.len() >= expected_len { Value::from_f32_slice(&comps[..expected_len]) } else { Value::Uninitialized }
            }
            TypeInner::Vector { scalar: Scalar { kind: ScalarKind::Sint, width: 4 }, .. } => {
                let comps = Value::collect_i32_components(vals);
                if comps.len() >= expected_len { Value::from_i32_slice(&comps[..expected_len]) } else { Value::Uninitialized }
            }
            TypeInner::Vector { scalar: Scalar { kind: ScalarKind::Uint, width: 4 }, .. } => {
                let comps = Value::collect_u32_components(vals);
                if comps.len() >= expected_len { Value::from_u32_slice(&comps[..expected_len]) } else { Value::Uninitialized }
            }
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_compose(
        &self,
        ty: naga::Handle<Type>,
        components: &[naga::Handle<Expression>],
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        let ty_inner = &self.module.types[ty].inner;
        let vals: Vec<Value> = components
            .iter()
            .map(|c| self.evaluate_expression(*c, state_index, parent_state_index).leaf_value())
            .collect();
        self.assemble_compose(ty_inner, &vals)
    }

    /// Splat a scalar value into a vector of the given size.
    fn splat_value(&self, size: VectorSize, val: Value) -> Value {
        match (size, val) {
            (VectorSize::Bi, Value::F32(v)) => Value::F32x2([v; 2]),
            (VectorSize::Tri, Value::F32(v)) => Value::F32x3([v; 3]),
            (VectorSize::Quad, Value::F32(v)) => Value::F32x4([v; 4]),
            (VectorSize::Bi, Value::I32(v)) => Value::I32x2([v; 2]),
            (VectorSize::Tri, Value::I32(v)) => Value::I32x3([v; 3]),
            (VectorSize::Quad, Value::I32(v)) => Value::I32x4([v; 4]),
            (VectorSize::Bi, Value::U32(v)) => Value::U32x2([v; 2]),
            (VectorSize::Tri, Value::U32(v)) => Value::U32x3([v; 3]),
            (VectorSize::Quad, Value::U32(v)) => Value::U32x4([v; 4]),
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_splat(
        &self,
        size: VectorSize,
        value: naga::Handle<Expression>,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        let val = self
            .evaluate_expression(value, state_index, parent_state_index)
            .leaf_value();
        self.splat_value(size, val)
    }

    fn evaluate_swizzle(
        &self,
        size: VectorSize,
        vector: naga::Handle<Expression>,
        pattern: [SwizzleComponent; 4],
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        let vec_val = self
            .evaluate_expression(vector, state_index, parent_state_index)
            .leaf_value();

        let count = match size {
            VectorSize::Bi => 2,
            VectorSize::Tri => 3,
            VectorSize::Quad => 4,
        };

        let components: Vec<Value> = (0..count)
            .map(|i| vec_val.extract_component(pattern[i] as usize))
            .collect();

        // Reconstruct using collect + from_*_slice based on first component's type
        match &components[0] {
            Value::F32(_) => Value::from_f32_slice(&Value::collect_f32_components(&components)),
            Value::I32(_) => Value::from_i32_slice(&Value::collect_i32_components(&components)),
            Value::U32(_) => Value::from_u32_slice(&Value::collect_u32_components(&components)),
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_unary(&self, op: UnaryOperator, val: Value) -> Value {
        match op {
            UnaryOperator::Negate => {
                // Handle F64/I64 scalars separately (not in map_numeric)
                match val {
                    Value::F64(v) => Value::F64(-v),
                    Value::I64(v) => Value::I64(v.wrapping_neg()),
                    _ => val.map_numeric(|f| -f, |i| i.wrapping_neg(), |_| 0), // U32 negate not valid in WGSL
                }
            }
            UnaryOperator::LogicalNot => {
                // Bool stored as U32 0/1
                match val {
                    Value::U32(v) => Value::U32(u32::from(v == 0)),
                    _ => Value::Uninitialized,
                }
            }
            UnaryOperator::BitwiseNot => {
                match val {
                    Value::I64(v) => Value::I64(!v),
                    Value::U64(v) => Value::U64(!v),
                    _ => val.map_numeric(|_| 0.0, |i| !i, |u| !u), // F32 bitwise not not valid in WGSL
                }
            }
        }
    }

    fn evaluate_select(
        &self,
        condition: naga::Handle<Expression>,
        accept: naga::Handle<Expression>,
        reject: naga::Handle<Expression>,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        let cond = self
            .evaluate_expression(condition, state_index, parent_state_index)
            .leaf_value();

        let is_true = match cond {
            Value::U32(v) => v != 0,
            _ => false,
        };

        if is_true {
            self.evaluate_expression(accept, state_index, parent_state_index)
        } else {
            self.evaluate_expression(reject, state_index, parent_state_index)
        }
    }

    /// Cast/bitcast a scalar or vector value to a different scalar kind.
    fn evaluate_as(&self, val: Value, kind: naga::ScalarKind, convert: Option<naga::Bytes>) -> Value {
        use naga::ScalarKind::*;

        let val = val.leaf_value();

        match convert {
            // Conversion cast (Some(width))
            Some(4) => match (kind, val) {
                // To F32
                (Float, Value::F32(v)) => Value::F32(v),
                (Float, Value::I32(v)) => Value::F32(v as f32),
                (Float, Value::U32(v)) => Value::F32(v as f32),
                (Float, Value::F64(v)) => Value::F32(v as f32),
                (Float, Value::I64(v)) => Value::F32(v as f32),
                (Float, Value::U64(v)) => Value::F32(v as f32),
                // To I32
                (Sint, Value::I32(v)) => Value::I32(v),
                (Sint, Value::F32(v)) => Value::I32(v as i32),
                (Sint, Value::U32(v)) => Value::I32(v as i32),
                (Sint, Value::F64(v)) => Value::I32(v as i32),
                (Sint, Value::I64(v)) => Value::I32(v as i32),
                (Sint, Value::U64(v)) => Value::I32(v as i32),
                // To U32
                (Uint, Value::U32(v)) => Value::U32(v),
                (Uint, Value::F32(v)) => Value::U32(v as u32),
                (Uint, Value::I32(v)) => Value::U32(v as u32),
                (Uint, Value::F64(v)) => Value::U32(v as u32),
                (Uint, Value::I64(v)) => Value::U32(v as u32),
                (Uint, Value::U64(v)) => Value::U32(v as u32),
                // To Bool (U32 0/1)
                (Bool, Value::U32(v)) => Value::U32(u32::from(v != 0)),
                (Bool, Value::I32(v)) => Value::U32(u32::from(v != 0)),
                (Bool, Value::F32(v)) => Value::U32(u32::from(v != 0.0)),
                // Vector conversions to F32xN
                (Float, Value::I32x2([a, b])) => Value::F32x2([a as f32, b as f32]),
                (Float, Value::I32x3([a, b, c])) => Value::F32x3([a as f32, b as f32, c as f32]),
                (Float, Value::I32x4([a, b, c, d])) => Value::F32x4([a as f32, b as f32, c as f32, d as f32]),
                (Float, Value::U32x2([a, b])) => Value::F32x2([a as f32, b as f32]),
                (Float, Value::U32x3([a, b, c])) => Value::F32x3([a as f32, b as f32, c as f32]),
                (Float, Value::U32x4([a, b, c, d])) => Value::F32x4([a as f32, b as f32, c as f32, d as f32]),
                // Vector conversions to I32xN
                (Sint, Value::F32x2([a, b])) => Value::I32x2([a as i32, b as i32]),
                (Sint, Value::F32x3([a, b, c])) => Value::I32x3([a as i32, b as i32, c as i32]),
                (Sint, Value::F32x4([a, b, c, d])) => Value::I32x4([a as i32, b as i32, c as i32, d as i32]),
                (Sint, Value::U32x2([a, b])) => Value::I32x2([a as i32, b as i32]),
                (Sint, Value::U32x3([a, b, c])) => Value::I32x3([a as i32, b as i32, c as i32]),
                (Sint, Value::U32x4([a, b, c, d])) => Value::I32x4([a as i32, b as i32, c as i32, d as i32]),
                // Vector conversions to U32xN
                (Uint, Value::F32x2([a, b])) => Value::U32x2([a as u32, b as u32]),
                (Uint, Value::F32x3([a, b, c])) => Value::U32x3([a as u32, b as u32, c as u32]),
                (Uint, Value::F32x4([a, b, c, d])) => Value::U32x4([a as u32, b as u32, c as u32, d as u32]),
                (Uint, Value::I32x2([a, b])) => Value::U32x2([a as u32, b as u32]),
                (Uint, Value::I32x3([a, b, c])) => Value::U32x3([a as u32, b as u32, c as u32]),
                (Uint, Value::I32x4([a, b, c, d])) => Value::U32x4([a as u32, b as u32, c as u32, d as u32]),
                _ => Value::Uninitialized,
            },
            Some(8) => match (kind, val) {
                // To F64
                (Float, Value::F32(v)) => Value::F64(v as f64),
                (Float, Value::F64(v)) => Value::F64(v),
                (Float, Value::I32(v)) => Value::F64(v as f64),
                (Float, Value::U32(v)) => Value::F64(v as f64),
                (Float, Value::I64(v)) => Value::F64(v as f64),
                (Float, Value::U64(v)) => Value::F64(v as f64),
                // To I64
                (Sint, Value::I32(v)) => Value::I64(v as i64),
                (Sint, Value::U32(v)) => Value::I64(v as i64),
                (Sint, Value::I64(v)) => Value::I64(v),
                (Sint, Value::U64(v)) => Value::I64(v as i64),
                (Sint, Value::F32(v)) => Value::I64(v as i64),
                (Sint, Value::F64(v)) => Value::I64(v as i64),
                // To U64
                (Uint, Value::I32(v)) => Value::U64(v as u64),
                (Uint, Value::U32(v)) => Value::U64(v as u64),
                (Uint, Value::I64(v)) => Value::U64(v as u64),
                (Uint, Value::U64(v)) => Value::U64(v),
                (Uint, Value::F32(v)) => Value::U64(v as u64),
                (Uint, Value::F64(v)) => Value::U64(v as u64),
                _ => Value::Uninitialized,
            },
            // Bitcast (None) — reinterpret bits
            None => match (kind, val) {
                (Float, Value::I32(v)) => Value::F32(f32::from_bits(v as u32)),
                (Float, Value::U32(v)) => Value::F32(f32::from_bits(v)),
                (Sint, Value::F32(v)) => Value::I32(v.to_bits() as i32),
                (Sint, Value::U32(v)) => Value::I32(v as i32),
                (Uint, Value::F32(v)) => Value::U32(v.to_bits()),
                (Uint, Value::I32(v)) => Value::U32(v as u32),
                // Vector bitcasts
                (Float, Value::I32x2([a, b])) => Value::F32x2([f32::from_bits(a as u32), f32::from_bits(b as u32)]),
                (Float, Value::I32x3([a, b, c])) => Value::F32x3([f32::from_bits(a as u32), f32::from_bits(b as u32), f32::from_bits(c as u32)]),
                (Float, Value::I32x4([a, b, c, d])) => Value::F32x4([f32::from_bits(a as u32), f32::from_bits(b as u32), f32::from_bits(c as u32), f32::from_bits(d as u32)]),
                (Float, Value::U32x2([a, b])) => Value::F32x2([f32::from_bits(a), f32::from_bits(b)]),
                (Float, Value::U32x3([a, b, c])) => Value::F32x3([f32::from_bits(a), f32::from_bits(b), f32::from_bits(c)]),
                (Float, Value::U32x4([a, b, c, d])) => Value::F32x4([f32::from_bits(a), f32::from_bits(b), f32::from_bits(c), f32::from_bits(d)]),
                (Sint, Value::F32x2([a, b])) => Value::I32x2([a.to_bits() as i32, b.to_bits() as i32]),
                (Sint, Value::F32x3([a, b, c])) => Value::I32x3([a.to_bits() as i32, b.to_bits() as i32, c.to_bits() as i32]),
                (Sint, Value::F32x4([a, b, c, d])) => Value::I32x4([a.to_bits() as i32, b.to_bits() as i32, c.to_bits() as i32, d.to_bits() as i32]),
                (Sint, Value::U32x2([a, b])) => Value::I32x2([a as i32, b as i32]),
                (Sint, Value::U32x3([a, b, c])) => Value::I32x3([a as i32, b as i32, c as i32]),
                (Sint, Value::U32x4([a, b, c, d])) => Value::I32x4([a as i32, b as i32, c as i32, d as i32]),
                (Uint, Value::F32x2([a, b])) => Value::U32x2([a.to_bits(), b.to_bits()]),
                (Uint, Value::F32x3([a, b, c])) => Value::U32x3([a.to_bits(), b.to_bits(), c.to_bits()]),
                (Uint, Value::F32x4([a, b, c, d])) => Value::U32x4([a.to_bits(), b.to_bits(), c.to_bits(), d.to_bits()]),
                (Uint, Value::I32x2([a, b])) => Value::U32x2([a as u32, b as u32]),
                (Uint, Value::I32x3([a, b, c])) => Value::U32x3([a as u32, b as u32, c as u32]),
                (Uint, Value::I32x4([a, b, c, d])) => Value::U32x4([a as u32, b as u32, c as u32, d as u32]),
                _ => Value::Uninitialized,
            },
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_math(
        &self,
        fun: MathFunction,
        arg: naga::Handle<Expression>,
        arg1: Option<naga::Handle<Expression>>,
        arg2: Option<naga::Handle<Expression>>,
        arg3: Option<naga::Handle<Expression>>,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) -> Value {
        let a = self.evaluate_expression(arg, state_index, parent_state_index).leaf_value();
        let b = arg1.map(|h| self.evaluate_expression(h, state_index, parent_state_index).leaf_value());
        let c = arg2.map(|h| self.evaluate_expression(h, state_index, parent_state_index).leaf_value());
        let _d = arg3.map(|h| self.evaluate_expression(h, state_index, parent_state_index).leaf_value());

        match fun {
            // --- Comparison ---
            MathFunction::Abs => self.math_unary_float_or_int(a, f32::abs, f64::abs, |v: i32| v.wrapping_abs(), |v: u32| v),
            MathFunction::Min => self.math_binary_float_or_int(a, b.unwrap_or(Value::Uninitialized), f32::min, f64::min, i32::min, u32::min),
            MathFunction::Max => self.math_binary_float_or_int(a, b.unwrap_or(Value::Uninitialized), f32::max, f64::max, i32::max, u32::max),
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
            MathFunction::Atan2 => self.math_binary_f32(a, b.unwrap_or(Value::Uninitialized), f32::atan2),
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
            MathFunction::Ldexp => self.math_binary_f32(a, b.unwrap_or(Value::Uninitialized), |base, exp| {
                base * 2.0f32.powi(exp as i32)
            }),

            // --- Exponent ---
            MathFunction::Exp => self.math_unary_f32(a, f32::exp),
            MathFunction::Exp2 => self.math_unary_f32(a, f32::exp2),
            MathFunction::Log => self.math_unary_f32(a, f32::ln),
            MathFunction::Log2 => self.math_unary_f32(a, f32::log2),
            MathFunction::Pow => self.math_binary_f32(a, b.unwrap_or(Value::Uninitialized), f32::powf),

            // --- Geometry ---
            MathFunction::Dot => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                match (a, b_val) {
                    (Value::F32x2([a0, a1]), Value::F32x2([b0, b1])) => Value::F32(a0 * b0 + a1 * b1),
                    (Value::F32x3([a0, a1, a2]), Value::F32x3([b0, b1, b2])) => Value::F32(a0 * b0 + a1 * b1 + a2 * b2),
                    (Value::F32x4([a0, a1, a2, a3]), Value::F32x4([b0, b1, b2, b3])) => Value::F32(a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3),
                    (Value::I32x2([a0, a1]), Value::I32x2([b0, b1])) => Value::I32(a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1))),
                    (Value::I32x3([a0, a1, a2]), Value::I32x3([b0, b1, b2])) => Value::I32(a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1)).wrapping_add(a2.wrapping_mul(b2))),
                    (Value::I32x4([a0, a1, a2, a3]), Value::I32x4([b0, b1, b2, b3])) => Value::I32(a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1)).wrapping_add(a2.wrapping_mul(b2)).wrapping_add(a3.wrapping_mul(b3))),
                    (Value::U32x2([a0, a1]), Value::U32x2([b0, b1])) => Value::U32(a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1))),
                    (Value::U32x3([a0, a1, a2]), Value::U32x3([b0, b1, b2])) => Value::U32(a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1)).wrapping_add(a2.wrapping_mul(b2))),
                    (Value::U32x4([a0, a1, a2, a3]), Value::U32x4([b0, b1, b2, b3])) => Value::U32(a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1)).wrapping_add(a2.wrapping_mul(b2)).wrapping_add(a3.wrapping_mul(b3))),
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Cross => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                match (a, b_val) {
                    (Value::F32x3([a0, a1, a2]), Value::F32x3([b0, b1, b2])) => {
                        Value::F32x3([
                            a1 * b2 - a2 * b1,
                            a2 * b0 - a0 * b2,
                            a0 * b1 - a1 * b0,
                        ])
                    }
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Distance => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                match (a, b_val) {
                    (Value::F32(a), Value::F32(b)) => Value::F32((a - b).abs()),
                    (Value::F32x2([a0, a1]), Value::F32x2([b0, b1])) => {
                        let d0 = a0 - b0;
                        let d1 = a1 - b1;
                        Value::F32((d0 * d0 + d1 * d1).sqrt())
                    }
                    (Value::F32x3([a0, a1, a2]), Value::F32x3([b0, b1, b2])) => {
                        let d0 = a0 - b0;
                        let d1 = a1 - b1;
                        let d2 = a2 - b2;
                        Value::F32((d0 * d0 + d1 * d1 + d2 * d2).sqrt())
                    }
                    (Value::F32x4([a0, a1, a2, a3]), Value::F32x4([b0, b1, b2, b3])) => {
                        let d0 = a0 - b0;
                        let d1 = a1 - b1;
                        let d2 = a2 - b2;
                        let d3 = a3 - b3;
                        Value::F32((d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3).sqrt())
                    }
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Length => match a {
                Value::F32(v) => Value::F32(v.abs()),
                Value::F32x2([x, y]) => Value::F32((x * x + y * y).sqrt()),
                Value::F32x3([x, y, z]) => Value::F32((x * x + y * y + z * z).sqrt()),
                Value::F32x4([x, y, z, w]) => Value::F32((x * x + y * y + z * z + w * w).sqrt()),
                _ => Value::Uninitialized,
            },
            MathFunction::Normalize => match a {
                Value::F32x2([x, y]) => {
                    let len = (x * x + y * y).sqrt();
                    if len != 0.0 { Value::F32x2([x / len, y / len]) } else { Value::F32x2([0.0; 2]) }
                }
                Value::F32x3([x, y, z]) => {
                    let len = (x * x + y * y + z * z).sqrt();
                    if len != 0.0 { Value::F32x3([x / len, y / len, z / len]) } else { Value::F32x3([0.0; 3]) }
                }
                Value::F32x4([x, y, z, w]) => {
                    let len = (x * x + y * y + z * z + w * w).sqrt();
                    if len != 0.0 { Value::F32x4([x / len, y / len, z / len, w / len]) } else { Value::F32x4([0.0; 4]) }
                }
                _ => Value::Uninitialized,
            },
            MathFunction::FaceForward => {
                // faceForward(e1, e2, e3) = select(-e1, e1, dot(e2, e3) < 0)
                // e1=arg, e2=arg1, e3=arg2
                // We need dot(e2, e3), so pass arg1 as the main arg and arg2 as the second
                let dot_val = if let (Some(e2), Some(e3)) = (arg1, arg2) {
                    self.evaluate_math(MathFunction::Dot, e2, Some(e3), None, None, state_index, parent_state_index)
                } else {
                    Value::Uninitialized
                };
                let dot_negative = match dot_val {
                    Value::F32(v) => v < 0.0,
                    _ => false,
                };
                if dot_negative { a } else { self.evaluate_unary(UnaryOperator::Negate, a) }
            }
            MathFunction::Reflect => {
                // reflect(e1, e2) = e1 - 2 * dot(e2, e1) * e2
                let b_val = b.unwrap_or(Value::Uninitialized);
                let dot = match (&a, &b_val) {
                    (Value::F32x2([a0, a1]), Value::F32x2([b0, b1])) => b0 * a0 + b1 * a1,
                    (Value::F32x3([a0, a1, a2]), Value::F32x3([b0, b1, b2])) => b0 * a0 + b1 * a1 + b2 * a2,
                    (Value::F32x4([a0, a1, a2, a3]), Value::F32x4([b0, b1, b2, b3])) => b0 * a0 + b1 * a1 + b2 * a2 + b3 * a3,
                    _ => return Value::Uninitialized,
                };
                let factor = 2.0 * dot;
                let scaled_n = b_val.map_f32(|x| x * factor);
                a.zip_map_f32(scaled_n, |e, s| e - s)
            }

            // --- Computational ---
            MathFunction::Sign => self.math_unary_float_or_int(a, |v: f32| v.signum(), |v: f64| v.signum(), |v: i32| v.signum(), |v: u32| if v > 0 { 1 } else { 0 }),
            MathFunction::Fma => {
                // fma(a, b, c) = a * b + c
                let b_val = b.unwrap_or(Value::Uninitialized);
                let c_val = c.unwrap_or(Value::Uninitialized);
                match (a, b_val, c_val) {
                    (Value::F32(a), Value::F32(b), Value::F32(c)) => Value::F32(a.mul_add(b, c)),
                    (Value::F32x2([a0, a1]), Value::F32x2([b0, b1]), Value::F32x2([c0, c1])) => Value::F32x2([a0.mul_add(b0, c0), a1.mul_add(b1, c1)]),
                    (Value::F32x3([a0, a1, a2]), Value::F32x3([b0, b1, b2]), Value::F32x3([c0, c1, c2])) => Value::F32x3([a0.mul_add(b0, c0), a1.mul_add(b1, c1), a2.mul_add(b2, c2)]),
                    (Value::F32x4([a0, a1, a2, a3]), Value::F32x4([b0, b1, b2, b3]), Value::F32x4([c0, c1, c2, c3])) => Value::F32x4([a0.mul_add(b0, c0), a1.mul_add(b1, c1), a2.mul_add(b2, c2), a3.mul_add(b3, c3)]),
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Mix => {
                // mix(a, b, t) = a * (1 - t) + b * t
                let b_val = b.unwrap_or(Value::Uninitialized);
                let c_val = c.unwrap_or(Value::Uninitialized);
                match (a, b_val, c_val) {
                    (Value::F32(a), Value::F32(b), Value::F32(t)) => Value::F32(a * (1.0 - t) + b * t),
                    (Value::F32x2([a0, a1]), Value::F32x2([b0, b1]), Value::F32x2([t0, t1])) => Value::F32x2([a0 * (1.0 - t0) + b0 * t0, a1 * (1.0 - t1) + b1 * t1]),
                    (Value::F32x3([a0, a1, a2]), Value::F32x3([b0, b1, b2]), Value::F32x3([t0, t1, t2])) => Value::F32x3([a0 * (1.0 - t0) + b0 * t0, a1 * (1.0 - t1) + b1 * t1, a2 * (1.0 - t2) + b2 * t2]),
                    (Value::F32x4([a0, a1, a2, a3]), Value::F32x4([b0, b1, b2, b3]), Value::F32x4([t0, t1, t2, t3])) => Value::F32x4([a0 * (1.0 - t0) + b0 * t0, a1 * (1.0 - t1) + b1 * t1, a2 * (1.0 - t2) + b2 * t2, a3 * (1.0 - t3) + b3 * t3]),
                    // mix with scalar t
                    (Value::F32x2([a0, a1]), Value::F32x2([b0, b1]), Value::F32(t)) => Value::F32x2([a0 * (1.0 - t) + b0 * t, a1 * (1.0 - t) + b1 * t]),
                    (Value::F32x3([a0, a1, a2]), Value::F32x3([b0, b1, b2]), Value::F32(t)) => Value::F32x3([a0 * (1.0 - t) + b0 * t, a1 * (1.0 - t) + b1 * t, a2 * (1.0 - t) + b2 * t]),
                    (Value::F32x4([a0, a1, a2, a3]), Value::F32x4([b0, b1, b2, b3]), Value::F32(t)) => Value::F32x4([a0 * (1.0 - t) + b0 * t, a1 * (1.0 - t) + b1 * t, a2 * (1.0 - t) + b2 * t, a3 * (1.0 - t) + b3 * t]),
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::Step => {
                // step(edge, x) = if x < edge { 0.0 } else { 1.0 }
                self.math_binary_f32(a, b.unwrap_or(Value::Uninitialized), |edge, x| if x < edge { 0.0 } else { 1.0 })
            }
            MathFunction::SmoothStep => {
                // smoothstep(low, high, x) = t*t*(3 - 2*t) where t = clamp((x-low)/(high-low), 0, 1)
                let b_val = b.unwrap_or(Value::Uninitialized);
                let c_val = c.unwrap_or(Value::Uninitialized);
                match (a, b_val, c_val) {
                    (Value::F32(low), Value::F32(high), Value::F32(x)) => {
                        let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
                        Value::F32(t * t * (3.0 - 2.0 * t))
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
            MathFunction::CountTrailingZeros => self.math_unary_int(a, |v: i32| v.trailing_zeros() as i32, |v: u32| v.trailing_zeros()),
            MathFunction::CountLeadingZeros => self.math_unary_int(a, |v: i32| v.leading_zeros() as i32, |v: u32| v.leading_zeros()),
            MathFunction::CountOneBits => self.math_unary_int(a, |v: i32| v.count_ones() as i32, |v: u32| v.count_ones()),
            MathFunction::ReverseBits => self.math_unary_int(a, |v: i32| v.reverse_bits(), |v: u32| v.reverse_bits()),
            MathFunction::FirstTrailingBit => self.math_unary_int(a,
                |v: i32| if v == 0 { -1i32 } else { v.trailing_zeros() as i32 },
                |v: u32| if v == 0 { 0xFFFFFFFF } else { v.trailing_zeros() },
            ),
            MathFunction::FirstLeadingBit => self.math_unary_int(a,
                |v: i32| {
                    if v == 0 || v == -1 { -1i32 }
                    else if v > 0 { 31 - v.leading_zeros() as i32 }
                    else { 31 - (!v).leading_zeros() as i32 }
                },
                |v: u32| if v == 0 { 0xFFFFFFFF } else { 31 - v.leading_zeros() },
            ),
            MathFunction::ExtractBits => {
                // extractBits(e, offset, count)
                let offset = match b.unwrap_or(Value::Uninitialized) {
                    Value::U32(v) => v,
                    _ => return Value::Uninitialized,
                };
                let count = match c.unwrap_or(Value::Uninitialized) {
                    Value::U32(v) => v,
                    _ => return Value::Uninitialized,
                };
                match a {
                    Value::I32(v) => {
                        if count == 0 { Value::I32(0) }
                        else {
                            let shifted = (v as u32).wrapping_shr(offset);
                            let mask = if count >= 32 { u32::MAX } else { (1u32 << count) - 1 };
                            let extracted = shifted & mask;
                            // Sign extend
                            let sign_bit = 1u32 << (count - 1);
                            let sign_extended = if extracted & sign_bit != 0 {
                                extracted | !mask
                            } else {
                                extracted
                            };
                            Value::I32(sign_extended as i32)
                        }
                    }
                    Value::U32(v) => {
                        if count == 0 { Value::U32(0) }
                        else {
                            let shifted = v.wrapping_shr(offset);
                            let mask = if count >= 32 { u32::MAX } else { (1u32 << count) - 1 };
                            Value::U32(shifted & mask)
                        }
                    }
                    _ => Value::Uninitialized,
                }
            }
            MathFunction::InsertBits => {
                // insertBits(e, newbits, offset, count)
                let newbits = match b.unwrap_or(Value::Uninitialized) {
                    Value::I32(v) => v as u32,
                    Value::U32(v) => v,
                    _ => return Value::Uninitialized,
                };
                let offset = match c.unwrap_or(Value::Uninitialized) {
                    Value::U32(v) => v,
                    _ => return Value::Uninitialized,
                };
                let count = match _d.unwrap_or(Value::Uninitialized) {
                    Value::U32(v) => v,
                    _ => return Value::Uninitialized,
                };
                match a {
                    Value::I32(v) => {
                        let mask = if count >= 32 { u32::MAX } else { ((1u32 << count) - 1) << offset };
                        let result = ((v as u32) & !mask) | ((newbits << offset) & mask);
                        Value::I32(result as i32)
                    }
                    Value::U32(v) => {
                        let mask = if count >= 32 { u32::MAX } else { ((1u32 << count) - 1) << offset };
                        let result = (v & !mask) | ((newbits << offset) & mask);
                        Value::U32(result)
                    }
                    _ => Value::Uninitialized,
                }
            }

            // Unsupported / GPU-specific functions return Uninitialized
            _ => Value::Uninitialized,
        }
    }

    // --- Math helper methods ---
    // These delegate to Value's generic component-wise operations.

    /// Apply a unary f32 function to a scalar or vector.
    fn math_unary_f32(&self, val: Value, f: impl Fn(f32) -> f32) -> Value {
        val.map_f32(f)
    }

    /// Apply a binary f32 function to scalars or vectors.
    fn math_binary_f32(&self, a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Value {
        a.zip_map_f32(b, f)
    }

    /// Apply a unary function that works on both float and int types.
    /// F64 is handled as a special case since it's not in map_numeric.
    fn math_unary_float_or_int(
        &self,
        val: Value,
        ff32: impl Fn(f32) -> f32,
        ff64: impl Fn(f64) -> f64,
        fi32: impl Fn(i32) -> i32,
        fu32: impl Fn(u32) -> u32,
    ) -> Value {
        match val {
            Value::F64(v) => Value::F64(ff64(v)),
            _ => val.map_numeric(ff32, fi32, fu32),
        }
    }

    /// Apply a binary function that works on both float and int types.
    /// F64 is handled as a special case since it's not in zip_map_numeric.
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
            (Value::F64(av), Value::F64(bv)) => Value::F64(ff64(*av, *bv)),
            _ => a.zip_map_numeric(b, ff32, fi32, fu32),
        }
    }

    /// Apply a unary integer function to i32 or u32 scalars/vectors.
    fn math_unary_int(
        &self,
        val: Value,
        fi32: impl Fn(i32) -> i32,
        fu32: impl Fn(u32) -> u32,
    ) -> Value {
        // Use map_numeric with a no-op for f32 (will return Uninitialized for f32 inputs
        // since int functions shouldn't be called on floats, but map_numeric handles all types)
        val.map_numeric(|_| 0.0, fi32, fu32)
    }

    /// Clamp a value between min and max.
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
                if let Some(ScalarComponents::U32(comps)) = val.to_components() {
                    Value::U32(u32::from(comps.iter().all(|&v| v != 0)))
                } else {
                    Value::Uninitialized
                }
            }
            RelationalFunction::Any => {
                if let Some(ScalarComponents::U32(comps)) = val.to_components() {
                    Value::U32(u32::from(comps.iter().any(|&v| v != 0)))
                } else {
                    Value::Uninitialized
                }
            }
            RelationalFunction::IsNan => {
                if let Some(ScalarComponents::F32(comps)) = val.to_components() {
                    let result: Vec<u32> = comps.iter().map(|v| u32::from(v.is_nan())).collect();
                    Value::from_u32_slice(&result)
                } else {
                    Value::Uninitialized
                }
            }
            RelationalFunction::IsInf => {
                if let Some(ScalarComponents::F32(comps)) = val.to_components() {
                    let result: Vec<u32> = comps.iter().map(|v| u32::from(v.is_infinite())).collect();
                    Value::from_u32_slice(&result)
                } else {
                    Value::Uninitialized
                }
            }
        }
    }
}
