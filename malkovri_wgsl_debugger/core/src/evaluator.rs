use crate::{
    function_state::{FunctionState, NextStatement},
    primitive::Primitive,
    value::Value,
};

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use naga::{
    Expression, GlobalVariable, Literal, LocalVariable, MathFunction, Module, RelationalFunction,
    Statement, SwizzleComponent, Type, TypeInner, UnaryOperator, VectorSize,
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
        let parent_state_index = if state_index > 0 {
            Some(state_index - 1)
        } else {
            None
        };
        let state = &self.stack[state_index];
        state
            .function
            .named_expressions
            .iter()
            .map(|(handle, name)| {
                (
                    name.clone(),
                    self.evaluate_expression(*handle, state_index, parent_state_index),
                )
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

            self.handle_statement(current_statement, current_state_index, parent_state_index);

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

    fn initialize_local_variables(
        &mut self,
        state_index: usize,
        parent_state_index: Option<usize>,
    ) {
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
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => self.evaluate_swizzle(
                *size,
                *vector,
                pattern.clone(),
                state_index,
                parent_state_index,
            ),
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
                let argument = self
                    .evaluate_expression(*expr, state_index, parent_state_index)
                    .leaf_value();
                match argument {
                    Value::Array(elements) => Primitive::U32(elements.len() as u32).into(),
                    _ => panic!("ArrayLength applied to non-array value: {:?}", argument),
                }
            }
            Expression::GlobalVariable(handle) => self.evaluate_global_variable(*handle),
            Expression::Access { base, index } => {
                self.evaluate_access(*base, *index, state_index, parent_state_index)
            }
            Expression::CallResult(_handle) => {
                state.returns.clone().unwrap_or(Value::Uninitialized)
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
        let value = self
            .evaluate_expression(base, state_index, parent_state_index)
            .leaf_value();
        let index = index as usize;

        match value {
            Value::Array(elements) => elements.get(index).cloned().unwrap_or(Value::Uninitialized),
            Value::Struct(fields) => fields
                .get(index)
                .map(|(_, v)| v.clone())
                .unwrap_or(Value::Uninitialized),
            Value::Primitive(p) => Value::Primitive(p.extract_component(index)),
            _ => Value::Uninitialized,
        }
    }

    fn evaluate_function_argument(&self, index: usize, state: &FunctionState) -> Value {
        let function_argument = &state.function.arguments[index];

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

        let l = self
            .evaluate_expression(left, state_index, parent_state_index)
            .leaf_value();
        let r = self
            .evaluate_expression(right, state_index, parent_state_index)
            .leaf_value();

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
                        if let (Some(a), Some(b)) = (lp.as_f32_slice(), rp.as_f32_slice()) {
                            if let Some(res) = cmp_slices(cmp, a, b) {
                                return Value::from(Primitive::from(res.as_slice()));
                            }
                        }
                        if let (Some(a), Some(b)) = (lp.as_i32_slice(), rp.as_i32_slice()) {
                            if let Some(res) = cmp_slices(cmp, a, b) {
                                return Value::from(Primitive::from(res.as_slice()));
                            }
                        }
                        if let (Some(a), Some(b)) = (lp.as_u32_slice(), rp.as_u32_slice()) {
                            if let Some(res) = cmp_slices(cmp, a, b) {
                                return Value::from(Primitive::from(res.as_slice()));
                            }
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
            Value::Primitive(Primitive::U32(i)) => i as usize,
            Value::Primitive(Primitive::I32(i)) => i.max(0) as usize,
            _ => return Value::Uninitialized,
        };

        match base_value {
            Value::Array(elements) => elements.get(index).cloned().unwrap_or(Value::Uninitialized),
            Value::Struct(fields) => fields
                .get(index)
                .map(|(_, v)| v.clone())
                .unwrap_or(Value::Uninitialized),
            Value::Primitive(p) => Value::Primitive(p.extract_component(index)),
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
            TypeInner::Vector { size, .. } => {
                let expected_len = match size {
                    VectorSize::Bi => 2,
                    VectorSize::Tri => 3,
                    VectorSize::Quad => 4,
                };
                match ty_inner {
                    TypeInner::Vector {
                        scalar:
                            Scalar {
                                kind: ScalarKind::Float,
                                width: 4,
                            },
                        ..
                    } => {
                        let comps = Value::collect_f32_components(vals);
                        if comps.len() >= expected_len {
                            Value::from(Primitive::from(&comps[..expected_len]))
                        } else {
                            Value::Uninitialized
                        }
                    }
                    TypeInner::Vector {
                        scalar:
                            Scalar {
                                kind: ScalarKind::Sint,
                                width: 4,
                            },
                        ..
                    } => {
                        let comps = Value::collect_i32_components(vals);
                        if comps.len() >= expected_len {
                            Value::from(Primitive::from(&comps[..expected_len]))
                        } else {
                            Value::Uninitialized
                        }
                    }
                    TypeInner::Vector {
                        scalar:
                            Scalar {
                                kind: ScalarKind::Uint,
                                width: 4,
                            },
                        ..
                    } => {
                        let comps = Value::collect_u32_components(vals);
                        if comps.len() >= expected_len {
                            Value::from(Primitive::from(&comps[..expected_len]))
                        } else {
                            Value::Uninitialized
                        }
                    }
                    _ => Value::Uninitialized,
                }
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
            .map(|c| {
                self.evaluate_expression(*c, state_index, parent_state_index)
                    .leaf_value()
            })
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
            Value::Primitive(Primitive::U32(v)) => v != 0,
            _ => false,
        };

        if is_true {
            self.evaluate_expression(accept, state_index, parent_state_index)
        } else {
            self.evaluate_expression(reject, state_index, parent_state_index)
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
            Some(4) => match (kind, p) {
                // To F32
                (Float, F32(v)) => F32(v).into(),
                (Float, I32(v)) => F32(v as f32).into(),
                (Float, U32(v)) => F32(v as f32).into(),
                (Float, F64(v)) => F32(v as f32).into(),
                (Float, I64(v)) => F32(v as f32).into(),
                (Float, U64(v)) => F32(v as f32).into(),
                // To I32
                (Sint, I32(v)) => I32(v).into(),
                (Sint, F32(v)) => I32(v as i32).into(),
                (Sint, U32(v)) => I32(v as i32).into(),
                (Sint, F64(v)) => I32(v as i32).into(),
                (Sint, I64(v)) => I32(v as i32).into(),
                (Sint, U64(v)) => I32(v as i32).into(),
                // To U32
                (Uint, U32(v)) => U32(v).into(),
                (Uint, F32(v)) => U32(v as u32).into(),
                (Uint, I32(v)) => U32(v as u32).into(),
                (Uint, F64(v)) => U32(v as u32).into(),
                (Uint, I64(v)) => U32(v as u32).into(),
                (Uint, U64(v)) => U32(v as u32).into(),
                // To Bool (U32 0/1)
                (Bool, U32(v)) => U32(u32::from(v != 0)).into(),
                (Bool, I32(v)) => U32(u32::from(v != 0)).into(),
                (Bool, F32(v)) => U32(u32::from(v != 0.0)).into(),
                // Vector → F32xN
                (Float, I32x2([a, b])) => F32x2([a as f32, b as f32]).into(),
                (Float, I32x3([a, b, c])) => F32x3([a as f32, b as f32, c as f32]).into(),
                (Float, I32x4([a, b, c, d])) => {
                    F32x4([a as f32, b as f32, c as f32, d as f32]).into()
                }
                (Float, U32x2([a, b])) => F32x2([a as f32, b as f32]).into(),
                (Float, U32x3([a, b, c])) => F32x3([a as f32, b as f32, c as f32]).into(),
                (Float, U32x4([a, b, c, d])) => {
                    F32x4([a as f32, b as f32, c as f32, d as f32]).into()
                }
                // Vector → I32xN
                (Sint, F32x2([a, b])) => I32x2([a as i32, b as i32]).into(),
                (Sint, F32x3([a, b, c])) => I32x3([a as i32, b as i32, c as i32]).into(),
                (Sint, F32x4([a, b, c, d])) => {
                    I32x4([a as i32, b as i32, c as i32, d as i32]).into()
                }
                (Sint, U32x2([a, b])) => I32x2([a as i32, b as i32]).into(),
                (Sint, U32x3([a, b, c])) => I32x3([a as i32, b as i32, c as i32]).into(),
                (Sint, U32x4([a, b, c, d])) => {
                    I32x4([a as i32, b as i32, c as i32, d as i32]).into()
                }
                // Vector → U32xN
                (Uint, F32x2([a, b])) => U32x2([a as u32, b as u32]).into(),
                (Uint, F32x3([a, b, c])) => U32x3([a as u32, b as u32, c as u32]).into(),
                (Uint, F32x4([a, b, c, d])) => {
                    U32x4([a as u32, b as u32, c as u32, d as u32]).into()
                }
                (Uint, I32x2([a, b])) => U32x2([a as u32, b as u32]).into(),
                (Uint, I32x3([a, b, c])) => U32x3([a as u32, b as u32, c as u32]).into(),
                (Uint, I32x4([a, b, c, d])) => {
                    U32x4([a as u32, b as u32, c as u32, d as u32]).into()
                }
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
            // Bitcast — reinterpret bits
            None => match (kind, p) {
                (Float, I32(v)) => F32(f32::from_bits(v as u32)).into(),
                (Float, U32(v)) => F32(f32::from_bits(v)).into(),
                (Sint, F32(v)) => I32(v.to_bits() as i32).into(),
                (Sint, U32(v)) => I32(v as i32).into(),
                (Uint, F32(v)) => U32(v.to_bits()).into(),
                (Uint, I32(v)) => U32(v as u32).into(),
                (Float, I32x2([a, b])) => {
                    F32x2([f32::from_bits(a as u32), f32::from_bits(b as u32)]).into()
                }
                (Float, I32x3([a, b, c])) => F32x3([
                    f32::from_bits(a as u32),
                    f32::from_bits(b as u32),
                    f32::from_bits(c as u32),
                ])
                .into(),
                (Float, I32x4([a, b, c, d])) => F32x4([
                    f32::from_bits(a as u32),
                    f32::from_bits(b as u32),
                    f32::from_bits(c as u32),
                    f32::from_bits(d as u32),
                ])
                .into(),
                (Float, U32x2([a, b])) => F32x2([f32::from_bits(a), f32::from_bits(b)]).into(),
                (Float, U32x3([a, b, c])) => {
                    F32x3([f32::from_bits(a), f32::from_bits(b), f32::from_bits(c)]).into()
                }
                (Float, U32x4([a, b, c, d])) => F32x4([
                    f32::from_bits(a),
                    f32::from_bits(b),
                    f32::from_bits(c),
                    f32::from_bits(d),
                ])
                .into(),
                (Sint, F32x2([a, b])) => I32x2([a.to_bits() as i32, b.to_bits() as i32]).into(),
                (Sint, F32x3([a, b, c])) => {
                    I32x3([a.to_bits() as i32, b.to_bits() as i32, c.to_bits() as i32]).into()
                }
                (Sint, F32x4([a, b, c, d])) => I32x4([
                    a.to_bits() as i32,
                    b.to_bits() as i32,
                    c.to_bits() as i32,
                    d.to_bits() as i32,
                ])
                .into(),
                (Sint, U32x2([a, b])) => I32x2([a as i32, b as i32]).into(),
                (Sint, U32x3([a, b, c])) => I32x3([a as i32, b as i32, c as i32]).into(),
                (Sint, U32x4([a, b, c, d])) => {
                    I32x4([a as i32, b as i32, c as i32, d as i32]).into()
                }
                (Uint, F32x2([a, b])) => U32x2([a.to_bits(), b.to_bits()]).into(),
                (Uint, F32x3([a, b, c])) => U32x3([a.to_bits(), b.to_bits(), c.to_bits()]).into(),
                (Uint, F32x4([a, b, c, d])) => {
                    U32x4([a.to_bits(), b.to_bits(), c.to_bits(), d.to_bits()]).into()
                }
                (Uint, I32x2([a, b])) => U32x2([a as u32, b as u32]).into(),
                (Uint, I32x3([a, b, c])) => U32x3([a as u32, b as u32, c as u32]).into(),
                (Uint, I32x4([a, b, c, d])) => {
                    U32x4([a as u32, b as u32, c as u32, d as u32]).into()
                }
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
        let a = self
            .evaluate_expression(arg, state_index, parent_state_index)
            .leaf_value();
        let b = arg1.map(|h| {
            self.evaluate_expression(h, state_index, parent_state_index)
                .leaf_value()
        });
        let c = arg2.map(|h| {
            self.evaluate_expression(h, state_index, parent_state_index)
                .leaf_value()
        });
        let _d = arg3.map(|h| {
            self.evaluate_expression(h, state_index, parent_state_index)
                .leaf_value()
        });

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

                let f: Vec<f32> = base
                    .iter()
                    .zip(exponent)
                    .map(|(base, exp)| base * i32::pow(2, *exp as u32) as f32)
                    .collect();

                Value::Primitive(Primitive::from(f.as_slice()))
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
                    self.evaluate_math(
                        MathFunction::Dot,
                        e2,
                        Some(e3),
                        None,
                        None,
                        state_index,
                        parent_state_index,
                    )
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
            Value::Primitive(Primitive::F64(v)) => Primitive::F64(ff64(v)).into(),
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
            (Value::Primitive(Primitive::F64(av)), Value::Primitive(Primitive::F64(bv))) => {
                Primitive::F64(ff64(*av, *bv)).into()
            }
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
