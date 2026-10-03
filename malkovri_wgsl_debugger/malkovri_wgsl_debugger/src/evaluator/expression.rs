use crate::{
    function_state::StackFrame,
    place::{ArgumentValue, EvaluatedExpression, Place, PlaceRoot},
    primitive::Primitive,
    value::Value,
};

use super::InvocationState;

use naga::{
    Expression, Handle, Literal, LocalVariable, SwizzleComponent, Type, TypeInner, UnaryOperator,
    VectorSize,
};

impl InvocationState {
    /// Evaluate an expression in the context of the current function frame.
    pub(crate) fn evaluate_expression(&self, expression_handle: Handle<Expression>) -> Value {
        let Ok(func_idx) = self.current_function_frame_index() else {
            return Value::Uninitialized;
        };
        self.eval_value(expression_handle, func_idx)
    }

    pub(crate) fn evaluate_argument(&self, expression_handle: Handle<Expression>) -> ArgumentValue {
        let Ok(func_idx) = self.current_function_frame_index() else {
            return ArgumentValue::Value(Value::Uninitialized);
        };
        match self.eval_expr(expression_handle, func_idx) {
            EvaluatedExpression::Value(value) => ArgumentValue::Value(value),
            EvaluatedExpression::Place(place) => ArgumentValue::Place(place),
        }
    }

    pub(crate) fn eval_value(
        &self,
        expression_handle: Handle<Expression>,
        func_idx: usize,
    ) -> Value {
        match self.eval_expr(expression_handle, func_idx) {
            EvaluatedExpression::Value(value) => value,
            EvaluatedExpression::Place(place) => self.read_place(&place),
        }
    }

    /// Internal expression evaluator that takes a pre-computed function frame index.
    /// All recursive calls use this to avoid redundant stack scans.
    pub(crate) fn eval_expr(
        &self,
        expression_handle: Handle<Expression>,
        func_idx: usize,
    ) -> EvaluatedExpression {
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return Value::Uninitialized.into();
        };
        if let Some(value) = frame.evaluated_expressions.get(&expression_handle) {
            return value.clone();
        }
        let function = self.resolve_function(&frame.function_id);
        let expression = &function.expressions[expression_handle];

        match expression {
            Expression::Literal(literal) => self.evaluate_literal(literal).into(),
            Expression::Constant(handle) => self
                .evaluate_global_expression(self.program.module().constants[*handle].init)
                .into(),
            Expression::Override(handle) => match self.program.module().overrides[*handle].init {
                Some(init) => self.evaluate_global_expression(init).into(),
                None => Value::Uninitialized.into(),
            },
            Expression::ZeroValue(ty) => Value::zero(self.program.module(), *ty).into(),
            Expression::Compose { ty, components } => {
                self.evaluate_compose(*ty, components, func_idx).into()
            }
            Expression::Splat { size, value } => {
                self.evaluate_splat(*size, *value, func_idx).into()
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => self
                .evaluate_swizzle(*size, *vector, *pattern, func_idx)
                .into(),
            Expression::Load { pointer } => self.evaluate_load(*pointer, func_idx).into(),
            Expression::AccessIndex { base, index } => {
                self.evaluate_access_index(*base, *index, func_idx)
            }
            Expression::FunctionArgument(index) => {
                match self.evaluate_argument_at(*index as usize, func_idx) {
                    ArgumentValue::Value(value) => value.into(),
                    ArgumentValue::Place(place) => place.into(),
                }
            }
            Expression::LocalVariable(handle) => Place::new(PlaceRoot::Local {
                function_frame_index: func_idx,
                handle: *handle,
            })
            .into(),
            Expression::Binary { op, left, right } => {
                self.evaluate_binary(*op, *left, *right, func_idx).into()
            }
            Expression::Unary { op, expr } => {
                let val = self.eval_value(*expr, func_idx);
                self.evaluate_unary(*op, val).into()
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => self
                .evaluate_select(*condition, *accept, *reject, func_idx)
                .into(),
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let val = self.eval_value(*expr, func_idx);
                super::cast::evaluate_as(val, *kind, *convert).into()
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self
                .evaluate_math(*fun, *arg, *arg1, *arg2, *arg3, func_idx)
                .into(),
            Expression::Relational { fun, argument } => {
                let val = self.eval_value(*argument, func_idx);
                self.evaluate_relational(*fun, val).into()
            }
            Expression::ArrayLength(expr) => {
                let argument = self.eval_value(*expr, func_idx);
                match argument {
                    Value::Array(elements) => Value::from(Primitive::U32(elements.len() as u32)),
                    _ => Value::Uninitialized,
                }
                .into()
            }
            Expression::GlobalVariable(handle) => {
                Place::new(PlaceRoot::Global { handle: *handle }).into()
            }
            Expression::Access { base, index } => self.evaluate_access(*base, *index, func_idx),
            Expression::CallResult(_)
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. } => {
                // Completed statement results were returned from the cache above.
                // A missing result is not available until its statement executes.
                Value::Uninitialized.into()
            }
            _ => Value::Uninitialized.into(),
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

    fn evaluate_load(&self, pointer: Handle<Expression>, func_idx: usize) -> Value {
        match self.eval_expr(pointer, func_idx) {
            EvaluatedExpression::Place(place) => self.read_place(&place),
            EvaluatedExpression::Value(value) => value,
        }
    }

    fn evaluate_access_index(
        &self,
        base: Handle<Expression>,
        index: u32,
        func_idx: usize,
    ) -> EvaluatedExpression {
        match self.eval_expr(base, func_idx) {
            EvaluatedExpression::Place(place) => place.with_index(index as usize).into(),
            EvaluatedExpression::Value(value) => value.index_into(index as usize).into(),
        }
    }

    pub(crate) fn evaluate_function_argument(&self, index: usize, func_idx: usize) -> Value {
        match self.evaluate_argument_at(index, func_idx) {
            ArgumentValue::Value(value) => value,
            ArgumentValue::Place(place) => self.read_place(&place),
        }
    }

    pub(crate) fn evaluate_argument_at(&self, index: usize, func_idx: usize) -> ArgumentValue {
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return ArgumentValue::Value(Value::Uninitialized);
        };
        let function = self.resolve_function(&frame.function_id);
        let function_argument = &function.arguments[index];

        if let Some(binding) = &function_argument.binding {
            ArgumentValue::Value(match binding {
                naga::Binding::BuiltIn(builtin) => {
                    self.inputs.builtin(*builtin, &self.global_constants)
                }
                naga::Binding::Location { .. } => Value::Uninitialized,
            })
        } else {
            frame
                .evaluated_function_arguments
                .get(index)
                .cloned()
                .unwrap_or(ArgumentValue::Value(Value::Uninitialized))
        }
    }

    pub(crate) fn evaluate_local_variable(
        &self,
        handle: Handle<LocalVariable>,
        func_idx: usize,
    ) -> Value {
        self.read_place(&Place::new(PlaceRoot::Local {
            function_frame_index: func_idx,
            handle,
        }))
    }

    fn evaluate_access(
        &self,
        base: Handle<Expression>,
        index: Handle<Expression>,
        func_idx: usize,
    ) -> EvaluatedExpression {
        let index_value = self.eval_value(index, func_idx);

        let index: usize = match index_value {
            Value::Primitive(Primitive::U32(i)) => i as usize,
            Value::Primitive(Primitive::I32(i)) => i.max(0) as usize,
            _ => return Value::Uninitialized.into(),
        };

        match self.eval_expr(base, func_idx) {
            EvaluatedExpression::Place(place) => place.with_index(index).into(),
            EvaluatedExpression::Value(value) => value.index_into(index).into(),
        }
    }

    /// Evaluate an expression from the module's global_expressions arena (used for constants/overrides).
    fn evaluate_global_expression(&self, expr_handle: Handle<Expression>) -> Value {
        evaluate_global_expression(self.program.module(), expr_handle)
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
                    (ScalarKind::Uint, 4) | (ScalarKind::Bool, _) => {
                        compose_vec!(Value::collect_u32_components)
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
        components: &[Handle<Expression>],
        func_idx: usize,
    ) -> Value {
        let ty_inner = &self.program.module().types[ty].inner;
        let vals: Vec<Value> = components
            .iter()
            .map(|c| self.eval_value(*c, func_idx))
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
    ) -> Value {
        let val = self.eval_value(value, func_idx);
        self.splat_value(size, val)
    }

    fn evaluate_swizzle(
        &self,
        size: VectorSize,
        vector: Handle<Expression>,
        pattern: [SwizzleComponent; 4],
        func_idx: usize,
    ) -> Value {
        let vec_val = self.eval_value(vector, func_idx);

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

    pub(crate) fn evaluate_unary(&self, op: UnaryOperator, val: Value) -> Value {
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
    ) -> Value {
        let cond = self.eval_value(condition, func_idx);

        if let Value::Primitive(Primitive::U32(condition)) = cond {
            return self.eval_value(if condition != 0 { accept } else { reject }, func_idx);
        }

        let accept = self.eval_value(accept, func_idx);
        let reject = self.eval_value(reject, func_idx);
        let Some(condition) = cond.as_primitive().and_then(Primitive::as_u32_slice) else {
            return Value::Uninitialized;
        };
        let (Some(accept), Some(reject)) = (accept.as_primitive(), reject.as_primitive()) else {
            return Value::Uninitialized;
        };
        fn select<T: Copy>(accept: &[T], reject: &[T], condition: &[u32]) -> Vec<T> {
            accept
                .iter()
                .zip(reject)
                .zip(condition)
                .map(|((&a, &r), &c)| if c != 0 { a } else { r })
                .collect()
        }
        if let (Some(a), Some(r)) = (accept.as_f32_slice(), reject.as_f32_slice()) {
            return Primitive::from(select(a, r, condition).as_slice()).into();
        }
        if let (Some(a), Some(r)) = (accept.as_i32_slice(), reject.as_i32_slice()) {
            return Primitive::from(select(a, r, condition).as_slice()).into();
        }
        if let (Some(a), Some(r)) = (accept.as_u32_slice(), reject.as_u32_slice()) {
            return Primitive::from(select(a, r, condition).as_slice()).into();
        }
        Value::Uninitialized
    }
}

/// Evaluate a global expression given only a module reference (no `InvocationState` instance needed).
pub(crate) fn evaluate_global_expression(
    module: &naga::Module,
    expr_handle: Handle<Expression>,
) -> Value {
    let expression = &module.global_expressions[expr_handle];
    match expression {
        Expression::Literal(literal) => match literal {
            Literal::F32(v) => Primitive::F32(*v).into(),
            Literal::F64(v) => Primitive::F64(*v).into(),
            Literal::I32(v) => Primitive::I32(*v).into(),
            Literal::I64(v) => Primitive::I64(*v).into(),
            Literal::U32(v) => Primitive::U32(*v).into(),
            Literal::U64(v) => Primitive::U64(*v).into(),
            Literal::Bool(v) => Primitive::U32(if *v { 1 } else { 0 }).into(),
            _ => Value::Uninitialized,
        },
        Expression::ZeroValue(ty) => Value::zero(module, *ty),
        Expression::Constant(handle) => {
            evaluate_global_expression(module, module.constants[*handle].init)
        }
        Expression::Compose { ty, components } => {
            let ty_inner = &module.types[*ty].inner;
            let vals: Vec<Value> = components
                .iter()
                .map(|c| evaluate_global_expression(module, *c))
                .collect();
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
                    macro_rules! compose_vec {
                        ($collect:expr) => {{
                            let comps = $collect(&vals);
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
                        (ScalarKind::Uint, 4) | (ScalarKind::Bool, _) => {
                            compose_vec!(Value::collect_u32_components)
                        }
                        _ => Value::Uninitialized,
                    }
                }
                _ => Value::Uninitialized,
            }
        }
        Expression::Splat { size, value } => {
            let val = evaluate_global_expression(module, *value);
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
        _ => Value::Uninitialized,
    }
}
