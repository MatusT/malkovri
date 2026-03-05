use crate::{evaluator::Evaluator, function_state::StackFrame, primitive::Primitive, value::Value};

use std::{cell::RefCell, rc::Rc};

use naga::{
    Expression, GlobalVariable, Handle, Literal, LocalVariable, SwizzleComponent, Type, TypeInner,
    UnaryOperator, VectorSize,
};

impl Evaluator {
    /// Evaluate an expression in the context of the current function frame.
    pub fn evaluate_expression(&self, expression_handle: Handle<Expression>) -> Value {
        let func_idx = match self.current_function_frame_index() {
            Some(i) => i,
            None => return Value::Uninitialized,
        };
        self.eval_expr(expression_handle, func_idx)
    }

    /// Internal expression evaluator that takes a pre-computed function frame index.
    /// All recursive calls use this to avoid redundant stack scans.
    pub(crate) fn eval_expr(
        &self,
        expression_handle: Handle<Expression>,
        func_idx: usize,
    ) -> Value {
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return Value::Uninitialized;
        };
        let function = self.resolve_function(&frame.function_ref);
        let expression = &function.expressions[expression_handle];

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
                self.evaluate_compose(*ty, components, func_idx)
            }
            Expression::Splat { size, value } => self.evaluate_splat(*size, *value, func_idx),
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => self.evaluate_swizzle(*size, *vector, *pattern, func_idx),
            Expression::Load { pointer } => self.evaluate_load(*pointer, func_idx),
            Expression::AccessIndex { base, index } => {
                self.evaluate_access_index(*base, *index, func_idx)
            }
            Expression::FunctionArgument(index) => {
                self.evaluate_function_argument(*index as usize, func_idx)
            }
            Expression::LocalVariable(handle) => self.evaluate_local_variable(*handle, func_idx),
            Expression::Binary { op, left, right } => {
                self.evaluate_binary(*op, *left, *right, func_idx)
            }
            Expression::Unary { op, expr } => {
                let val = self.eval_expr(*expr, func_idx).leaf_value();
                self.evaluate_unary(*op, val)
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => self.evaluate_select(*condition, *accept, *reject, func_idx),
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let val = self.eval_expr(*expr, func_idx);
                crate::eval_cast::evaluate_as(val, *kind, *convert)
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self.evaluate_math(*fun, *arg, *arg1, *arg2, *arg3, func_idx),
            Expression::Relational { fun, argument } => {
                let val = self.eval_expr(*argument, func_idx).leaf_value();
                self.evaluate_relational(*fun, val)
            }
            Expression::ArrayLength(expr) => {
                let argument = self.eval_expr(*expr, func_idx).leaf_value();
                match argument {
                    Value::Array(elements) => Primitive::U32(elements.len() as u32).into(),
                    _ => Value::Uninitialized,
                }
            }
            Expression::GlobalVariable(handle) => self.evaluate_global_variable(*handle),
            Expression::Access { base, index } => self.evaluate_access(*base, *index, func_idx),
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

    fn evaluate_load(&self, pointer: Handle<Expression>, func_idx: usize) -> Value {
        // Check expression cache first.
        let cached = match &self.stack[func_idx] {
            StackFrame::Function(frame) => frame.evaluated_expressions.get(&pointer).cloned(),
            StackFrame::Block(_) => None,
        };
        if let Some(value) = cached {
            return value;
        }
        match self.eval_expr(pointer, func_idx) {
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
    ) -> Value {
        let value = self.eval_expr(base, func_idx).leaf_value();
        value.index_into(index as usize)
    }

    fn evaluate_function_argument(&self, index: usize, func_idx: usize) -> Value {
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return Value::Uninitialized;
        };
        let function = self.resolve_function(&frame.function_ref);
        let function_argument = &function.arguments[index];

        if let Some(binding) = &function_argument.binding {
            match binding {
                naga::ir::Binding::BuiltIn(built_in) => match built_in {
                    // vertex
                    naga::ir::BuiltIn::BaseInstance => {
                        Primitive::U32(self.entry_point_inputs.base_instance).into()
                    }
                    naga::ir::BuiltIn::BaseVertex => {
                        Primitive::I32(self.entry_point_inputs.base_vertex).into()
                    }
                    naga::ir::BuiltIn::ClipDistance => Value::Array(
                        self.entry_point_inputs
                            .clip_distance
                            .iter()
                            .map(|&v| Primitive::F32(v).into())
                            .collect(),
                    ),
                    naga::ir::BuiltIn::CullDistance => Value::Array(
                        self.entry_point_inputs
                            .cull_distance
                            .iter()
                            .map(|&v| Primitive::F32(v).into())
                            .collect(),
                    ),
                    naga::ir::BuiltIn::InstanceIndex => {
                        Primitive::U32(self.entry_point_inputs.instance_index).into()
                    }
                    naga::ir::BuiltIn::PointSize => {
                        Primitive::F32(self.entry_point_inputs.point_size).into()
                    }
                    naga::ir::BuiltIn::VertexIndex => {
                        Primitive::U32(self.entry_point_inputs.vertex_index).into()
                    }
                    naga::ir::BuiltIn::DrawID => {
                        Primitive::U32(self.entry_point_inputs.draw_id).into()
                    }
                    // fragment
                    naga::ir::BuiltIn::Position { .. } => {
                        Primitive::F32x4(self.entry_point_inputs.position).into()
                    }
                    naga::ir::BuiltIn::ViewIndex => {
                        Primitive::I32(self.entry_point_inputs.view_index).into()
                    }
                    naga::ir::BuiltIn::FragDepth => {
                        Primitive::F32(self.entry_point_inputs.frag_depth).into()
                    }
                    naga::ir::BuiltIn::PointCoord => {
                        Primitive::F32x2(self.entry_point_inputs.point_coord).into()
                    }
                    naga::ir::BuiltIn::FrontFacing => {
                        Primitive::U32(self.entry_point_inputs.front_facing as u32).into()
                    }
                    naga::ir::BuiltIn::PrimitiveIndex => {
                        Primitive::U32(self.entry_point_inputs.primitive_index).into()
                    }
                    naga::ir::BuiltIn::SampleIndex => {
                        Primitive::U32(self.entry_point_inputs.sample_index).into()
                    }
                    naga::ir::BuiltIn::SampleMask => {
                        Primitive::U32(self.entry_point_inputs.sample_mask).into()
                    }
                    // compute
                    naga::ir::BuiltIn::GlobalInvocationId => {
                        Primitive::U32x3(self.entry_point_inputs.global_invocation_id).into()
                    }
                    naga::ir::BuiltIn::LocalInvocationId => {
                        Primitive::U32x3(self.entry_point_inputs.local_invocation_id).into()
                    }
                    naga::ir::BuiltIn::LocalInvocationIndex => {
                        Primitive::U32(self.entry_point_inputs.local_invocation_index).into()
                    }
                    naga::ir::BuiltIn::WorkGroupId => {
                        Primitive::U32x3(self.entry_point_inputs.workgroup_id).into()
                    }
                    naga::ir::BuiltIn::WorkGroupSize => {
                        Primitive::U32x3(self.entry_point_inputs.workgroup_size).into()
                    }
                    naga::ir::BuiltIn::NumWorkGroups => {
                        Primitive::U32x3(self.entry_point_inputs.num_workgroups).into()
                    }
                    // subgroup
                    naga::ir::BuiltIn::NumSubgroups => {
                        Primitive::U32(self.entry_point_inputs.num_subgroups).into()
                    }
                    naga::ir::BuiltIn::SubgroupId => {
                        Primitive::U32(self.entry_point_inputs.subgroup_id).into()
                    }
                    naga::ir::BuiltIn::SubgroupSize => {
                        Primitive::U32(self.entry_point_inputs.subgroup_size).into()
                    }
                    naga::ir::BuiltIn::SubgroupInvocationId => {
                        Primitive::U32(self.entry_point_inputs.subgroup_invocation_id).into()
                    }
                },
                naga::ir::Binding::Location { .. } => Value::Uninitialized,
            }
        } else {
            frame.evaluated_function_arguments[index].clone()
        }
    }

    fn evaluate_local_variable(&self, handle: Handle<LocalVariable>, func_idx: usize) -> Value {
        // Check if already initialized.
        let (cached, init_expr) = match &self.stack[func_idx] {
            StackFrame::Function(frame) => {
                let cached = frame.local_variables.get(&handle).cloned();
                let function = self.resolve_function(&frame.function_ref);
                let init = function.local_variables[handle].init;
                (cached, init)
            }
            StackFrame::Block(_) => return Value::Uninitialized,
        };

        if let Some(value) = cached {
            return value;
        }

        match init_expr {
            Some(expr) => Value::Pointer(Rc::new(RefCell::new(self.eval_expr(expr, func_idx)))),
            None => Value::Uninitialized,
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
    ) -> Value {
        let base_value = self.eval_expr(base, func_idx).leaf_value();
        let index_value = self.eval_expr(index, func_idx).leaf_value();

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
    ) -> Value {
        let ty_inner = &self.module.types[ty].inner;
        let vals: Vec<Value> = components
            .iter()
            .map(|c| self.eval_expr(*c, func_idx).leaf_value())
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
        let val = self.eval_expr(value, func_idx).leaf_value();
        self.splat_value(size, val)
    }

    fn evaluate_swizzle(
        &self,
        size: VectorSize,
        vector: Handle<Expression>,
        pattern: [SwizzleComponent; 4],
        func_idx: usize,
    ) -> Value {
        let vec_val = self.eval_expr(vector, func_idx).leaf_value();

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
        let cond = self.eval_expr(condition, func_idx).leaf_value();

        let is_true = match cond {
            Value::Primitive(Primitive::U32(v)) => v != 0,
            _ => false,
        };

        if is_true {
            self.eval_expr(accept, func_idx)
        } else {
            self.eval_expr(reject, func_idx)
        }
    }
}
