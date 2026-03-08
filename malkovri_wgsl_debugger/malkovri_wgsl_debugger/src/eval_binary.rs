use crate::evaluator::Evaluator;
use crate::primitive::Primitive;
use crate::value::Value;
use naga::{Expression, Handle};

impl Evaluator {
    pub(crate) fn evaluate_binary(
        &self,
        op: naga::BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
        func_idx: usize,
    ) -> Value {
        use naga::BinaryOperator::*;

        let l = self.eval_expr(left, func_idx).leaf_value();
        let r = self.eval_expr(right, func_idx).leaf_value();

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
}
