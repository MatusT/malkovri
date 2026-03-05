use crate::{evaluator::Evaluator, primitive::Primitive, value::Value};

use naga::{Expression, Handle, MathFunction, RelationalFunction, UnaryOperator};

// --- Free helper functions (no &self needed) ---

fn math_unary_f32(val: Value, f: impl Fn(f32) -> f32) -> Value {
    val.map_f32(f)
}

fn math_binary_f32(a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Value {
    a.zip_map_f32(b, f)
}

fn math_unary_float_or_int(
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

fn math_unary_int(val: Value, fi32: impl Fn(i32) -> i32, fu32: impl Fn(u32) -> u32) -> Value {
    // Guard: integer-only functions should not silently corrupt float inputs.
    if let Some(p) = val.as_primitive() {
        if p.as_f32_slice().is_some() || matches!(p, Primitive::F64(_)) {
            return Value::Uninitialized;
        }
    }
    val.map_numeric(|_| 0.0, fi32, fu32)
}

fn math_clamp(val: Value, min_val: Value, max_val: Value) -> Value {
    val.zip3_map_numeric(
        min_val,
        max_val,
        |v, lo, hi| v.clamp(lo, hi),
        |v, lo, hi| v.clamp(lo, hi),
        |v, lo, hi| v.clamp(lo, hi),
    )
}

impl Evaluator {
    pub(crate) fn evaluate_math(
        &self,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        arg2: Option<Handle<Expression>>,
        arg3: Option<Handle<Expression>>,
        func_idx: usize,
    ) -> Value {
        let a = self.eval_expr(arg, func_idx).leaf_value();
        let b = arg1.map(|h| self.eval_expr(h, func_idx).leaf_value());
        let c = arg2.map(|h| self.eval_expr(h, func_idx).leaf_value());
        let _d = arg3.map(|h| self.eval_expr(h, func_idx).leaf_value());

        match fun {
            // --- Comparison ---
            MathFunction::Abs => math_unary_float_or_int(
                a,
                f32::abs,
                f64::abs,
                |v: i32| v.wrapping_abs(),
                |v: u32| v,
            ),
            MathFunction::Min => math_binary_float_or_int(
                a,
                b.unwrap_or(Value::Uninitialized),
                f32::min,
                f64::min,
                i32::min,
                u32::min,
            ),
            MathFunction::Max => math_binary_float_or_int(
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
                math_clamp(a, b_val, c_val)
            }
            MathFunction::Saturate => math_unary_f32(a, |v| v.clamp(0.0, 1.0)),

            // --- Trigonometry ---
            MathFunction::Cos => math_unary_f32(a, f32::cos),
            MathFunction::Cosh => math_unary_f32(a, f32::cosh),
            MathFunction::Sin => math_unary_f32(a, f32::sin),
            MathFunction::Sinh => math_unary_f32(a, f32::sinh),
            MathFunction::Tan => math_unary_f32(a, f32::tan),
            MathFunction::Tanh => math_unary_f32(a, f32::tanh),
            MathFunction::Acos => math_unary_f32(a, f32::acos),
            MathFunction::Asin => math_unary_f32(a, f32::asin),
            MathFunction::Atan => math_unary_f32(a, f32::atan),
            MathFunction::Atan2 => {
                math_binary_f32(a, b.unwrap_or(Value::Uninitialized), f32::atan2)
            }
            MathFunction::Asinh => math_unary_f32(a, f32::asinh),
            MathFunction::Acosh => math_unary_f32(a, f32::acosh),
            MathFunction::Atanh => math_unary_f32(a, f32::atanh),
            MathFunction::Radians => math_unary_f32(a, |v| v.to_radians()),
            MathFunction::Degrees => math_unary_f32(a, |v| v.to_degrees()),

            // --- Decomposition ---
            MathFunction::Ceil => math_unary_f32(a, f32::ceil),
            MathFunction::Floor => math_unary_f32(a, f32::floor),
            MathFunction::Round => math_unary_f32(a, f32::round),
            MathFunction::Fract => math_unary_f32(a, f32::fract),
            MathFunction::Trunc => math_unary_f32(a, f32::trunc),
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
            MathFunction::Exp => math_unary_f32(a, f32::exp),
            MathFunction::Exp2 => math_unary_f32(a, f32::exp2),
            MathFunction::Log => math_unary_f32(a, f32::ln),
            MathFunction::Log2 => math_unary_f32(a, f32::log2),
            MathFunction::Pow => math_binary_f32(a, b.unwrap_or(Value::Uninitialized), f32::powf),

            // --- Geometry ---
            MathFunction::Dot => {
                let b_val = b.unwrap_or(Value::Uninitialized);
                if let (Some(ap), Some(bp)) = (a.as_primitive(), b_val.as_primitive()) {
                    if let (Some(af), Some(bf)) = (ap.as_f32_slice(), bp.as_f32_slice()) {
                        return Primitive::F32(af.iter().zip(bf).map(|(a, b)| a * b).sum()).into();
                    }
                    if let (Some(ai), Some(bi)) = (ap.as_i32_slice(), bp.as_i32_slice()) {
                        return Primitive::I32(
                            ai.iter()
                                .zip(bi)
                                .map(|(a, b)| a.wrapping_mul(*b))
                                .fold(0i32, i32::wrapping_add),
                        )
                        .into();
                    }
                    if let (Some(au), Some(bu)) = (ap.as_u32_slice(), bp.as_u32_slice()) {
                        return Primitive::U32(
                            au.iter()
                                .zip(bu)
                                .map(|(a, b)| a.wrapping_mul(*b))
                                .fold(0u32, u32::wrapping_add),
                        )
                        .into();
                    }
                }
                Value::Uninitialized
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
                    self.evaluate_math(MathFunction::Dot, e2, Some(e3), None, None, func_idx)
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
            MathFunction::Sign => math_unary_float_or_int(
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
                math_binary_f32(a, b.unwrap_or(Value::Uninitialized), |edge, x| {
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
            MathFunction::Sqrt => math_unary_f32(a, f32::sqrt),
            MathFunction::InverseSqrt => math_unary_f32(a, |v| 1.0 / v.sqrt()),
            MathFunction::QuantizeToF16 => {
                // Approximate: round to f16 precision
                math_unary_f32(a, |v| {
                    let bits = v.to_bits();
                    // Simple truncation of mantissa to 10 bits
                    let truncated = bits & 0xFFFFE000;
                    f32::from_bits(truncated)
                })
            }

            // --- Bits ---
            MathFunction::CountTrailingZeros => math_unary_int(
                a,
                |v: i32| v.trailing_zeros() as i32,
                |v: u32| v.trailing_zeros(),
            ),
            MathFunction::CountLeadingZeros => math_unary_int(
                a,
                |v: i32| v.leading_zeros() as i32,
                |v: u32| v.leading_zeros(),
            ),
            MathFunction::CountOneBits => {
                math_unary_int(a, |v: i32| v.count_ones() as i32, |v: u32| v.count_ones())
            }
            MathFunction::ReverseBits => {
                math_unary_int(a, |v: i32| v.reverse_bits(), |v: u32| v.reverse_bits())
            }
            MathFunction::FirstTrailingBit => math_unary_int(
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
            MathFunction::FirstLeadingBit => math_unary_int(
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

    pub(crate) fn evaluate_relational(&self, fun: RelationalFunction, val: Value) -> Value {
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
