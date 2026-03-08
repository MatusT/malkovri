use crate::primitive::Primitive;
use crate::value::Value;

/// Cast/bitcast a scalar or vector value to a different scalar kind.
pub(crate) fn evaluate_as(
    val: Value,
    kind: naga::ScalarKind,
    convert: Option<naga::Bytes>,
) -> Value {
    use Primitive::*;
    use naga::ScalarKind::*;

    let val = val.leaf_value();
    let Value::Primitive(ref p) = val else {
        return Value::Uninitialized;
    };

    match convert {
        Some(4) => match (kind, p) {
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
            (Float, F32(v)) => F64(*v as f64).into(),
            (Float, F64(v)) => F64(*v).into(),
            (Float, I32(v)) => F64(*v as f64).into(),
            (Float, U32(v)) => F64(*v as f64).into(),
            (Float, I64(v)) => F64(*v as f64).into(),
            (Float, U64(v)) => F64(*v as f64).into(),
            (Sint, I32(v)) => I64(*v as i64).into(),
            (Sint, U32(v)) => I64(*v as i64).into(),
            (Sint, I64(v)) => I64(*v).into(),
            (Sint, U64(v)) => I64(*v as i64).into(),
            (Sint, F32(v)) => I64(*v as i64).into(),
            (Sint, F64(v)) => I64(*v as i64).into(),
            (Uint, I32(v)) => U64(*v as u64).into(),
            (Uint, U32(v)) => U64(*v as u64).into(),
            (Uint, I64(v)) => U64(*v as u64).into(),
            (Uint, U64(v)) => U64(*v).into(),
            (Uint, F32(v)) => U64(*v as u64).into(),
            (Uint, F64(v)) => U64(*v as u64).into(),
            _ => Value::Uninitialized,
        },
        // Bitcast — reinterpret bits (scalars handled explicitly, vectors via cross-type maps)
        None => match (kind, p) {
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
