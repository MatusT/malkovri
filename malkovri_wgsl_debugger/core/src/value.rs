use std::{cell::RefCell, rc::Rc};

use naga::TypeInner;

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub enum Value {
    #[default]
    Uninitialized,
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    // Vector types
    U32x2([u32; 2]),
    U32x3([u32; 3]),
    U32x4([u32; 4]),
    I32x2([i32; 2]),
    I32x3([i32; 3]),
    I32x4([i32; 4]),
    F32x2([f32; 2]),
    F32x3([f32; 3]),
    F32x4([f32; 4]),
    // Composite types
    Array(Vec<Rc<RefCell<Value>>>),
    Pointer(Rc<RefCell<Value>>),
}

/// Decomposed scalar components of a Value, used for generic operations.
pub enum ScalarComponents {
    F32(Vec<f32>),
    I32(Vec<i32>),
    U32(Vec<u32>),
}

impl Value {
    pub fn leaf_value(&self) -> Value {
        match self {
            Value::Pointer(inner) => inner.borrow().leaf_value(),
            _ => self.clone(),
        }
    }

    // ── Component decomposition / reconstruction ──────────────────────

    /// Decompose a scalar or vector Value into its scalar components.
    /// Returns None for non-numeric types (Array, Pointer, Uninitialized, F64, I64, U64).
    pub fn to_components(&self) -> Option<ScalarComponents> {
        match self {
            Value::F32(v) => Some(ScalarComponents::F32(vec![*v])),
            Value::F32x2(a) => Some(ScalarComponents::F32(a.to_vec())),
            Value::F32x3(a) => Some(ScalarComponents::F32(a.to_vec())),
            Value::F32x4(a) => Some(ScalarComponents::F32(a.to_vec())),
            Value::I32(v) => Some(ScalarComponents::I32(vec![*v])),
            Value::I32x2(a) => Some(ScalarComponents::I32(a.to_vec())),
            Value::I32x3(a) => Some(ScalarComponents::I32(a.to_vec())),
            Value::I32x4(a) => Some(ScalarComponents::I32(a.to_vec())),
            Value::U32(v) => Some(ScalarComponents::U32(vec![*v])),
            Value::U32x2(a) => Some(ScalarComponents::U32(a.to_vec())),
            Value::U32x3(a) => Some(ScalarComponents::U32(a.to_vec())),
            Value::U32x4(a) => Some(ScalarComponents::U32(a.to_vec())),
            _ => None,
        }
    }

    /// Reconstruct a Value from f32 components, matching the original size.
    pub fn from_f32_slice(s: &[f32]) -> Value {
        match s.len() {
            1 => Value::F32(s[0]),
            2 => Value::F32x2([s[0], s[1]]),
            3 => Value::F32x3([s[0], s[1], s[2]]),
            4 => Value::F32x4([s[0], s[1], s[2], s[3]]),
            _ => Value::Uninitialized,
        }
    }

    /// Reconstruct a Value from i32 components, matching the original size.
    pub fn from_i32_slice(s: &[i32]) -> Value {
        match s.len() {
            1 => Value::I32(s[0]),
            2 => Value::I32x2([s[0], s[1]]),
            3 => Value::I32x3([s[0], s[1], s[2]]),
            4 => Value::I32x4([s[0], s[1], s[2], s[3]]),
            _ => Value::Uninitialized,
        }
    }

    /// Reconstruct a Value from u32 components, matching the original size.
    pub fn from_u32_slice(s: &[u32]) -> Value {
        match s.len() {
            1 => Value::U32(s[0]),
            2 => Value::U32x2([s[0], s[1]]),
            3 => Value::U32x3([s[0], s[1], s[2]]),
            4 => Value::U32x4([s[0], s[1], s[2], s[3]]),
            _ => Value::Uninitialized,
        }
    }

    /// Number of scalar components (1 for scalars, 2-4 for vectors, 0 for others).
    #[allow(dead_code)]
    pub fn component_count(&self) -> usize {
        match self {
            Value::F32(_) | Value::I32(_) | Value::U32(_) |
            Value::F64(_) | Value::I64(_) | Value::U64(_) => 1,
            Value::F32x2(_) | Value::I32x2(_) | Value::U32x2(_) => 2,
            Value::F32x3(_) | Value::I32x3(_) | Value::U32x3(_) => 3,
            Value::F32x4(_) | Value::I32x4(_) | Value::U32x4(_) => 4,
            _ => 0,
        }
    }

    // ── Unary map (component-wise) ────────────────────────────────────

    /// Apply a component-wise f32 function to an f32 scalar or vector.
    pub fn map_f32(self, f: impl Fn(f32) -> f32) -> Value {
        match self.to_components() {
            Some(ScalarComponents::F32(v)) => {
                Value::from_f32_slice(&v.iter().map(|x| f(*x)).collect::<Vec<_>>())
            }
            _ => self,
        }
    }

    /// Apply a component-wise i32 function to an i32 scalar or vector.
    pub fn map_i32(self, f: impl Fn(i32) -> i32) -> Value {
        match self.to_components() {
            Some(ScalarComponents::I32(v)) => {
                Value::from_i32_slice(&v.iter().map(|x| f(*x)).collect::<Vec<_>>())
            }
            _ => self,
        }
    }

    /// Apply a component-wise u32 function to a u32 scalar or vector.
    pub fn map_u32(self, f: impl Fn(u32) -> u32) -> Value {
        match self.to_components() {
            Some(ScalarComponents::U32(v)) => {
                Value::from_u32_slice(&v.iter().map(|x| f(*x)).collect::<Vec<_>>())
            }
            _ => self,
        }
    }

    // ── Binary zip-map (component-wise, same type) ────────────────────

    /// Component-wise binary f32 operation on two matching f32 values.
    pub fn zip_map_f32(self, other: Value, f: impl Fn(f32, f32) -> f32) -> Value {
        match (self.to_components(), other.to_components()) {
            (Some(ScalarComponents::F32(a)), Some(ScalarComponents::F32(b))) if a.len() == b.len() => {
                Value::from_f32_slice(&a.iter().zip(b.iter()).map(|(x, y)| f(*x, *y)).collect::<Vec<_>>())
            }
            _ => Value::Uninitialized,
        }
    }

    /// Component-wise binary f32 comparison returning u32 booleans.
    pub fn zip_cmp_f32(self, other: Value, f: impl Fn(f32, f32) -> bool) -> Value {
        match (self.to_components(), other.to_components()) {
            (Some(ScalarComponents::F32(a)), Some(ScalarComponents::F32(b))) if a.len() == b.len() => {
                Value::from_u32_slice(&a.iter().zip(b.iter()).map(|(x, y)| u32::from(f(*x, *y))).collect::<Vec<_>>())
            }
            _ => Value::Uninitialized,
        }
    }

    /// Component-wise binary i32 operation on two matching i32 values.
    pub fn zip_map_i32(self, other: Value, f: impl Fn(i32, i32) -> i32) -> Value {
        match (self.to_components(), other.to_components()) {
            (Some(ScalarComponents::I32(a)), Some(ScalarComponents::I32(b))) if a.len() == b.len() => {
                Value::from_i32_slice(&a.iter().zip(b.iter()).map(|(x, y)| f(*x, *y)).collect::<Vec<_>>())
            }
            _ => Value::Uninitialized,
        }
    }

    /// Component-wise binary i32 comparison returning u32 booleans.
    pub fn zip_cmp_i32(self, other: Value, f: impl Fn(i32, i32) -> bool) -> Value {
        match (self.to_components(), other.to_components()) {
            (Some(ScalarComponents::I32(a)), Some(ScalarComponents::I32(b))) if a.len() == b.len() => {
                Value::from_u32_slice(&a.iter().zip(b.iter()).map(|(x, y)| u32::from(f(*x, *y))).collect::<Vec<_>>())
            }
            _ => Value::Uninitialized,
        }
    }

    /// Component-wise binary u32 operation on two matching u32 values.
    pub fn zip_map_u32(self, other: Value, f: impl Fn(u32, u32) -> u32) -> Value {
        match (self.to_components(), other.to_components()) {
            (Some(ScalarComponents::U32(a)), Some(ScalarComponents::U32(b))) if a.len() == b.len() => {
                Value::from_u32_slice(&a.iter().zip(b.iter()).map(|(x, y)| f(*x, *y)).collect::<Vec<_>>())
            }
            _ => Value::Uninitialized,
        }
    }

    /// Component-wise binary u32 comparison returning u32 booleans.
    pub fn zip_cmp_u32(self, other: Value, f: impl Fn(u32, u32) -> bool) -> Value {
        match (self.to_components(), other.to_components()) {
            (Some(ScalarComponents::U32(a)), Some(ScalarComponents::U32(b))) if a.len() == b.len() => {
                Value::from_u32_slice(&a.iter().zip(b.iter()).map(|(x, y)| u32::from(f(*x, *y))).collect::<Vec<_>>())
            }
            _ => Value::Uninitialized,
        }
    }

    // ── Generic component-wise operations ─────────────────────────────

    /// Apply a unary operation dispatching on the scalar kind.
    /// Handles f32/i32/u32 scalars and vectors.
    pub fn map_numeric(
        self,
        ff32: impl Fn(f32) -> f32,
        fi32: impl Fn(i32) -> i32,
        fu32: impl Fn(u32) -> u32,
    ) -> Value {
        match self.to_components() {
            Some(ScalarComponents::F32(v)) => Value::from_f32_slice(&v.iter().map(|x| ff32(*x)).collect::<Vec<_>>()),
            Some(ScalarComponents::I32(v)) => Value::from_i32_slice(&v.iter().map(|x| fi32(*x)).collect::<Vec<_>>()),
            Some(ScalarComponents::U32(v)) => Value::from_u32_slice(&v.iter().map(|x| fu32(*x)).collect::<Vec<_>>()),
            None => Value::Uninitialized,
        }
    }

    /// Apply a binary operation dispatching on the scalar kind.
    /// Both values must have the same type and size.
    pub fn zip_map_numeric(
        self,
        other: Value,
        ff32: impl Fn(f32, f32) -> f32,
        fi32: impl Fn(i32, i32) -> i32,
        fu32: impl Fn(u32, u32) -> u32,
    ) -> Value {
        match (self.to_components(), other.to_components()) {
            (Some(ScalarComponents::F32(a)), Some(ScalarComponents::F32(b))) if a.len() == b.len() => {
                Value::from_f32_slice(&a.iter().zip(b.iter()).map(|(x, y)| ff32(*x, *y)).collect::<Vec<_>>())
            }
            (Some(ScalarComponents::I32(a)), Some(ScalarComponents::I32(b))) if a.len() == b.len() => {
                Value::from_i32_slice(&a.iter().zip(b.iter()).map(|(x, y)| fi32(*x, *y)).collect::<Vec<_>>())
            }
            (Some(ScalarComponents::U32(a)), Some(ScalarComponents::U32(b))) if a.len() == b.len() => {
                Value::from_u32_slice(&a.iter().zip(b.iter()).map(|(x, y)| fu32(*x, *y)).collect::<Vec<_>>())
            }
            _ => Value::Uninitialized,
        }
    }

    /// Apply a ternary operation dispatching on the scalar kind.
    /// All three values must have the same type and size.
    pub fn zip3_map_numeric(
        self,
        b: Value,
        c: Value,
        ff32: impl Fn(f32, f32, f32) -> f32,
        fi32: impl Fn(i32, i32, i32) -> i32,
        fu32: impl Fn(u32, u32, u32) -> u32,
    ) -> Value {
        match (self.to_components(), b.to_components(), c.to_components()) {
            (Some(ScalarComponents::F32(a)), Some(ScalarComponents::F32(b)), Some(ScalarComponents::F32(c)))
                if a.len() == b.len() && b.len() == c.len() =>
            {
                Value::from_f32_slice(&a.iter().zip(b.iter()).zip(c.iter()).map(|((x, y), z)| ff32(*x, *y, *z)).collect::<Vec<_>>())
            }
            (Some(ScalarComponents::I32(a)), Some(ScalarComponents::I32(b)), Some(ScalarComponents::I32(c)))
                if a.len() == b.len() && b.len() == c.len() =>
            {
                Value::from_i32_slice(&a.iter().zip(b.iter()).zip(c.iter()).map(|((x, y), z)| fi32(*x, *y, *z)).collect::<Vec<_>>())
            }
            (Some(ScalarComponents::U32(a)), Some(ScalarComponents::U32(b)), Some(ScalarComponents::U32(c)))
                if a.len() == b.len() && b.len() == c.len() =>
            {
                Value::from_u32_slice(&a.iter().zip(b.iter()).zip(c.iter()).map(|((x, y), z)| fu32(*x, *y, *z)).collect::<Vec<_>>())
            }
            _ => Value::Uninitialized,
        }
    }

    // ── Flatten components for Compose ─────────────────────────────────

    /// Collect all f32 scalar components from a slice of Values (flattening vectors).
    pub fn collect_f32_components(vals: &[Value]) -> Vec<f32> {
        let mut result = Vec::new();
        for v in vals {
            if let Some(ScalarComponents::F32(comps)) = v.to_components() {
                result.extend(comps);
            }
        }
        result
    }

    /// Collect all i32 scalar components from a slice of Values (flattening vectors).
    pub fn collect_i32_components(vals: &[Value]) -> Vec<i32> {
        let mut result = Vec::new();
        for v in vals {
            if let Some(ScalarComponents::I32(comps)) = v.to_components() {
                result.extend(comps);
            }
        }
        result
    }

    /// Collect all u32 scalar components from a slice of Values (flattening vectors).
    pub fn collect_u32_components(vals: &[Value]) -> Vec<u32> {
        let mut result = Vec::new();
        for v in vals {
            if let Some(ScalarComponents::U32(comps)) = v.to_components() {
                result.extend(comps);
            }
        }
        result
    }

    /// Extract a single scalar component from a vector by index.
    pub fn extract_component(&self, index: usize) -> Value {
        match self.to_components() {
            Some(ScalarComponents::F32(v)) => v.get(index).copied().map(Value::F32).unwrap_or(Value::Uninitialized),
            Some(ScalarComponents::I32(v)) => v.get(index).copied().map(Value::I32).unwrap_or(Value::Uninitialized),
            Some(ScalarComponents::U32(v)) => v.get(index).copied().map(Value::U32).unwrap_or(Value::Uninitialized),
            None => Value::Uninitialized,
        }
    }
}

impl From<&TypeInner> for Value {
    fn from(ty: &TypeInner) -> Self {
        use naga::{Scalar, ScalarKind, VectorSize};
        match ty {
            TypeInner::Scalar(Scalar { kind: ScalarKind::Float, width: 4 }) => Value::F32(0.0),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Float, width: 8 }) => Value::F64(0.0),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Sint, width: 4 }) => Value::I32(0),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Sint, width: 8 }) => Value::I64(0),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Uint, width: 4 }) => Value::U32(0),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Uint, width: 8 }) => Value::U64(0),
            TypeInner::Vector { size, scalar: Scalar { kind, width: 4 } } => {
                match (size, kind) {
                    (VectorSize::Bi,   ScalarKind::Float) => Value::F32x2([0.0; 2]),
                    (VectorSize::Tri,  ScalarKind::Float) => Value::F32x3([0.0; 3]),
                    (VectorSize::Quad, ScalarKind::Float) => Value::F32x4([0.0; 4]),
                    (VectorSize::Bi,   ScalarKind::Sint)  => Value::I32x2([0; 2]),
                    (VectorSize::Tri,  ScalarKind::Sint)  => Value::I32x3([0; 3]),
                    (VectorSize::Quad, ScalarKind::Sint)  => Value::I32x4([0; 4]),
                    (VectorSize::Bi,   ScalarKind::Uint)  => Value::U32x2([0; 2]),
                    (VectorSize::Tri,  ScalarKind::Uint)  => Value::U32x3([0; 3]),
                    (VectorSize::Quad, ScalarKind::Uint)  => Value::U32x4([0; 4]),
                    _ => Value::Uninitialized,
                }
            }
            TypeInner::Array { .. } => Value::Array(Vec::new()),
            _ => Value::Uninitialized,
        }
    }
}
