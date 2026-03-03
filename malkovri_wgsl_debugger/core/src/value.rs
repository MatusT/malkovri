use std::{cell::RefCell, rc::Rc};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};
use naga::TypeInner;

use crate::primitive::Primitive;

#[derive(Clone, Debug, Default)]
pub enum Value {
    #[default]
    Uninitialized,
    Primitive(Primitive),
    Array(Vec<Value>),
    /// Named fields in declaration order.
    Struct(Vec<(String, Value)>),
    Pointer(Rc<RefCell<Value>>),
}

// ── Core accessors ─────────────────────────────────────────────────────────────

impl Value {
    pub fn leaf_value(&self) -> Value {
        match self {
            Value::Pointer(inner) => inner.borrow().leaf_value(),
            _ => self.clone(),
        }
    }

    pub fn as_primitive(&self) -> Option<&Primitive> {
        match self {
            Value::Primitive(p) => Some(p),
            _ => None,
        }
    }

    /// Index into a composite value (array, struct, or vector) by position.
    pub fn index_into(&self, index: usize) -> Value {
        match self {
            Value::Array(elements) => elements.get(index).cloned().unwrap_or(Value::Uninitialized),
            Value::Struct(fields) => fields
                .get(index)
                .map(|(_, v)| v.clone())
                .unwrap_or(Value::Uninitialized),
            Value::Primitive(p) => Value::Primitive(p.extract_component(index)),
            _ => Value::Uninitialized,
        }
    }
}

// ── Delegation to Primitive — keeps the evaluator's helper call-sites unchanged ─

impl Value {
    pub fn component_count(&self) -> usize {
        self.as_primitive().map_or(0, Primitive::component_count)
    }

    pub fn extract_component(&self, index: usize) -> Value {
        let p = self.as_primitive()
            .unwrap_or_else(|| panic!("extract_component called on non-primitive: {:?}", self));
        Value::Primitive(p.extract_component(index))
    }

    pub fn map_f32(self, f: impl Fn(f32) -> f32) -> Value {
        match self {
            Value::Primitive(p) => Value::Primitive(
                p.map_f32(f).unwrap_or_else(|| panic!("map_f32 called on non-f32 primitive"))
            ),
            other => other,
        }
    }

    pub fn map_i32(self, f: impl Fn(i32) -> i32) -> Value {
        match self {
            Value::Primitive(p) => Value::Primitive(
                p.map_i32(f).unwrap_or_else(|| panic!("map_i32 called on non-i32 primitive"))
            ),
            other => other,
        }
    }

    pub fn map_u32(self, f: impl Fn(u32) -> u32) -> Value {
        match self {
            Value::Primitive(p) => Value::Primitive(
                p.map_u32(f).unwrap_or_else(|| panic!("map_u32 called on non-u32 primitive"))
            ),
            other => other,
        }
    }

    pub fn map_numeric(
        self,
        ff32: impl Fn(f32) -> f32,
        fi32: impl Fn(i32) -> i32,
        fu32: impl Fn(u32) -> u32,
    ) -> Value {
        match self {
            Value::Primitive(p) => Value::Primitive(
                p.map_numeric(ff32, fi32, fu32)
                    .unwrap_or_else(|| panic!("map_numeric called on unsupported primitive type"))
            ),
            other => other,
        }
    }

    pub fn zip_map_f32(self, other: Value, f: impl Fn(f32, f32) -> f32) -> Value {
        match (self, other) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a.zip_map_f32(b, f)),
            (a, b) => panic!("zip_map_f32: expected two primitives, got {:?} and {:?}", a, b),
        }
    }

    pub fn zip_cmp_f32(self, other: Value, f: impl Fn(f32, f32) -> bool) -> Value {
        match (self, other) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a.zip_cmp_f32(b, f)),
            (a, b) => panic!("zip_cmp_f32: expected two primitives, got {:?} and {:?}", a, b),
        }
    }

    pub fn zip_map_i32(self, other: Value, f: impl Fn(i32, i32) -> i32) -> Value {
        match (self, other) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a.zip_map_i32(b, f)),
            (a, b) => panic!("zip_map_i32: expected two primitives, got {:?} and {:?}", a, b),
        }
    }

    pub fn zip_cmp_i32(self, other: Value, f: impl Fn(i32, i32) -> bool) -> Value {
        match (self, other) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a.zip_cmp_i32(b, f)),
            (a, b) => panic!("zip_cmp_i32: expected two primitives, got {:?} and {:?}", a, b),
        }
    }

    pub fn zip_map_u32(self, other: Value, f: impl Fn(u32, u32) -> u32) -> Value {
        match (self, other) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a.zip_map_u32(b, f)),
            (a, b) => panic!("zip_map_u32: expected two primitives, got {:?} and {:?}", a, b),
        }
    }

    pub fn zip_cmp_u32(self, other: Value, f: impl Fn(u32, u32) -> bool) -> Value {
        match (self, other) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a.zip_cmp_u32(b, f)),
            (a, b) => panic!("zip_cmp_u32: expected two primitives, got {:?} and {:?}", a, b),
        }
    }

    pub fn zip_map_numeric(
        self,
        other: Value,
        ff32: impl Fn(f32, f32) -> f32,
        fi32: impl Fn(i32, i32) -> i32,
        fu32: impl Fn(u32, u32) -> u32,
    ) -> Value {
        match (self, other) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(
                a.zip_map_numeric(b, ff32, fi32, fu32)
                    .unwrap_or_else(|| panic!("zip_map_numeric: type mismatch"))
            ),
            (a, b) => panic!("zip_map_numeric: expected two primitives, got {:?} and {:?}", a, b),
        }
    }

    pub fn zip3_map_numeric(
        self,
        b: Value,
        c: Value,
        ff32: impl Fn(f32, f32, f32) -> f32,
        fi32: impl Fn(i32, i32, i32) -> i32,
        fu32: impl Fn(u32, u32, u32) -> u32,
    ) -> Value {
        match (self, b, c) {
            (Value::Primitive(a), Value::Primitive(b), Value::Primitive(c)) => Value::Primitive(
                a.zip3_map_numeric(b, c, ff32, fi32, fu32)
                    .unwrap_or_else(|| panic!("zip3_map_numeric: type mismatch"))
            ),
            (a, b, c) => panic!("zip3_map_numeric: expected three primitives, got {:?}, {:?}, {:?}", a, b, c),
        }
    }

    /// Flatten f32 components from a slice of Values (vectors are expanded).
    pub fn collect_f32_components(vals: &[Value]) -> Vec<f32> {
        vals.iter()
            .filter_map(|v| v.as_primitive())
            .flat_map(|p| p.as_f32_slice().unwrap_or(&[]).iter().copied())
            .collect()
    }

    /// Flatten i32 components from a slice of Values.
    pub fn collect_i32_components(vals: &[Value]) -> Vec<i32> {
        vals.iter()
            .filter_map(|v| v.as_primitive())
            .flat_map(|p| p.as_i32_slice().unwrap_or(&[]).iter().copied())
            .collect()
    }

    /// Flatten u32 components from a slice of Values.
    pub fn collect_u32_components(vals: &[Value]) -> Vec<u32> {
        vals.iter()
            .filter_map(|v| v.as_primitive())
            .flat_map(|p| p.as_u32_slice().unwrap_or(&[]).iter().copied())
            .collect()
    }
}

// ── From conversions ───────────────────────────────────────────────────────────

impl From<Primitive> for Value {
    fn from(p: Primitive) -> Self { Value::Primitive(p) }
}

impl From<&[f32]> for Value {
    fn from(s: &[f32]) -> Self { Value::Primitive(Primitive::from(s)) }
}
impl From<&[i32]> for Value {
    fn from(s: &[i32]) -> Self { Value::Primitive(Primitive::from(s)) }
}
impl From<&[u32]> for Value {
    fn from(s: &[u32]) -> Self { Value::Primitive(Primitive::from(s)) }
}
impl From<&Vec<f32>> for Value { fn from(v: &Vec<f32>) -> Self { Value::from(v.as_slice()) } }
impl From<&Vec<i32>> for Value { fn from(v: &Vec<i32>) -> Self { Value::from(v.as_slice()) } }
impl From<&Vec<u32>> for Value { fn from(v: &Vec<u32>) -> Self { Value::from(v.as_slice()) } }

impl From<&TypeInner> for Value {
    fn from(ty: &TypeInner) -> Self {
        match ty {
            TypeInner::Array { .. }  => Value::Array(Vec::new()),
            TypeInner::Struct { .. } => Value::Struct(Vec::new()),
            _ => Primitive::try_from(ty)
                .map(Value::Primitive)
                .unwrap_or(Value::Uninitialized),
        }
    }
}

// ── IntoIterator ───────────────────────────────────────────────────────────────

impl IntoIterator for Value {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Value>;

    fn into_iter(self) -> Self::IntoIter {
        match self.leaf_value() {
            Value::Array(elements) => elements.into_iter(),
            Value::Primitive(p) => p.into_iter().map(Value::Primitive).collect::<Vec<_>>().into_iter(),
            Value::Struct(fields) => fields.into_iter().map(|(_, v)| v).collect::<Vec<_>>().into_iter(),
            other => vec![other].into_iter(),
        }
    }
}

// ── PartialEq / PartialOrd ─────────────────────────────────────────────────────

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Primitive(a),  Value::Primitive(b))  => a == b,
            (Value::Array(a),      Value::Array(b))      => a == b,
            (Value::Struct(a),     Value::Struct(b))     => a == b,
            (Value::Pointer(a),    Value::Pointer(b))    => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::Primitive(a), Value::Primitive(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

// ── Arithmetic traits — delegate to Primitive ──────────────────────────────────

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Array(a), Value::Array(b)) if a.len() == b.len() =>
                Value::Array(a.into_iter().zip(b).map(|(x, y)| x + y).collect()),
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a + b),
            (a, b) => panic!("Value::add type mismatch: {:?} + {:?}", a, b),
        }
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Array(a), Value::Array(b)) if a.len() == b.len() =>
                Value::Array(a.into_iter().zip(b).map(|(x, y)| x - y).collect()),
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a - b),
            (a, b) => panic!("Value::sub type mismatch: {:?} - {:?}", a, b),
        }
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Array(a), Value::Array(b)) if a.len() == b.len() =>
                Value::Array(a.into_iter().zip(b).map(|(x, y)| x * y).collect()),
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a * b),
            (a, b) => panic!("Value::mul type mismatch: {:?} * {:?}", a, b),
        }
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Array(a), Value::Array(b)) if a.len() == b.len() =>
                Value::Array(a.into_iter().zip(b).map(|(x, y)| x / y).collect()),
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a / b),
            (a, b) => panic!("Value::div type mismatch: {:?} / {:?}", a, b),
        }
    }
}

impl Rem for Value {
    type Output = Value;
    fn rem(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Array(a), Value::Array(b)) if a.len() == b.len() =>
                Value::Array(a.into_iter().zip(b).map(|(x, y)| x % y).collect()),
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a % b),
            (a, b) => panic!("Value::rem type mismatch: {:?} % {:?}", a, b),
        }
    }
}

impl BitAnd for Value {
    type Output = Value;
    fn bitand(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a & b),
            (a, b) => panic!("Value::bitand type mismatch: {:?} & {:?}", a, b),
        }
    }
}

impl BitOr for Value {
    type Output = Value;
    fn bitor(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a | b),
            (a, b) => panic!("Value::bitor type mismatch: {:?} | {:?}", a, b),
        }
    }
}

impl BitXor for Value {
    type Output = Value;
    fn bitxor(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a ^ b),
            (a, b) => panic!("Value::bitxor type mismatch: {:?} ^ {:?}", a, b),
        }
    }
}

impl Shl for Value {
    type Output = Value;
    fn shl(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a << b),
            (a, b) => panic!("Value::shl type mismatch: {:?} << {:?}", a, b),
        }
    }
}

impl Shr for Value {
    type Output = Value;
    fn shr(self, rhs: Self) -> Value {
        match (self.leaf_value(), rhs.leaf_value()) {
            (Value::Primitive(a), Value::Primitive(b)) => Value::Primitive(a >> b),
            (a, b) => panic!("Value::shr type mismatch: {:?} >> {:?}", a, b),
        }
    }
}
