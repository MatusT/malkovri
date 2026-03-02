use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};
use naga::TypeInner;

#[derive(Clone, Debug)]
pub enum Primitive {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    F32x2([f32; 2]),
    F32x3([f32; 3]),
    F32x4([f32; 4]),
    I32x2([i32; 2]),
    I32x3([i32; 3]),
    I32x4([i32; 4]),
    U32x2([u32; 2]),
    U32x3([u32; 3]),
    U32x4([u32; 4]),
}

impl Primitive {
    /// Returns a slice view of the f32 components, or None if this is not an f32 primitive.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Primitive::F32(v)    => Some(std::slice::from_ref(v)),
            Primitive::F32x2(a)  => Some(a),
            Primitive::F32x3(a)  => Some(a),
            Primitive::F32x4(a)  => Some(a),
            _                    => None,
        }
    }

    /// Returns a slice view of the i32 components, or None if this is not an i32 primitive.
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        match self {
            Primitive::I32(v)    => Some(std::slice::from_ref(v)),
            Primitive::I32x2(a)  => Some(a),
            Primitive::I32x3(a)  => Some(a),
            Primitive::I32x4(a)  => Some(a),
            _                    => None,
        }
    }

    /// Returns a slice view of the u32 components, or None if this is not a u32 primitive.
    pub fn as_u32_slice(&self) -> Option<&[u32]> {
        match self {
            Primitive::U32(v)    => Some(std::slice::from_ref(v)),
            Primitive::U32x2(a)  => Some(a),
            Primitive::U32x3(a)  => Some(a),
            Primitive::U32x4(a)  => Some(a),
            _                    => None,
        }
    }

    pub fn component_count(&self) -> usize {
        match self {
            Primitive::F32(_) | Primitive::F64(_) | Primitive::I32(_)
            | Primitive::I64(_) | Primitive::U32(_) | Primitive::U64(_) => 1,
            Primitive::F32x2(_) | Primitive::I32x2(_) | Primitive::U32x2(_) => 2,
            Primitive::F32x3(_) | Primitive::I32x3(_) | Primitive::U32x3(_) => 3,
            Primitive::F32x4(_) | Primitive::I32x4(_) | Primitive::U32x4(_) => 4,
        }
    }

    /// Extract a single scalar component by index. Panics if index is out of bounds.
    pub fn extract_component(&self, index: usize) -> Primitive {
        if let Some(s) = self.as_f32_slice() {
            return s.get(index).copied().map(Primitive::F32)
                .unwrap_or_else(|| panic!("extract_component: index {} out of bounds for {:?}", index, self));
        }
        if let Some(s) = self.as_i32_slice() {
            return s.get(index).copied().map(Primitive::I32)
                .unwrap_or_else(|| panic!("extract_component: index {} out of bounds for {:?}", index, self));
        }
        if let Some(s) = self.as_u32_slice() {
            return s.get(index).copied().map(Primitive::U32)
                .unwrap_or_else(|| panic!("extract_component: index {} out of bounds for {:?}", index, self));
        }
        match (self, index) {
            (Primitive::F64(v), 0) => Primitive::F64(*v),
            (Primitive::I64(v), 0) => Primitive::I64(*v),
            (Primitive::U64(v), 0) => Primitive::U64(*v),
            _ => panic!("extract_component: index {} out of bounds for {:?}", index, self),
        }
    }
}

fn apply_f32(lhs: &Primitive, rhs: &Primitive, f: impl Fn(f32, f32) -> f32) -> Option<Primitive> {
    let a = lhs.as_f32_slice()?;
    let b = rhs.as_f32_slice()?;
    let out: Vec<f32> = if a.len() == b.len() {
        a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect()
    } else if a.len() == 1 {
        b.iter().map(|y| f(a[0], *y)).collect()
    } else if b.len() == 1 {
        a.iter().map(|x| f(*x, b[0])).collect()
    } else {
        panic!("apply_f32 length mismatch: {} vs {}", a.len(), b.len())
    };
    Some(Primitive::from(out.as_slice()))
}

fn apply_i32(lhs: &Primitive, rhs: &Primitive, f: impl Fn(i32, i32) -> i32) -> Option<Primitive> {
    let a = lhs.as_i32_slice()?;
    let b = rhs.as_i32_slice()?;
    let out: Vec<i32> = if a.len() == b.len() {
        a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect()
    } else if a.len() == 1 {
        b.iter().map(|y| f(a[0], *y)).collect()
    } else if b.len() == 1 {
        a.iter().map(|x| f(*x, b[0])).collect()
    } else {
        panic!("apply_i32 length mismatch: {} vs {}", a.len(), b.len())
    };
    Some(Primitive::from(out.as_slice()))
}

fn apply_u32(lhs: &Primitive, rhs: &Primitive, f: impl Fn(u32, u32) -> u32) -> Option<Primitive> {
    let a = lhs.as_u32_slice()?;
    let b = rhs.as_u32_slice()?;
    let out: Vec<u32> = if a.len() == b.len() {
        a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect()
    } else if a.len() == 1 {
        b.iter().map(|y| f(a[0], *y)).collect()
    } else if b.len() == 1 {
        a.iter().map(|x| f(*x, b[0])).collect()
    } else {
        panic!("apply_u32 length mismatch: {} vs {}", a.len(), b.len())
    };
    Some(Primitive::from(out.as_slice()))
}

fn apply3_f32(a: &Primitive, b: &Primitive, c: &Primitive, f: impl Fn(f32, f32, f32) -> f32) -> Option<Primitive> {
    let as_ = a.as_f32_slice()?;
    let bs  = b.as_f32_slice()?;
    let cs  = c.as_f32_slice()?;
    assert!(as_.len() == bs.len() && bs.len() == cs.len(),
        "apply3_f32 length mismatch: {} {} {}", as_.len(), bs.len(), cs.len());
    let out: Vec<f32> = as_.iter().zip(bs).zip(cs).map(|((x, y), z)| f(*x, *y, *z)).collect();
    Some(Primitive::from(out.as_slice()))
}

fn apply3_i32(a: &Primitive, b: &Primitive, c: &Primitive, f: impl Fn(i32, i32, i32) -> i32) -> Option<Primitive> {
    let as_ = a.as_i32_slice()?;
    let bs  = b.as_i32_slice()?;
    let cs  = c.as_i32_slice()?;
    assert!(as_.len() == bs.len() && bs.len() == cs.len(),
        "apply3_i32 length mismatch: {} {} {}", as_.len(), bs.len(), cs.len());
    let out: Vec<i32> = as_.iter().zip(bs).zip(cs).map(|((x, y), z)| f(*x, *y, *z)).collect();
    Some(Primitive::from(out.as_slice()))
}

fn apply3_u32(a: &Primitive, b: &Primitive, c: &Primitive, f: impl Fn(u32, u32, u32) -> u32) -> Option<Primitive> {
    let as_ = a.as_u32_slice()?;
    let bs  = b.as_u32_slice()?;
    let cs  = c.as_u32_slice()?;
    assert!(as_.len() == bs.len() && bs.len() == cs.len(),
        "apply3_u32 length mismatch: {} {} {}", as_.len(), bs.len(), cs.len());
    let out: Vec<u32> = as_.iter().zip(bs).zip(cs).map(|((x, y), z)| f(*x, *y, *z)).collect();
    Some(Primitive::from(out.as_slice()))
}

impl Primitive {
    pub fn map_f32(self, f: impl Fn(f32) -> f32) -> Option<Self> {
        match self {
            Primitive::F32(v)          => Some(Primitive::F32(f(v))),
            Primitive::F32x2([a, b])   => Some(Primitive::F32x2([f(a), f(b)])),
            Primitive::F32x3([a, b, c])   => Some(Primitive::F32x3([f(a), f(b), f(c)])),
            Primitive::F32x4([a, b, c, d]) => Some(Primitive::F32x4([f(a), f(b), f(c), f(d)])),
            _ => None,
        }
    }

    pub fn map_i32(self, f: impl Fn(i32) -> i32) -> Option<Self> {
        match self {
            Primitive::I32(v)          => Some(Primitive::I32(f(v))),
            Primitive::I32x2([a, b])   => Some(Primitive::I32x2([f(a), f(b)])),
            Primitive::I32x3([a, b, c])   => Some(Primitive::I32x3([f(a), f(b), f(c)])),
            Primitive::I32x4([a, b, c, d]) => Some(Primitive::I32x4([f(a), f(b), f(c), f(d)])),
            _ => None,
        }
    }

    pub fn map_u32(self, f: impl Fn(u32) -> u32) -> Option<Self> {
        match self {
            Primitive::U32(v)          => Some(Primitive::U32(f(v))),
            Primitive::U32x2([a, b])   => Some(Primitive::U32x2([f(a), f(b)])),
            Primitive::U32x3([a, b, c])   => Some(Primitive::U32x3([f(a), f(b), f(c)])),
            Primitive::U32x4([a, b, c, d]) => Some(Primitive::U32x4([f(a), f(b), f(c), f(d)])),
            _ => None,
        }
    }

    /// Component-wise unary over f32/i32/u32 scalars and vectors.
    pub fn map_numeric(
        self,
        ff32: impl Fn(f32) -> f32,
        fi32: impl Fn(i32) -> i32,
        fu32: impl Fn(u32) -> u32,
    ) -> Option<Self> {
        if self.as_f32_slice().is_some() {
            self.map_f32(ff32)
        } else if self.as_i32_slice().is_some() {
            self.map_i32(fi32)
        } else if self.as_u32_slice().is_some() {
            self.map_u32(fu32)
        } else {
            None
        }
    }

    pub fn zip_map_f32(self, other: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        apply_f32(&self, &other, f)
            .unwrap_or_else(|| panic!("zip_map_f32: not f32: {:?} and {:?}", self, other))
    }

    pub fn zip_cmp_f32(self, other: Self, f: impl Fn(f32, f32) -> bool) -> Self {
        let a = self.as_f32_slice().unwrap_or_else(|| panic!("zip_cmp_f32: lhs is not f32"));
        let b = other.as_f32_slice().unwrap_or_else(|| panic!("zip_cmp_f32: rhs is not f32"));
        assert_eq!(a.len(), b.len(), "zip_cmp_f32 length mismatch: {} vs {}", a.len(), b.len());
        let out: Vec<u32> = a.iter().zip(b).map(|(x, y)| u32::from(f(*x, *y))).collect();
        Primitive::from(out.as_slice())
    }

    pub fn zip_map_i32(self, other: Self, f: impl Fn(i32, i32) -> i32) -> Self {
        apply_i32(&self, &other, f)
            .unwrap_or_else(|| panic!("zip_map_i32: not i32: {:?} and {:?}", self, other))
    }

    pub fn zip_cmp_i32(self, other: Self, f: impl Fn(i32, i32) -> bool) -> Self {
        let a = self.as_i32_slice().unwrap_or_else(|| panic!("zip_cmp_i32: lhs is not i32"));
        let b = other.as_i32_slice().unwrap_or_else(|| panic!("zip_cmp_i32: rhs is not i32"));
        assert_eq!(a.len(), b.len(), "zip_cmp_i32 length mismatch: {} vs {}", a.len(), b.len());
        let out: Vec<u32> = a.iter().zip(b).map(|(x, y)| u32::from(f(*x, *y))).collect();
        Primitive::from(out.as_slice())
    }

    pub fn zip_map_u32(self, other: Self, f: impl Fn(u32, u32) -> u32) -> Self {
        apply_u32(&self, &other, f)
            .unwrap_or_else(|| panic!("zip_map_u32: not u32: {:?} and {:?}", self, other))
    }

    pub fn zip_cmp_u32(self, other: Self, f: impl Fn(u32, u32) -> bool) -> Self {
        let a = self.as_u32_slice().unwrap_or_else(|| panic!("zip_cmp_u32: lhs is not u32"));
        let b = other.as_u32_slice().unwrap_or_else(|| panic!("zip_cmp_u32: rhs is not u32"));
        assert_eq!(a.len(), b.len(), "zip_cmp_u32 length mismatch: {} vs {}", a.len(), b.len());
        let out: Vec<u32> = a.iter().zip(b).map(|(x, y)| u32::from(f(*x, *y))).collect();
        Primitive::from(out.as_slice())
    }

    /// Component-wise binary over f32/i32/u32, with scalar-broadcast support.
    pub fn zip_map_numeric(
        self,
        other: Self,
        ff32: impl Fn(f32, f32) -> f32,
        fi32: impl Fn(i32, i32) -> i32,
        fu32: impl Fn(u32, u32) -> u32,
    ) -> Option<Self> {
        apply_f32(&self, &other, ff32)
            .or_else(|| apply_i32(&self, &other, fi32))
            .or_else(|| apply_u32(&self, &other, fu32))
    }

    /// Component-wise ternary over f32/i32/u32.
    pub fn zip3_map_numeric(
        self,
        b: Self,
        c: Self,
        ff32: impl Fn(f32, f32, f32) -> f32,
        fi32: impl Fn(i32, i32, i32) -> i32,
        fu32: impl Fn(u32, u32, u32) -> u32,
    ) -> Option<Self> {
        apply3_f32(&self, &b, &c, ff32)
            .or_else(|| apply3_i32(&self, &b, &c, fi32))
            .or_else(|| apply3_u32(&self, &b, &c, fu32))
    }

    // ── Component collection (flattening a slice of Primitives) ───────────────

    pub fn collect_f32(vals: &[Primitive]) -> Vec<f32> {
        vals.iter()
            .filter_map(|p| p.as_f32_slice())
            .flat_map(|s| s.iter().copied())
            .collect()
    }

    pub fn collect_i32(vals: &[Primitive]) -> Vec<i32> {
        vals.iter()
            .filter_map(|p| p.as_i32_slice())
            .flat_map(|s| s.iter().copied())
            .collect()
    }

    pub fn collect_u32(vals: &[Primitive]) -> Vec<u32> {
        vals.iter()
            .filter_map(|p| p.as_u32_slice())
            .flat_map(|s| s.iter().copied())
            .collect()
    }
}

impl From<&[f32]> for Primitive {
    fn from(s: &[f32]) -> Self {
        match s.len() {
            1 => Primitive::F32(s[0]),
            2 => Primitive::F32x2([s[0], s[1]]),
            3 => Primitive::F32x3([s[0], s[1], s[2]]),
            4 => Primitive::F32x4([s[0], s[1], s[2], s[3]]),
            _ => panic!("invalid slice length {} for Primitive::from(&[f32])", s.len()),
        }
    }
}

impl From<&[i32]> for Primitive {
    fn from(s: &[i32]) -> Self {
        match s.len() {
            1 => Primitive::I32(s[0]),
            2 => Primitive::I32x2([s[0], s[1]]),
            3 => Primitive::I32x3([s[0], s[1], s[2]]),
            4 => Primitive::I32x4([s[0], s[1], s[2], s[3]]),
            _ => panic!("invalid slice length {} for Primitive::from(&[i32])", s.len()),
        }
    }
}

impl From<&[u32]> for Primitive {
    fn from(s: &[u32]) -> Self {
        match s.len() {
            1 => Primitive::U32(s[0]),
            2 => Primitive::U32x2([s[0], s[1]]),
            3 => Primitive::U32x3([s[0], s[1], s[2]]),
            4 => Primitive::U32x4([s[0], s[1], s[2], s[3]]),
            _ => panic!("invalid slice length {} for Primitive::from(&[u32])", s.len()),
        }
    }
}

impl From<&Vec<f32>> for Primitive { fn from(v: &Vec<f32>) -> Self { Primitive::from(v.as_slice()) } }
impl From<&Vec<i32>> for Primitive { fn from(v: &Vec<i32>) -> Self { Primitive::from(v.as_slice()) } }
impl From<&Vec<u32>> for Primitive { fn from(v: &Vec<u32>) -> Self { Primitive::from(v.as_slice()) } }

impl TryFrom<&TypeInner> for Primitive {
    type Error = ();

    fn try_from(ty: &TypeInner) -> Result<Self, ()> {
        use naga::{Scalar, ScalarKind, VectorSize};
        match ty {
            TypeInner::Scalar(Scalar { kind: ScalarKind::Float, width: 4 }) => Ok(Primitive::F32(0.0)),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Float, width: 8 }) => Ok(Primitive::F64(0.0)),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Sint,  width: 4 }) => Ok(Primitive::I32(0)),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Sint,  width: 8 }) => Ok(Primitive::I64(0)),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Uint,  width: 4 }) => Ok(Primitive::U32(0)),
            TypeInner::Scalar(Scalar { kind: ScalarKind::Uint,  width: 8 }) => Ok(Primitive::U64(0)),
            TypeInner::Vector { size, scalar: Scalar { kind, width: 4 } } => match (size, kind) {
                (VectorSize::Bi,   ScalarKind::Float) => Ok(Primitive::F32x2([0.0; 2])),
                (VectorSize::Tri,  ScalarKind::Float) => Ok(Primitive::F32x3([0.0; 3])),
                (VectorSize::Quad, ScalarKind::Float) => Ok(Primitive::F32x4([0.0; 4])),
                (VectorSize::Bi,   ScalarKind::Sint)  => Ok(Primitive::I32x2([0; 2])),
                (VectorSize::Tri,  ScalarKind::Sint)  => Ok(Primitive::I32x3([0; 3])),
                (VectorSize::Quad, ScalarKind::Sint)  => Ok(Primitive::I32x4([0; 4])),
                (VectorSize::Bi,   ScalarKind::Uint)  => Ok(Primitive::U32x2([0; 2])),
                (VectorSize::Tri,  ScalarKind::Uint)  => Ok(Primitive::U32x3([0; 3])),
                (VectorSize::Quad, ScalarKind::Uint)  => Ok(Primitive::U32x4([0; 4])),
                _ => Err(()),
            },
            _ => Err(()),
        }
    }
}
impl IntoIterator for Primitive {
    type Item = Primitive;
    type IntoIter = std::vec::IntoIter<Primitive>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Primitive::F32(v)             => vec![Primitive::F32(v)],
            Primitive::F32x2([a, b])      => vec![Primitive::F32(a), Primitive::F32(b)],
            Primitive::F32x3([a, b, c])   => vec![Primitive::F32(a), Primitive::F32(b), Primitive::F32(c)],
            Primitive::F32x4([a, b, c, d]) => vec![Primitive::F32(a), Primitive::F32(b), Primitive::F32(c), Primitive::F32(d)],
            Primitive::I32(v)             => vec![Primitive::I32(v)],
            Primitive::I32x2([a, b])      => vec![Primitive::I32(a), Primitive::I32(b)],
            Primitive::I32x3([a, b, c])   => vec![Primitive::I32(a), Primitive::I32(b), Primitive::I32(c)],
            Primitive::I32x4([a, b, c, d]) => vec![Primitive::I32(a), Primitive::I32(b), Primitive::I32(c), Primitive::I32(d)],
            Primitive::U32(v)             => vec![Primitive::U32(v)],
            Primitive::U32x2([a, b])      => vec![Primitive::U32(a), Primitive::U32(b)],
            Primitive::U32x3([a, b, c])   => vec![Primitive::U32(a), Primitive::U32(b), Primitive::U32(c)],
            Primitive::U32x4([a, b, c, d]) => vec![Primitive::U32(a), Primitive::U32(b), Primitive::U32(c), Primitive::U32(d)],
            Primitive::F64(v) => vec![Primitive::F64(v)],
            Primitive::I64(v) => vec![Primitive::I64(v)],
            Primitive::U64(v) => vec![Primitive::U64(v)],
        }
        .into_iter()
    }
}

impl PartialEq for Primitive {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Primitive::F32(a),      Primitive::F32(b))      => a == b,
            (Primitive::F64(a),      Primitive::F64(b))      => a == b,
            (Primitive::I32(a),      Primitive::I32(b))      => a == b,
            (Primitive::I64(a),      Primitive::I64(b))      => a == b,
            (Primitive::U32(a),      Primitive::U32(b))      => a == b,
            (Primitive::U64(a),      Primitive::U64(b))      => a == b,
            (Primitive::F32x2(a),    Primitive::F32x2(b))    => a == b,
            (Primitive::F32x3(a),    Primitive::F32x3(b))    => a == b,
            (Primitive::F32x4(a),    Primitive::F32x4(b))    => a == b,
            (Primitive::I32x2(a),    Primitive::I32x2(b))    => a == b,
            (Primitive::I32x3(a),    Primitive::I32x3(b))    => a == b,
            (Primitive::I32x4(a),    Primitive::I32x4(b))    => a == b,
            (Primitive::U32x2(a),    Primitive::U32x2(b))    => a == b,
            (Primitive::U32x3(a),    Primitive::U32x3(b))    => a == b,
            (Primitive::U32x4(a),    Primitive::U32x4(b))    => a == b,
            _ => false,
        }
    }
}

impl PartialOrd for Primitive {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Primitive::F32(a), Primitive::F32(b)) => a.partial_cmp(b),
            (Primitive::F64(a), Primitive::F64(b)) => a.partial_cmp(b),
            (Primitive::I32(a), Primitive::I32(b)) => a.partial_cmp(b),
            (Primitive::I64(a), Primitive::I64(b)) => a.partial_cmp(b),
            (Primitive::U32(a), Primitive::U32(b)) => a.partial_cmp(b),
            (Primitive::U64(a), Primitive::U64(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl Add for Primitive {
    type Output = Primitive;
    fn add(self, rhs: Self) -> Primitive {
        if let Some(p) = apply_f32(&self, &rhs, |a, b| a + b)                          { return p; }
        if let Some(p) = apply_i32(&self, &rhs, |a, b| a.wrapping_add(b))              { return p; }
        if let Some(p) = apply_u32(&self, &rhs, |a, b| a.wrapping_add(b))              { return p; }
        match (&self, &rhs) {
            (Primitive::F64(a), Primitive::F64(b)) => Primitive::F64(a + b),
            (Primitive::I64(a), Primitive::I64(b)) => Primitive::I64(a.wrapping_add(*b)),
            (Primitive::U64(a), Primitive::U64(b)) => Primitive::U64(a.wrapping_add(*b)),
            _ => panic!("Primitive::add type mismatch: {:?} + {:?}", self, rhs),
        }
    }
}

impl Sub for Primitive {
    type Output = Primitive;
    fn sub(self, rhs: Self) -> Primitive {
        if let Some(p) = apply_f32(&self, &rhs, |a, b| a - b)                          { return p; }
        if let Some(p) = apply_i32(&self, &rhs, |a, b| a.wrapping_sub(b))              { return p; }
        if let Some(p) = apply_u32(&self, &rhs, |a, b| a.wrapping_sub(b))              { return p; }
        match (&self, &rhs) {
            (Primitive::F64(a), Primitive::F64(b)) => Primitive::F64(a - b),
            (Primitive::I64(a), Primitive::I64(b)) => Primitive::I64(a.wrapping_sub(*b)),
            (Primitive::U64(a), Primitive::U64(b)) => Primitive::U64(a.wrapping_sub(*b)),
            _ => panic!("Primitive::sub type mismatch: {:?} - {:?}", self, rhs),
        }
    }
}

impl Mul for Primitive {
    type Output = Primitive;
    fn mul(self, rhs: Self) -> Primitive {
        if let Some(p) = apply_f32(&self, &rhs, |a, b| a * b)                          { return p; }
        if let Some(p) = apply_i32(&self, &rhs, |a, b| a.wrapping_mul(b))              { return p; }
        if let Some(p) = apply_u32(&self, &rhs, |a, b| a.wrapping_mul(b))              { return p; }
        match (&self, &rhs) {
            (Primitive::F64(a), Primitive::F64(b)) => Primitive::F64(a * b),
            (Primitive::I64(a), Primitive::I64(b)) => Primitive::I64(a.wrapping_mul(*b)),
            (Primitive::U64(a), Primitive::U64(b)) => Primitive::U64(a.wrapping_mul(*b)),
            _ => panic!("Primitive::mul type mismatch: {:?} * {:?}", self, rhs),
        }
    }
}

impl Div for Primitive {
    type Output = Primitive;
    fn div(self, rhs: Self) -> Primitive {
        if let Some(p) = apply_f32(&self, &rhs, |a, b| a / b)                                       { return p; }
        if let Some(p) = apply_i32(&self, &rhs, |a, b| a.checked_div(b).unwrap_or(a))               { return p; }
        if let Some(p) = apply_u32(&self, &rhs, |a, b| a.checked_div(b).unwrap_or(a))               { return p; }
        match (&self, &rhs) {
            (Primitive::F64(a), Primitive::F64(b)) => Primitive::F64(a / b),
            (Primitive::I64(a), Primitive::I64(b)) => Primitive::I64(a.checked_div(*b).unwrap_or(*a)),
            (Primitive::U64(a), Primitive::U64(b)) => Primitive::U64(a.checked_div(*b).unwrap_or(*a)),
            _ => panic!("Primitive::div type mismatch: {:?} / {:?}", self, rhs),
        }
    }
}

impl Rem for Primitive {
    type Output = Primitive;
    fn rem(self, rhs: Self) -> Primitive {
        if let Some(p) = apply_f32(&self, &rhs, |a, b| a % b) { return p; }
        if let Some(p) = apply_i32(&self, &rhs, |a, b| {
            a.checked_rem(b).unwrap_or(if b == 0 { a } else { 0 })
        }) { return p; }
        if let Some(p) = apply_u32(&self, &rhs, |a, b| a.checked_rem(b).unwrap_or(a)) { return p; }
        match (&self, &rhs) {
            (Primitive::F64(a), Primitive::F64(b)) => Primitive::F64(a % b),
            (Primitive::I64(a), Primitive::I64(b)) => Primitive::I64(
                a.checked_rem(*b).unwrap_or(if *b == 0 { *a } else { 0 })
            ),
            (Primitive::U64(a), Primitive::U64(b)) => Primitive::U64(a.checked_rem(*b).unwrap_or(*a)),
            _ => panic!("Primitive::rem type mismatch: {:?} % {:?}", self, rhs),
        }
    }
}

impl BitAnd for Primitive {
    type Output = Primitive;
    fn bitand(self, rhs: Self) -> Primitive {
        match (&self, &rhs) {
            (Primitive::I64(a), Primitive::I64(b)) => return Primitive::I64(a & b),
            (Primitive::U64(a), Primitive::U64(b)) => return Primitive::U64(a & b),
            _ => {}
        }
        self.zip_map_numeric(rhs, |_, _| 0.0, |x, y| x & y, |x, y| x & y)
            .unwrap_or_else(|| panic!("Primitive::bitand type mismatch"))
    }
}

impl BitOr for Primitive {
    type Output = Primitive;
    fn bitor(self, rhs: Self) -> Primitive {
        match (&self, &rhs) {
            (Primitive::I64(a), Primitive::I64(b)) => return Primitive::I64(a | b),
            (Primitive::U64(a), Primitive::U64(b)) => return Primitive::U64(a | b),
            _ => {}
        }
        self.zip_map_numeric(rhs, |_, _| 0.0, |x, y| x | y, |x, y| x | y)
            .unwrap_or_else(|| panic!("Primitive::bitor type mismatch"))
    }
}

impl BitXor for Primitive {
    type Output = Primitive;
    fn bitxor(self, rhs: Self) -> Primitive {
        match (&self, &rhs) {
            (Primitive::I64(a), Primitive::I64(b)) => return Primitive::I64(a ^ b),
            (Primitive::U64(a), Primitive::U64(b)) => return Primitive::U64(a ^ b),
            _ => {}
        }
        self.zip_map_numeric(rhs, |_, _| 0.0, |x, y| x ^ y, |x, y| x ^ y)
            .unwrap_or_else(|| panic!("Primitive::bitxor type mismatch"))
    }
}

impl Shl for Primitive {
    type Output = Primitive;
    fn shl(self, rhs: Self) -> Primitive {
        // Scalar cross-type: WGSL shift amount is always u32
        match (&self, &rhs) {
            (Primitive::I32(l), Primitive::U32(r)) => return Primitive::I32(l.wrapping_shl(*r)),
            (Primitive::I64(l), Primitive::U32(r)) => return Primitive::I64(l.wrapping_shl(*r)),
            (Primitive::U64(l), Primitive::U32(r)) => return Primitive::U64(l.wrapping_shl(*r)),
            _ => {}
        }
        // Vector cross-type: I32xN << U32xN or I32xN << U32 (scalar broadcast)
        if let (Some(a), Some(b)) = (self.as_i32_slice(), rhs.as_u32_slice()) {
            let a = a.to_owned();
            let b = b.to_owned();
            let out: Vec<i32> = if a.len() == b.len() {
                a.iter().zip(&b).map(|(x, y)| x.wrapping_shl(*y)).collect()
            } else if b.len() == 1 {
                a.iter().map(|x| x.wrapping_shl(b[0])).collect()
            } else {
                panic!("Primitive::shl length mismatch: {:?} << {:?}", self, rhs)
            };
            return Primitive::from(out.as_slice());
        }
        // Same-type: U32/U32xN
        if let Some(p) = apply_u32(&self, &rhs, |x, y| x.wrapping_shl(y)) { return p; }
        panic!("Primitive::shl type mismatch: {:?} << {:?}", self, rhs)
    }
}

impl Shr for Primitive {
    type Output = Primitive;
    fn shr(self, rhs: Self) -> Primitive {
        // Scalar cross-type: WGSL shift amount is always u32
        match (&self, &rhs) {
            (Primitive::I32(l), Primitive::U32(r)) => return Primitive::I32(l.wrapping_shr(*r)),
            (Primitive::I64(l), Primitive::U32(r)) => return Primitive::I64(l.wrapping_shr(*r)),
            (Primitive::U64(l), Primitive::U32(r)) => return Primitive::U64(l.wrapping_shr(*r)),
            _ => {}
        }
        // Vector cross-type: I32xN >> U32xN or I32xN >> U32 (scalar broadcast)
        if let (Some(a), Some(b)) = (self.as_i32_slice(), rhs.as_u32_slice()) {
            let a = a.to_owned();
            let b = b.to_owned();
            let out: Vec<i32> = if a.len() == b.len() {
                a.iter().zip(&b).map(|(x, y)| x.wrapping_shr(*y)).collect()
            } else if b.len() == 1 {
                a.iter().map(|x| x.wrapping_shr(b[0])).collect()
            } else {
                panic!("Primitive::shr length mismatch: {:?} >> {:?}", self, rhs)
            };
            return Primitive::from(out.as_slice());
        }
        // Same-type: U32/U32xN
        if let Some(p) = apply_u32(&self, &rhs, |x, y| x.wrapping_shr(y)) { return p; }
        panic!("Primitive::shr type mismatch: {:?} >> {:?}", self, rhs)
    }
}
