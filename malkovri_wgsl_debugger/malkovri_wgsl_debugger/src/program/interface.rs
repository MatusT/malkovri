use std::collections::BTreeMap;

use naga::{Binding, Handle, Interpolation, Module, Sampling, ScalarKind, Type, TypeInner};

use crate::{Primitive, ShaderProgram, Value};

/// Portable type of a supported user-defined shader input/output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderIoType {
    pub kind: ScalarKind,
    pub components: u8,
}

impl ShaderIoType {
    pub(crate) fn from_type(module: &Module, ty: Handle<Type>) -> Result<Self, String> {
        let (scalar, components) = match module.types[ty].inner {
            TypeInner::Scalar(scalar) => (scalar, 1),
            TypeInner::Vector { scalar, size } => (scalar, size as u8),
            ref other => return Err(format!("unsupported shader IO type {other:?}")),
        };
        if scalar.width != 4
            || !matches!(
                scalar.kind,
                ScalarKind::Float | ScalarKind::Sint | ScalarKind::Uint
            )
        {
            return Err(format!("unsupported shader IO scalar {scalar:?}"));
        }
        Ok(Self {
            kind: scalar.kind,
            components,
        })
    }

    pub fn validate(self, value: &Value) -> Result<(), String> {
        let valid = value.as_primitive().is_some_and(|p| match self.kind {
            ScalarKind::Float => p.as_f32_slice().is_some_and(|v| {
                v.len() == self.components as usize && v.iter().all(|v| v.is_finite())
            }),
            ScalarKind::Sint => p
                .as_i32_slice()
                .is_some_and(|v| v.len() == self.components as usize),
            ScalarKind::Uint => p
                .as_u32_slice()
                .is_some_and(|v| v.len() == self.components as usize),
            _ => false,
        });
        if valid {
            Ok(())
        } else {
            Err(format!("expected {self:?}, got {value:?}"))
        }
    }

    /// Decode a JSON scalar or vector without losing integer range information.
    pub fn parse_json(self, value: &serde_json::Value) -> Result<Value, String> {
        let values = if self.components == 1 {
            vec![value]
        } else {
            value
                .as_array()
                .ok_or("expected a vector array")?
                .iter()
                .collect()
        };
        if values.len() != self.components as usize {
            return Err(format!("expected {} components", self.components));
        }
        let bad = || format!("invalid value for {self:?}: {value}");
        let primitive = match self.kind {
            ScalarKind::Float => Primitive::from(
                &values
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|v| v as f32)
                            .filter(|v| v.is_finite())
                            .ok_or_else(bad)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ScalarKind::Sint => Primitive::from(
                &values
                    .iter()
                    .map(|v| {
                        v.as_i64()
                            .and_then(|v| i32::try_from(v).ok())
                            .ok_or_else(bad)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ScalarKind::Uint => Primitive::from(
                &values
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|v| u32::try_from(v).ok())
                            .ok_or_else(bad)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            _ => return Err(bad()),
        };
        Ok(primitive.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocationInput {
    pub location: u32,
    pub ty: ShaderIoType,
    pub interpolation: Option<Interpolation>,
    pub sampling: Option<Sampling>,
}

pub(crate) fn location_fields(
    module: &Module,
    ty: Handle<Type>,
    binding: Option<&Binding>,
    fields: &mut Vec<LocationInput>,
) -> Result<(), String> {
    match binding {
        Some(Binding::Location {
            location,
            interpolation,
            sampling,
            ..
        }) => fields.push(LocationInput {
            location: *location,
            ty: ShaderIoType::from_type(module, ty)?,
            interpolation: *interpolation,
            sampling: *sampling,
        }),
        Some(Binding::BuiltIn(_)) => {}
        None => {
            if let TypeInner::Struct { members, .. } = &module.types[ty].inner {
                for member in members {
                    location_fields(module, member.ty, member.binding.as_ref(), fields)?;
                }
            }
        }
    }
    Ok(())
}

impl ShaderProgram {
    pub fn input_locations(&self, entry: usize) -> Result<Vec<LocationInput>, String> {
        let entry = self
            .module()
            .entry_points
            .get(entry)
            .ok_or("invalid entry point")?;
        let mut fields = Vec::new();
        for arg in &entry.function.arguments {
            location_fields(self.module(), arg.ty, arg.binding.as_ref(), &mut fields)?;
        }
        Ok(fields)
    }

    pub fn parse_location_values(
        &self,
        entry: usize,
        values: &BTreeMap<u32, serde_json::Value>,
    ) -> Result<BTreeMap<u32, Value>, String> {
        let fields = self.input_locations(entry)?;
        if values
            .keys()
            .any(|key| !fields.iter().any(|f| f.location == *key))
        {
            return Err("unknown input location".into());
        }
        fields
            .iter()
            .map(|f| {
                let value = values
                    .get(&f.location)
                    .ok_or_else(|| format!("missing @location({})", f.location))?;
                Ok((
                    f.location,
                    f.ty.parse_json(value)
                        .map_err(|e| format!("@location({}): {e}", f.location))?,
                ))
            })
            .collect()
    }
}
