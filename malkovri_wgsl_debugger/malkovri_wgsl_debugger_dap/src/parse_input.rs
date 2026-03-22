use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
use std::{fs, path::Path};

use crate::error::DebugAdapterError;
use malkovri_wgsl_debugger::{EntryPointInputs, Primitive, ResourceBinding, Value};

pub fn parse_shader_inputs(
    arguments: &serde_json::Map<String, serde_json::Value>,
) -> Result<EntryPointInputs, DebugAdapterError> {
    match arguments.get("shaderInputs") {
        Some(value) => Ok(serde_json::from_value(value.clone())?),
        None => Ok(EntryPointInputs::default()),
    }
}

pub fn parse_bindings(
    arguments: &serde_json::Map<String, serde_json::Value>,
    #[cfg(not(target_arch = "wasm32"))] program_dir: &Path,
) -> Result<HashMap<ResourceBinding, Value>, DebugAdapterError> {
    let Some(bindings) = arguments.get("bindings").and_then(|v| v.as_object()) else {
        return Ok(HashMap::new());
    };

    bindings
        .iter()
        .map(|(key, config)| {
            let (group, binding) = parse_binding_key(key)?;

            let obj = config.as_object().ok_or_else(|| {
                DebugAdapterError::Parse(format!("Binding '{key}' is not an object"))
            })?;

            let type_str = obj.get("type").and_then(|v| v.as_str()).unwrap_or("f32");
            let value = if let Some(inline) = obj.get("inline") {
                parse_inline(key, type_str, inline)?
            } else if let Some(content) = obj.get("fileContent").and_then(|v| v.as_str()) {
                let format = obj.get("format").and_then(|v| v.as_str()).unwrap_or("ron");
                parse_file_content(key, type_str, format, content)?
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(path) = obj.get("file").and_then(|v| v.as_str()) {
                    let format = obj.get("format").and_then(|v| v.as_str()).unwrap_or("ron");
                    parse_file(key, type_str, format, &program_dir.join(path))?
                } else {
                    return Err(DebugAdapterError::Parse(format!(
                        "Binding '{key}' has neither 'inline' nor 'file'"
                    )));
                }
                #[cfg(target_arch = "wasm32")]
                {
                    return Err(DebugAdapterError::Parse(format!(
                        "Binding '{key}' missing 'inline' data (file bindings not supported in WASM)"
                    )));
                }
            };

            Ok((ResourceBinding { group, binding }, value))
        })
        .collect()
}

fn parse_binding_key(key: &str) -> Result<(u32, u32), DebugAdapterError> {
    let (group_str, binding_str) = key.split_once(':').ok_or_else(|| {
        DebugAdapterError::Parse(format!(
            "Invalid binding key '{key}': expected 'group:binding'"
        ))
    })?;
    let group = group_str.parse::<u32>().map_err(|_| {
        DebugAdapterError::Parse(format!("Invalid group in binding key '{key}'"))
    })?;
    let binding = binding_str.parse::<u32>().map_err(|_| {
        DebugAdapterError::Parse(format!("Invalid binding in binding key '{key}'"))
    })?;
    Ok((group, binding))
}

fn parse_inline(
    key: &str,
    type_str: &str,
    inline: &serde_json::Value,
) -> Result<Value, DebugAdapterError> {
    let arr = inline.as_array().ok_or_else(|| {
        DebugAdapterError::Parse(format!("Binding '{key}' inline value is not an array"))
    })?;
    typed_array_from_json(key, type_str, arr)
}

fn typed_array_from_json(
    key: &str,
    type_str: &str,
    arr: &[serde_json::Value],
) -> Result<Value, DebugAdapterError> {
    Ok(match type_str {
        "f32" => Value::Array(
            arr.iter()
                .map(|v| Primitive::F32(v.as_f64().unwrap_or(0.0) as f32).into())
                .collect(),
        ),
        "i32" => Value::Array(
            arr.iter()
                .map(|v| Primitive::I32(v.as_i64().unwrap_or(0) as i32).into())
                .collect(),
        ),
        "u32" => Value::Array(
            arr.iter()
                .map(|v| Primitive::U32(v.as_u64().unwrap_or(0) as u32).into())
                .collect(),
        ),
        _ => {
            return Err(DebugAdapterError::Parse(format!(
                "Unknown type '{type_str}' for binding '{key}'"
            )));
        }
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn typed_array_from_bytes(
    key: &str,
    type_str: &str,
    bytes: &[u8],
) -> Result<Value, DebugAdapterError> {
    Ok(match type_str {
        "f32" => Value::Array(
            bytes
                .chunks_exact(4)
                .map(|c| Primitive::F32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).into())
                .collect(),
        ),
        "i32" => Value::Array(
            bytes
                .chunks_exact(4)
                .map(|c| Primitive::I32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])).into())
                .collect(),
        ),
        "u32" => Value::Array(
            bytes
                .chunks_exact(4)
                .map(|c| Primitive::U32(u32::from_le_bytes([c[0], c[1], c[2], c[3]])).into())
                .collect(),
        ),
        _ => {
            return Err(DebugAdapterError::Parse(format!(
                "Unknown type '{type_str}' for binding '{key}'"
            )));
        }
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_file(
    key: &str,
    type_str: &str,
    format: &str,
    path: &Path,
) -> Result<Value, DebugAdapterError> {
    match format {
        "binary" => {
            let bytes = fs::read(path)?;
            typed_array_from_bytes(key, type_str, &bytes)
        }
        "ron" => {
            let content = fs::read_to_string(path)?;
            parse_ron(key, type_str, &content)
        }
        _ => Err(DebugAdapterError::Parse(format!(
            "Unknown format '{format}' for binding '{key}'"
        ))),
    }
}

fn parse_file_content(
    key: &str,
    type_str: &str,
    format: &str,
    content: &str,
) -> Result<Value, DebugAdapterError> {
    match format {
        "ron" => parse_ron(key, type_str, content),
        other => Err(DebugAdapterError::Parse(format!(
            "Format '{other}' with fileContent is not supported for binding '{key}'; use 'inline' instead"
        ))),
    }
}

fn parse_ron(key: &str, type_str: &str, content: &str) -> Result<Value, DebugAdapterError> {
    let ron_err = |e| DebugAdapterError::Parse(format!("RON parse error for binding '{key}': {e}"));
    Ok(match type_str {
        "f32" => {
            let vals: Vec<f64> = ron::from_str(content).map_err(ron_err)?;
            Value::Array(
                vals.into_iter()
                    .map(|v| Primitive::F32(v as f32).into())
                    .collect(),
            )
        }
        "i32" => {
            let vals: Vec<i64> = ron::from_str(content).map_err(ron_err)?;
            Value::Array(
                vals.into_iter()
                    .map(|v| Primitive::I32(v as i32).into())
                    .collect(),
            )
        }
        "u32" => {
            let vals: Vec<u64> = ron::from_str(content).map_err(ron_err)?;
            Value::Array(
                vals.into_iter()
                    .map(|v| Primitive::U32(v as u32).into())
                    .collect(),
            )
        }
        _ => {
            return Err(DebugAdapterError::Parse(format!(
                "Unknown type '{type_str}' for binding '{key}'"
            )));
        }
    })
}
