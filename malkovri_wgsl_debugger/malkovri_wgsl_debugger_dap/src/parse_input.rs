use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
use std::{fs, path::Path};

use crate::error::DebugAdapterError;
use malkovri_wgsl_debugger::{
    DrawConfig, ExecutionConfig, GlobalConstants, Primitive, ResourceBinding, ShaderStage, Value,
    WorkgroupConfig,
};

pub fn parse_execution_config(
    arguments: &serde_json::Map<String, serde_json::Value>,
    stage: ShaderStage,
) -> Result<ExecutionConfig, DebugAdapterError> {
    let has_workgroup_config = [
        "workgroupConfig",
        "workgroupSize",
        "workgroupId",
        "subgroupSize",
        "numWorkgroups",
        "size",
        "id",
        "count",
    ]
    .iter()
    .any(|key| arguments.contains_key(*key))
        || arguments.get("shaderInputs").is_some_and(|inputs| {
            inputs.get("global_invocation_id").is_some()
                || inputs.get("globalInvocationId").is_some()
        });
    if stage != ShaderStage::Compute && has_workgroup_config {
        return Err(DebugAdapterError::Parse(
            "workgroupConfig is only supported for compute shaders; use drawConfig for vertex shaders".into(),
        ));
    }
    if stage != ShaderStage::Vertex && arguments.contains_key("drawConfig") {
        return Err(DebugAdapterError::Parse(
            "drawConfig is only supported for vertex shaders".into(),
        ));
    }
    match stage {
        ShaderStage::Compute => Ok(parse_workgroup_config(arguments)?.into()),
        ShaderStage::Vertex => {
            let config = arguments
                .get("drawConfig")
                .map(|value| serde_json::from_value::<DrawConfig>(value.clone()))
                .transpose()?
                .unwrap_or_default();
            config.validate().map_err(DebugAdapterError::Parse)?;
            Ok(config.into())
        }
        ShaderStage::Fragment => Ok(ExecutionConfig::Fragment),
        _ => Err(DebugAdapterError::Parse("unsupported shader stage".into())),
    }
}

pub fn parse_global_constants(
    arguments: &serde_json::Map<String, serde_json::Value>,
) -> Result<GlobalConstants, DebugAdapterError> {
    match arguments.get("globalConstants") {
        Some(value) => Ok(serde_json::from_value(value.clone())?),
        None => Ok(GlobalConstants::default()),
    }
}

#[derive(Default, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct PartialWorkgroupConfig {
    #[serde(alias = "size")]
    workgroup_size: Option<[u32; 3]>,
    #[serde(alias = "id")]
    workgroup_id: Option<[u32; 3]>,
    subgroup_size: Option<u32>,
    #[serde(alias = "count")]
    num_workgroups: Option<[u32; 3]>,
}

impl PartialWorkgroupConfig {
    fn merge(&mut self, other: Self) {
        self.workgroup_size = other.workgroup_size.or(self.workgroup_size);
        self.workgroup_id = other.workgroup_id.or(self.workgroup_id);
        self.subgroup_size = other.subgroup_size.or(self.subgroup_size);
        self.num_workgroups = other.num_workgroups.or(self.num_workgroups);
    }

    fn build(self) -> Result<WorkgroupConfig, DebugAdapterError> {
        let defaults = WorkgroupConfig::default();
        WorkgroupConfig::new(
            self.workgroup_size.unwrap_or(defaults.workgroup_size()),
            self.workgroup_id.unwrap_or(defaults.workgroup_id()),
            self.subgroup_size.unwrap_or(defaults.subgroup_size()),
            self.num_workgroups.unwrap_or(defaults.num_workgroups()),
        )
        .map_err(DebugAdapterError::Parse)
    }
}

pub fn parse_workgroup_config(
    arguments: &serde_json::Map<String, serde_json::Value>,
) -> Result<WorkgroupConfig, DebugAdapterError> {
    let mut config = serde_json::from_value::<PartialWorkgroupConfig>(serde_json::Value::Object(
        arguments.clone(),
    ))?;

    if let Some(value) = arguments.get("workgroupConfig") {
        config.merge(serde_json::from_value(value.clone())?);
    }

    // Backward compatibility with the old single-invocation input form.
    if let Some(global_id) = arguments
        .get("shaderInputs")
        .and_then(|v| v.get("global_invocation_id"))
        .or_else(|| {
            arguments
                .get("shaderInputs")
                .and_then(|v| v.get("globalInvocationId"))
        })
        && config.workgroup_size.unwrap_or([1, 1, 1]) == [1, 1, 1]
        && config.workgroup_id.unwrap_or([0, 0, 0]) == [0, 0, 0]
    {
        config.workgroup_id = Some(serde_json::from_value(global_id.clone())?);
    }

    config.build()
}

pub fn parse_bindings(
    arguments: &serde_json::Map<String, serde_json::Value>,
    #[cfg(not(target_arch = "wasm32"))] program_dir: &Path,
) -> Result<HashMap<ResourceBinding, Value>, DebugAdapterError> {
    let Some(bindings) = arguments.get("bindings") else {
        return Ok(HashMap::new());
    };
    let bindings = bindings
        .as_object()
        .ok_or_else(|| DebugAdapterError::Parse("bindings must be an object".into()))?;

    bindings
        .iter()
        .map(|(key, config)| {
            let (group, binding) = parse_binding_key(key)?;

            let obj = config.as_object().ok_or_else(|| {
                DebugAdapterError::Parse(format!("Binding '{key}' is not an object"))
            })?;

            let source_count = ["inline", "fileContent", "fileBytes", "file"]
                .iter()
                .filter(|name| obj.contains_key(**name))
                .count();
            if source_count != 1 {
                return Err(DebugAdapterError::Parse(format!(
                    "Binding '{key}' must specify exactly one of 'inline', 'fileContent', 'fileBytes', or 'file'"
                )));
            }

            let type_str = obj.get("type").and_then(|v| v.as_str()).unwrap_or("f32");
            let value = if let Some(inline) = obj.get("inline") {
                parse_inline(key, type_str, inline)?
            } else if let Some(bytes) = obj.get("fileBytes") {
                let bytes: Vec<u8> = serde_json::from_value(bytes.clone()).map_err(|error| {
                    DebugAdapterError::Parse(format!("Binding '{key}' fileBytes: {error}"))
                })?;
                typed_array_from_bytes(key, type_str, &bytes)?
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
    let group = group_str
        .parse::<u32>()
        .map_err(|_| DebugAdapterError::Parse(format!("Invalid group in binding key '{key}'")))?;
    let binding = binding_str
        .parse::<u32>()
        .map_err(|_| DebugAdapterError::Parse(format!("Invalid binding in binding key '{key}'")))?;
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
    validate_binding_type(key, type_str)?;
    checked_array(key, type_str, arr.iter(), |value| match type_str {
        "f32" => value
            .as_f64()
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .map(Primitive::F32),
        "i32" => value
            .as_i64()
            .and_then(|value| i32::try_from(value).ok())
            .map(Primitive::I32),
        "u32" => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .map(Primitive::U32),
        _ => unreachable!("binding type was validated"),
    })
}

fn checked_array<T: std::fmt::Debug>(
    key: &str,
    type_str: &str,
    values: impl IntoIterator<Item = T>,
    convert: impl Fn(&T) -> Option<Primitive>,
) -> Result<Value, DebugAdapterError> {
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            convert(&value).map(Value::from).ok_or_else(|| {
                DebugAdapterError::Parse(format!(
                    "Binding '{key}' element {index}: {value:?} is not a valid {type_str}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn validate_binding_type(key: &str, type_str: &str) -> Result<(), DebugAdapterError> {
    match type_str {
        "f32" | "i32" | "u32" => Ok(()),
        _ => Err(DebugAdapterError::Parse(format!(
            "Unknown type '{type_str}' for binding '{key}'"
        ))),
    }
}

fn typed_array_from_bytes(
    key: &str,
    type_str: &str,
    bytes: &[u8],
) -> Result<Value, DebugAdapterError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(DebugAdapterError::Parse(format!(
            "Binding '{key}' binary data has {} bytes; expected a multiple of 4",
            bytes.len()
        )));
    }
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
    match type_str {
        "f32" => {
            let values: Vec<f64> = ron::from_str(content).map_err(ron_err)?;
            checked_array(key, type_str, values, |value| {
                let value = *value as f32;
                value.is_finite().then_some(Primitive::F32(value))
            })
        }
        "i32" => {
            let values: Vec<i64> = ron::from_str(content).map_err(ron_err)?;
            checked_array(key, type_str, values, |value| {
                i32::try_from(*value).ok().map(Primitive::I32)
            })
        }
        "u32" => {
            let values: Vec<u64> = ron::from_str(content).map_err(ron_err)?;
            checked_array(key, type_str, values, |value| {
                u32::try_from(*value).ok().map(Primitive::U32)
            })
        }
        _ => Err(DebugAdapterError::Parse(format!(
            "Unknown type '{type_str}' for binding '{key}'"
        ))),
    }
}
