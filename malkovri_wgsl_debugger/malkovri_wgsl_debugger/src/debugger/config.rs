use super::WorkgroupConfig;
use crate::{ShaderProgram, Value};
use std::collections::BTreeMap;

/// Invocation configuration for the selected shader stage.
#[derive(Clone, Debug)]
pub enum ExecutionConfig {
    Compute(WorkgroupConfig),
    Vertex(VertexConfig),
    /// Execute one fragment with default fragment inputs.
    Fragment,
    /// Execute internally generated fragment quads.
    FragmentQuads(Vec<crate::graphics::FragmentQuad>),
}

impl From<WorkgroupConfig> for ExecutionConfig {
    fn from(config: WorkgroupConfig) -> Self {
        Self::Compute(config)
    }
}

impl From<DrawConfig> for ExecutionConfig {
    fn from(config: DrawConfig) -> Self {
        Self::Vertex(VertexConfig {
            draw: config,
            attributes: BTreeMap::new(),
        })
    }
}

/// A non-indexed draw. Each vertex/instance pair becomes one debugger thread.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct DrawConfig {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

impl Default for DrawConfig {
    fn default() -> Self {
        Self {
            vertex_count: 1,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        }
    }
}

impl DrawConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.vertex_count == 0 || self.instance_count == 0 {
            return Err(
                "drawConfig must contain at least one vertex and one instance to debug".into(),
            );
        }
        if self
            .first_vertex
            .checked_add(self.vertex_count - 1)
            .is_none()
        {
            return Err("drawConfig vertex indices exceed u32".into());
        }
        if self
            .first_instance
            .checked_add(self.instance_count - 1)
            .is_none()
        {
            return Err("drawConfig instance indices exceed u32".into());
        }
        if self.vertex_count.checked_mul(self.instance_count).is_none() {
            return Err("drawConfig invocation count exceeds u32".into());
        }
        Ok(())
    }
}

/// Decoded attribute data relative to the start of a draw.
#[derive(Clone, Debug)]
pub struct VertexAttribute {
    pub step_mode: VertexStepMode,
    pub values: Vec<Value>,
}

#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VertexStepMode {
    #[default]
    Vertex,
    Instance,
}

#[derive(Clone, Debug, Default)]
pub struct VertexConfig {
    pub draw: DrawConfig,
    pub attributes: BTreeMap<u32, VertexAttribute>,
}

impl From<VertexConfig> for ExecutionConfig {
    fn from(config: VertexConfig) -> Self {
        Self::Vertex(config)
    }
}

impl VertexConfig {
    pub(crate) fn validate(&self, program: &ShaderProgram, entry: usize) -> Result<(), String> {
        self.draw.validate()?;
        let fields = program.input_locations(entry)?;
        for location in self.attributes.keys() {
            if !fields.iter().any(|f| f.location == *location) {
                return Err(format!("unknown vertex @location({location})"));
            }
        }
        for field in fields {
            let attribute = self
                .attributes
                .get(&field.location)
                .ok_or_else(|| format!("missing vertex @location({})", field.location))?;
            let count = match attribute.step_mode {
                VertexStepMode::Vertex => self.draw.vertex_count,
                VertexStepMode::Instance => self.draw.instance_count,
            };
            if attribute.values.len() != count as usize {
                return Err(format!(
                    "@location({}) requires {count} values",
                    field.location
                ));
            }
            for (index, value) in attribute.values.iter().enumerate() {
                field
                    .ty
                    .validate(value)
                    .map_err(|e| format!("@location({}) values[{index}]: {e}", field.location))?;
            }
        }
        Ok(())
    }

    pub(crate) fn locations(&self, vertex: u32, instance: u32) -> BTreeMap<u32, Value> {
        self.attributes
            .iter()
            .map(|(location, attribute)| {
                let index = match attribute.step_mode {
                    VertexStepMode::Vertex => vertex,
                    VertexStepMode::Instance => instance,
                };
                (*location, attribute.values[index as usize].clone())
            })
            .collect()
    }
}
