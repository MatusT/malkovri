use super::WorkgroupConfig;

/// Invocation configuration for the selected shader stage.
#[derive(Clone, Debug)]
pub enum ExecutionConfig {
    Compute(WorkgroupConfig),
    Vertex(DrawConfig),
    /// Execute one fragment with default fragment inputs.
    Fragment,
}

impl From<WorkgroupConfig> for ExecutionConfig {
    fn from(config: WorkgroupConfig) -> Self {
        Self::Compute(config)
    }
}

impl From<DrawConfig> for ExecutionConfig {
    fn from(config: DrawConfig) -> Self {
        Self::Vertex(config)
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
