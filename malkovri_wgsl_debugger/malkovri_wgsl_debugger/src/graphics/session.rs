use super::{RasterConfig, VertexOutput};
use crate::{
    Debugger, DebuggerError, ExecutionConfig, GlobalConstants, ResourceBinding, ShaderProgram,
    Value, VertexConfig,
};
use std::{collections::HashMap, sync::Arc};

pub enum GraphicsSource {
    Vertex { entry: usize, config: VertexConfig },
    Outputs(Vec<VertexOutput>),
}

/// Successive stage debuggers sharing resource storage, with retained vertex snapshots.
pub struct GraphicsSession {
    active: Debugger,
    pending: Option<(usize, RasterConfig)>,
    program: Option<Arc<ShaderProgram>>,
    vertex_outputs: Vec<VertexOutput>,
}

impl GraphicsSession {
    pub fn single(debugger: Debugger) -> Self {
        Self {
            active: debugger,
            pending: None,
            program: None,
            vertex_outputs: Vec::new(),
        }
    }

    pub fn new(
        program: Arc<ShaderProgram>,
        fragment: usize,
        source: GraphicsSource,
        raster: RasterConfig,
        constants: GlobalConstants,
        bindings: HashMap<ResourceBinding, Value>,
    ) -> Result<Self, DebuggerError> {
        let invalid = DebuggerError::InvalidConfig;
        raster.validate().map_err(invalid)?;
        if program
            .entry_points()
            .nth(fragment)
            .is_none_or(|e| e.stage != naga::ShaderStage::Fragment)
        {
            return Err(invalid("expected fragment entry".into()));
        }
        match source {
            GraphicsSource::Vertex { entry, config } => {
                if program
                    .entry_points()
                    .nth(entry)
                    .is_none_or(|e| e.stage != naga::ShaderStage::Vertex)
                {
                    return Err(invalid("expected vertex entry".into()));
                }
                if !config.draw.vertex_count.is_multiple_of(3) {
                    return Err(invalid(
                        "triangle-list draw requires vertexCount divisible by three".into(),
                    ));
                }
                let function = &program.module().entry_points[entry].function;
                let result = function
                    .result
                    .as_ref()
                    .ok_or_else(|| invalid("vertex entry requires outputs".into()))?;
                let mut outputs = Vec::new();
                crate::program::interface::location_fields(
                    program.module(),
                    result.ty,
                    result.binding.as_ref(),
                    &mut outputs,
                )
                .map_err(invalid)?;
                for input in program.input_locations(fragment).map_err(invalid)? {
                    if !outputs.iter().any(|o| o == &input) {
                        return Err(invalid(format!(
                            "fragment @location({}) type/interpolation does not match vertex output",
                            input.location
                        )));
                    }
                }
                let active = program.create_debugger(entry, config, constants, bindings)?;
                Ok(Self {
                    active,
                    pending: Some((fragment, raster)),
                    program: Some(program),
                    vertex_outputs: Vec::new(),
                })
            }
            GraphicsSource::Outputs(vertices) => {
                let fields = program.input_locations(fragment).map_err(invalid)?;
                if vertices.iter().any(|v| {
                    v.locations
                        .keys()
                        .any(|key| !fields.iter().any(|f| f.location == *key))
                }) {
                    return Err(invalid("unknown supplied vertex output location".into()));
                }
                let quads = program
                    .interpolate_fragments(fragment, &vertices, &raster)
                    .map_err(invalid)?;
                let active = program.create_debugger(
                    fragment,
                    ExecutionConfig::FragmentQuads(quads),
                    constants,
                    bindings,
                )?;
                Ok(Self {
                    active,
                    pending: None,
                    program: Some(program),
                    vertex_outputs: vertices,
                })
            }
        }
    }

    pub fn debugger(&self) -> &Debugger {
        &self.active
    }
    pub fn debugger_mut(&mut self) -> &mut Debugger {
        &mut self.active
    }
    pub fn has_next_stage(&self) -> bool {
        self.pending.is_some()
    }
    pub fn vertex_outputs(&self) -> &[VertexOutput] {
        &self.vertex_outputs
    }

    /// Transition only after vertex completion. Failure preserves the current debugger.
    pub fn advance_stage(&mut self) -> Result<(), DebuggerError> {
        let (fragment, raster) = self
            .pending
            .as_ref()
            .ok_or_else(|| DebuggerError::InvalidConfig("no next stage".into()))?;
        let vertices = self
            .active
            .vertex_outputs()
            .map_err(DebuggerError::InvalidConfig)?;
        let quads = self
            .program
            .as_ref()
            .unwrap()
            .interpolate_fragments(*fragment, &vertices, raster)
            .map_err(DebuggerError::InvalidConfig)?;
        let next = self
            .active
            .create_next_stage(*fragment, ExecutionConfig::FragmentQuads(quads))?;
        self.active = next;
        self.vertex_outputs = vertices;
        self.pending = None;
        Ok(())
    }
}
