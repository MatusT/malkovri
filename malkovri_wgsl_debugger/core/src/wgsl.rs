use naga::{
    Module,
    front::wgsl,
    valid::{self, ValidationError},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WgslToModuleError {
    #[error("Parse error: {0}")]
    ParseError(#[from] naga::front::wgsl::ParseError),
    #[error("Validation error: {0}")]
    ValidationError(Box<naga::WithSpan<ValidationError>>),
}

impl From<naga::WithSpan<ValidationError>> for WgslToModuleError {
    fn from(e: naga::WithSpan<ValidationError>) -> Self {
        WgslToModuleError::ValidationError(Box::new(e))
    }
}

pub fn wgsl_to_module(source: &str) -> Result<Module, WgslToModuleError> {
    let mut frontend = wgsl::Frontend::new();
    let module = frontend.parse(source)?;

    let mut validator =
        valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all());
    let _module_info = validator.validate(&module)?;

    Ok(module)
}
