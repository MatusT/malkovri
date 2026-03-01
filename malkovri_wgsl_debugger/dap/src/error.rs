use thiserror::Error;

use malkovri_wgsl_debugger::WgslToModuleError;

#[derive(Debug, Error)]
pub enum DebugAdapterError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Invalid program: {0}")]
    InvalidProgram(String),
    #[error("WGSL to Module error: {0}")]
    WgslToModuleError(Box<WgslToModuleError>),
}

impl From<WgslToModuleError> for DebugAdapterError {
    fn from(e: WgslToModuleError) -> Self {
        DebugAdapterError::WgslToModuleError(Box::new(e))
    }
}
