use thiserror::Error;

use malkovri_wgsl_debugger::{DebuggerError, EvaluatorError};

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
    #[error("Debugger error: {0}")]
    Debugger(#[from] DebuggerError),
    #[error("Evaluator error: {0}")]
    Evaluator(#[from] EvaluatorError),
}
