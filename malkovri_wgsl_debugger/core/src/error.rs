use naga::{GlobalVariable, Handle};
use thiserror::Error;

#[allow(dead_code)]
#[derive(Debug, Clone, Error)]
pub enum EvaluatorError {
    #[error("Unsupported local variable type: {0:?}")]
    UnsupportedLocalVariableType(Handle<naga::Type>),
    #[error("Unsupported literal type: {0}")]
    UnsupportedLiteralType(String),
    #[error("Load expression did not evaluate to a pointer")]
    LoadNotPointer,
    #[error("Store to non-pointer value")]
    StoreToNonPointer,
    #[error("AccessIndex on unsupported type: {0}")]
    AccessIndexUnsupportedType(String),
    #[error("Unsupported built-in variable: {0:?}")]
    UnsupportedBuiltIn(naga::ir::BuiltIn),
    #[error("Unsupported binary operation: {0} with values: {1}, {2}")]
    UnsupportedBinaryOperation(String, String, String),
    #[error("Global variable {0:?} not found")]
    GlobalVariableNotFound(Handle<GlobalVariable>),
    #[error("Index is not a U32: {0}")]
    IndexNotU32(String),
    #[error("Access on non-array type")]
    AccessOnNonArray,
    #[error("Unknown expression: {0}")]
    UnknownExpression(String),
    #[error("Unsupported vector type: {0}")]
    UnsupportedVectorType(String),
}
