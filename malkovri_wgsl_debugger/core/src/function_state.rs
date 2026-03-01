use std::collections::HashMap;

use naga::{Expression, Function, Handle, LocalVariable};

use crate::value::Value;

#[derive(Clone, Debug)]
pub struct FunctionState {
    pub function: Function,
    pub function_handle: Option<Handle<Function>>,
    pub local_variables: HashMap<Handle<LocalVariable>, Value>,
    pub evaluated_expressions: HashMap<Handle<Expression>, Value>,
    pub evaluated_function_arguments: Vec<Value>,
    pub current_statement_index: usize,
    pub returns: Option<Value>,
}

/// The statement that will be executed next, along with its context.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct NextStatement {
    pub function: Function,
    pub statement: naga::Statement,
    pub statement_index: usize,
}
