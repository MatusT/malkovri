use std::collections::HashMap;

use naga::{Expression, Function, Handle, LocalVariable};

use crate::value::Value;

/// Control-flow signal set on a [`FunctionFrame`] by `break`, `continue`, or `return`.
/// [`Evaluator::step`] reads these signals and performs the appropriate stack
/// manipulation before continuing execution.
#[derive(Clone, Debug, Default)]
pub enum ControlFlow {
    #[default]
    None,
    Break,
    Continue,
    Return(Option<Value>),
}

/// The kind of block a [`BlockFrame`] represents, carrying the information needed to
/// implement its control-flow semantics.
#[derive(Clone, Debug)]
pub enum BlockKind {
    /// Plain block: the body of an `if`/`else`, or a bare `Statement::Block`.
    Plain,
    /// A loop body with its associated `continuing` block.
    /// `body` is stored separately so the loop can restart after continuing finishes.
    Loop {
        body: naga::Block,
        continuing: naga::Block,
        break_if: Option<Handle<Expression>>,
        /// `true` once we have finished the body and are executing the continuing block.
        in_continuing: bool,
    },
    /// A switch-case body.
    Switch,
}

/// A function call frame on the unified execution stack.
#[derive(Clone, Debug)]
pub struct FunctionFrame {
    pub function: Function,
    pub function_handle: Option<Handle<Function>>,
    pub local_variables: HashMap<Handle<LocalVariable>, Value>,
    pub evaluated_expressions: HashMap<Handle<Expression>, Value>,
    pub evaluated_function_arguments: Vec<Value>,
    /// The top-level statements of the function body.
    pub statements: naga::Block,
    pub current_statement_index: usize,
    /// The `Expression::CallResult` handle in the *parent* frame that should receive
    /// this function's return value.  `None` for the entry-point frame and for
    /// calls whose result is discarded.
    pub call_result_handle: Option<Handle<Expression>>,
    /// Control-flow signal written by `break`/`continue`/`return` handlers and consumed
    /// by [`Evaluator::next_statement`].
    pub control_flow: ControlFlow,
}

/// A block frame (if-body, loop-body, switch-case, plain block) on the unified stack.
#[derive(Clone, Debug)]
pub struct BlockFrame {
    /// The currently active statements (either the loop body or the continuing block).
    pub statements: naga::Block,
    pub current_statement_index: usize,
    pub kind: BlockKind,
}

impl BlockFrame {
    /// Switch this loop frame to its continuing block. No-op if not a Loop.
    pub fn switch_to_continuing(&mut self) {
        if let BlockKind::Loop {
            ref continuing,
            ref mut in_continuing,
            ..
        } = self.kind
        {
            self.statements = continuing.clone();
            self.current_statement_index = 0;
            *in_continuing = true;
        }
    }

    /// Restart the loop body from the beginning. No-op if not a Loop.
    pub fn restart_body(&mut self) {
        if let BlockKind::Loop {
            ref body,
            ref mut in_continuing,
            ..
        } = self.kind
        {
            self.statements = body.clone();
            self.current_statement_index = 0;
            *in_continuing = false;
        }
    }
}

/// A single entry on the evaluator's unified execution stack.
/// Function calls push a [`StackFrame::Function`]; entering any nested block
/// (`if`, `loop`, `switch`, bare block) pushes a [`StackFrame::Block`].
#[derive(Clone, Debug)]
pub enum StackFrame {
    Function(Box<FunctionFrame>),
    Block(BlockFrame),
}

impl StackFrame {
    /// The currently active statements for this frame.
    pub fn statements(&self) -> &naga::Block {
        match self {
            StackFrame::Function(f) => &f.statements,
            StackFrame::Block(b) => &b.statements,
        }
    }

    /// The current statement index for this frame.
    pub fn current_statement_index(&self) -> usize {
        match self {
            StackFrame::Function(f) => f.current_statement_index,
            StackFrame::Block(b) => b.current_statement_index,
        }
    }

    /// Increment the current statement index.
    pub fn increment_statement_index(&mut self) {
        match self {
            StackFrame::Function(f) => f.current_statement_index += 1,
            StackFrame::Block(b) => b.current_statement_index += 1,
        }
    }

    /// Whether this frame has executed all its statements.
    pub fn is_exhausted(&self) -> bool {
        self.current_statement_index() >= self.statements().len()
    }
}

/// The statement that will be executed next, along with its context.
// TODO: Consider storing only the function name or Handle<Function> instead of
// cloning the entire Function to reduce allocation on every step.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct NextStatement {
    pub function: Function,
    pub statement: naga::Statement,
    pub statement_index: usize,
}
