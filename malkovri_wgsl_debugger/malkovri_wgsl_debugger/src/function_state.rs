use std::collections::HashMap;

use naga::{Expression, Function, Handle, LocalVariable};

use crate::{place::ArgumentValue, place::ExpressionCache, value::Value};

/// Control-flow signal set on a [`FunctionFrame`] by `break`, `continue`, or `return`.
/// [`Evaluator::step`] reads these signals and performs the appropriate stack
/// manipulation before continuing execution.
#[derive(Clone, Debug, Default)]
pub(crate) enum ControlFlow {
    #[default]
    None,
    Break,
    Continue,
    Return(Option<Value>),
}

/// Identifies a function in the module without owning/cloning it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FunctionRef {
    /// An entry-point function, looked up via `module.entry_points[index].function`.
    EntryPoint(usize),
    /// A regular function, looked up via `module.functions[handle]`.
    Called(Handle<Function>),
}

/// The kind of block a [`BlockFrame`] represents, carrying the information needed to
/// implement its control-flow semantics.
#[derive(Clone, Debug)]
pub(crate) enum BlockKind {
    /// Plain block: the body of an `if`/`else`, or a bare `Statement::Block`.
    Plain,
    /// A loop with body and continuing phases.
    /// `other_block` holds whichever block is *not* currently executing:
    /// while running the body it holds `continuing`, and vice versa.
    Loop {
        other_block: naga::Block,
        break_if: Option<Handle<Expression>>,
        /// `true` once we have finished the body and are executing the continuing block.
        in_continuing: bool,
    },
    /// A switch-case body.
    Switch,
}

/// A function call frame on the unified execution stack.
#[derive(Clone, Debug)]
pub(crate) struct FunctionFrame {
    /// Reference to the function in the module.
    pub(crate) function_ref: FunctionRef,
    pub(crate) local_variables: HashMap<Handle<LocalVariable>, Value>,
    pub(crate) evaluated_expressions: ExpressionCache,
    pub(crate) evaluated_function_arguments: Vec<ArgumentValue>,
    /// The top-level statements of the function body.
    pub(crate) statements: naga::Block,
    pub(crate) current_statement_index: usize,
    /// The `Expression::CallResult` handle in the *parent* frame that should receive
    /// this function's return value.  `None` for the entry-point frame and for
    /// calls whose result is discarded.
    pub(crate) call_result_handle: Option<Handle<Expression>>,
    /// Control-flow signal written by `break`/`continue`/`return` handlers and consumed
    /// by [`Evaluator::next_statement`].
    pub(crate) control_flow: ControlFlow,
}

/// A block frame (if-body, loop-body, switch-case, plain block) on the unified stack.
#[derive(Clone, Debug)]
pub(crate) struct BlockFrame {
    /// The currently active statements (either the loop body or the continuing block).
    pub(crate) statements: naga::Block,
    pub(crate) current_statement_index: usize,
    pub(crate) kind: BlockKind,
}

impl BlockFrame {
    /// Switch this loop frame to its continuing block. No-op if not a Loop.
    pub(crate) fn switch_to_continuing(&mut self) {
        if let BlockKind::Loop {
            ref mut other_block,
            ref mut in_continuing,
            ..
        } = self.kind
        {
            std::mem::swap(&mut self.statements, other_block);
            self.current_statement_index = 0;
            *in_continuing = true;
        }
    }

    /// Restart the loop body from the beginning. No-op if not a Loop.
    pub(crate) fn restart_body(&mut self) {
        if let BlockKind::Loop {
            ref mut other_block,
            ref mut in_continuing,
            ..
        } = self.kind
        {
            std::mem::swap(&mut self.statements, other_block);
            self.current_statement_index = 0;
            *in_continuing = false;
        }
    }
}

/// A single entry on the evaluator's unified execution stack.
/// Function calls push a [`StackFrame::Function`]; entering any nested block
/// (`if`, `loop`, `switch`, bare block) pushes a [`StackFrame::Block`].
#[derive(Clone, Debug)]
pub(crate) enum StackFrame {
    Function(Box<FunctionFrame>),
    Block(BlockFrame),
}

impl StackFrame {
    /// The currently active statements for this frame.
    pub(crate) fn statements(&self) -> &naga::Block {
        match self {
            StackFrame::Function(f) => &f.statements,
            StackFrame::Block(b) => &b.statements,
        }
    }

    /// The current statement index for this frame.
    pub(crate) fn current_statement_index(&self) -> usize {
        match self {
            StackFrame::Function(f) => f.current_statement_index,
            StackFrame::Block(b) => b.current_statement_index,
        }
    }

    /// Increment the current statement index.
    pub(crate) fn increment_statement_index(&mut self) {
        match self {
            StackFrame::Function(f) => f.current_statement_index += 1,
            StackFrame::Block(b) => b.current_statement_index += 1,
        }
    }

    /// Whether this frame has executed all its statements.
    pub(crate) fn is_exhausted(&self) -> bool {
        self.current_statement_index() >= self.statements().len()
    }
}

/// The statement that will be executed next.
#[derive(Clone, Debug)]
pub(crate) struct NextStatement {
    pub(crate) statement: naga::Statement,
}
