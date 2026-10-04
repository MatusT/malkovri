use std::collections::HashMap;

use naga::{Expression, Handle, LocalVariable};

use crate::{
    invocation::place::ExpressionCache,
    invocation::place::{ArgumentValue, EvaluatedExpression},
    program::{BlockId, FunctionId, ShaderProgram, StatementId},
    value::Value,
};

/// Control-flow signal set on a [`FunctionFrame`] by `break`, `continue`, or `return`.
/// [`InvocationState::step`] reads these signals and performs the appropriate stack
/// manipulation before continuing execution.
#[derive(Clone, Debug, Default)]
pub(crate) enum ControlFlow {
    #[default]
    None,
    Break,
    Continue,
    Return(Option<Value>),
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
        other_block: BlockId,
        break_if: Option<Handle<Expression>>,
        /// `true` once we have finished the body and are executing the continuing block.
        in_continuing: bool,
    },
    /// A switch-case body.
    Switch {
        statement: StatementId,
        next_case: Option<usize>,
    },
}

/// A function call frame on the unified execution stack.
#[derive(Clone, Debug)]
pub(crate) struct FunctionFrame {
    /// Reference to the function in the module.
    function_id: FunctionId,
    local_variables: HashMap<Handle<LocalVariable>, Value>,
    evaluated_expressions: ExpressionCache,
    evaluated_function_arguments: Vec<ArgumentValue>,
    /// The top-level statements of the function body.
    block: BlockId,
    current_statement_index: usize,
    /// The `Expression::CallResult` handle in the *parent* frame that should receive
    /// this function's return value.  `None` for the entry-point frame and for
    /// calls whose result is discarded.
    call_result_handle: Option<Handle<Expression>>,
    /// Control-flow signal written by `break`/`continue`/`return` handlers and consumed
    /// by [`InvocationState::next_statement`].
    control_flow: ControlFlow,
}

/// A block frame (if-body, loop-body, switch-case, plain block) on the unified stack.
#[derive(Clone, Debug)]
pub(crate) struct BlockFrame {
    /// The currently active statements (either the loop body or the continuing block).
    block: BlockId,
    current_statement_index: usize,
    kind: BlockKind,
    iteration: u64,
}

impl BlockFrame {
    pub(crate) fn new(block: BlockId, kind: BlockKind) -> Self {
        Self {
            block,
            kind,
            iteration: 0,
            current_statement_index: 0,
        }
    }

    pub(crate) fn iteration(&self) -> u64 {
        self.iteration
    }

    pub(crate) fn kind(&self) -> &BlockKind {
        &self.kind
    }

    /// Switch this loop frame to its continuing block. No-op if not a Loop.
    pub(crate) fn switch_to_continuing(&mut self) {
        if let BlockKind::Loop {
            ref mut other_block,
            ref mut in_continuing,
            ..
        } = self.kind
        {
            std::mem::swap(&mut self.block, other_block);
            self.current_statement_index = 0;
            *in_continuing = true;
        }
    }

    /// Restart the loop body from the beginning. No-op if not a Loop.
    pub(crate) fn restart_body(&mut self) {
        self.iteration += 1;
        if let BlockKind::Loop {
            ref mut other_block,
            ref mut in_continuing,
            ..
        } = self.kind
        {
            std::mem::swap(&mut self.block, other_block);
            self.current_statement_index = 0;
            *in_continuing = false;
        }
    }
}

/// A single entry on the invocation's unified execution stack.
/// Function calls push a [`StackFrame::Function`]; entering any nested block
/// (`if`, `loop`, `switch`, bare block) pushes a [`StackFrame::Block`].
#[derive(Clone, Debug)]
pub(crate) enum StackFrame {
    Function(Box<FunctionFrame>),
    Block(BlockFrame),
}

/// Inspection position for a function, including a caller suspended at a call.
#[derive(Clone, Copy)]
pub(crate) struct FrameContext {
    function_index: usize,
    block_index: usize,
    statement_index: usize,
}

impl StackFrame {
    pub(crate) fn block(&self) -> BlockId {
        match self {
            StackFrame::Function(frame) => frame.block,
            StackFrame::Block(frame) => frame.block,
        }
    }

    pub(crate) fn position(&self) -> StatementId {
        StatementId::new(self.block(), self.current_statement_index())
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
    pub(crate) fn is_exhausted(&self, program: &ShaderProgram) -> bool {
        self.current_statement_index() >= program.block(self.block()).len()
    }
}

impl FunctionFrame {
    pub(crate) fn new(
        function_id: FunctionId,
        block: BlockId,
        arguments: Vec<ArgumentValue>,
        call_result_handle: Option<Handle<Expression>>,
    ) -> Self {
        Self {
            function_id,
            block,
            local_variables: HashMap::new(),
            evaluated_expressions: HashMap::new(),
            evaluated_function_arguments: arguments,
            current_statement_index: 0,
            call_result_handle,
            control_flow: ControlFlow::None,
        }
    }

    pub(crate) fn function_id(&self) -> FunctionId {
        self.function_id
    }
    pub(crate) fn call_result_handle(&self) -> Option<Handle<Expression>> {
        self.call_result_handle
    }
    pub(crate) fn argument(&self, index: usize) -> Option<&ArgumentValue> {
        self.evaluated_function_arguments.get(index)
    }
    pub(crate) fn local(&self, handle: Handle<LocalVariable>) -> Option<&Value> {
        self.local_variables.get(&handle)
    }
    pub(crate) fn local_mut(&mut self, handle: Handle<LocalVariable>) -> Option<&mut Value> {
        self.local_variables.get_mut(&handle)
    }
    pub(crate) fn set_local(&mut self, handle: Handle<LocalVariable>, value: Value) {
        self.local_variables.insert(handle, value);
    }
    pub(crate) fn expression(&self, handle: Handle<Expression>) -> Option<&EvaluatedExpression> {
        self.evaluated_expressions.get(&handle)
    }
    pub(crate) fn set_expression(
        &mut self,
        handle: Handle<Expression>,
        value: EvaluatedExpression,
    ) {
        self.evaluated_expressions.insert(handle, value);
    }
    pub(crate) fn forget_expression(&mut self, handle: Handle<Expression>) {
        self.evaluated_expressions.remove(&handle);
    }
    pub(crate) fn set_control_flow(&mut self, signal: ControlFlow) {
        self.control_flow = signal;
    }
    pub(crate) fn take_control_flow(&mut self) -> ControlFlow {
        std::mem::take(&mut self.control_flow)
    }
}

impl FrameContext {
    pub(crate) fn new(function_index: usize, block_index: usize, statement_index: usize) -> Self {
        Self {
            function_index,
            block_index,
            statement_index,
        }
    }
    pub(crate) fn function_index(self) -> usize {
        self.function_index
    }
    pub(crate) fn block_index(self) -> usize {
        self.block_index
    }
    pub(crate) fn statement_index(self) -> usize {
        self.statement_index
    }
}
