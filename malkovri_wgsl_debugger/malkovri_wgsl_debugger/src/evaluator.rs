use crate::{
    function_state::{
        BlockFrame, BlockKind, ControlFlow, FunctionFrame, FunctionRef, NextStatement, StackFrame,
    },
    primitive::Primitive,
    value::Value,
};

use std::{cell::RefCell, collections::HashMap, collections::HashSet, rc::Rc, sync::Arc};

use naga::{Expression, GlobalVariable, Handle, LocalVariable, Module, Span, Statement};

#[derive(Copy, Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct EntryPointInputs {
    // vertex
    pub base_instance: u32,
    pub base_vertex: i32,
    pub clip_distance: [f32; 8],
    pub cull_distance: [f32; 8],
    pub instance_index: u32,
    pub point_size: f32,
    pub vertex_index: u32,
    pub draw_id: u32,

    // fragment
    pub position: [f32; 4],
    pub view_index: i32,
    pub frag_depth: f32,
    pub point_coord: [f32; 2],
    pub front_facing: bool,
    pub primitive_index: u32,
    pub sample_index: u32,
    pub sample_mask: u32,

    // compute
    pub global_invocation_id: [u32; 3],
    pub local_invocation_id: [u32; 3],
    pub local_invocation_index: u32,
    pub workgroup_id: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub num_workgroups: [u32; 3],

    // subgroup
    pub num_subgroups: u32,
    pub subgroup_id: u32,
    pub subgroup_size: u32,
    pub subgroup_invocation_id: u32,
}

pub struct Evaluator {
    pub(crate) module: Arc<Module>,
    pub(crate) entry_point_inputs: EntryPointInputs,
    pub(crate) global_values: HashMap<naga::Handle<GlobalVariable>, Value>,
    pub(crate) stack: Vec<StackFrame>,
    /// Pre-computed: maps each local variable to the span of the block that declares it.
    function_declaring_scopes:
        HashMap<FunctionRef, HashMap<Handle<LocalVariable>, std::ops::Range<usize>>>,
    /// Pre-computed: maps each named expression to the span of the block that declares it
    /// (determined by the `Emit` statement that covers the expression handle).
    named_expr_declaring_scopes:
        HashMap<FunctionRef, HashMap<Handle<Expression>, std::ops::Range<usize>>>,
}

impl Evaluator {
    pub fn new(
        module: Arc<Module>,
        entry_point_index: usize,
        entry_point_inputs: EntryPointInputs,
        global_values: HashMap<naga::ResourceBinding, Value>,
    ) -> Self {
        let statements = module.entry_points[entry_point_index].function.body.clone();

        let mut function_declaring_scopes = HashMap::new();
        let mut named_expr_declaring_scopes = HashMap::new();
        for (handle, function) in module.functions.iter() {
            let fref = FunctionRef::Called(handle);
            let block_scopes = Self::collect_all_block_scopes(function);
            function_declaring_scopes.insert(
                fref.clone(),
                Self::local_declaring_scopes(function, &block_scopes),
            );
            named_expr_declaring_scopes.insert(
                fref,
                Self::named_expression_declaring_scopes(function, &block_scopes),
            );
        }
        for (i, entry_point) in module.entry_points.iter().enumerate() {
            let fref = FunctionRef::EntryPoint(i);
            let block_scopes = Self::collect_all_block_scopes(&entry_point.function);
            function_declaring_scopes.insert(
                fref.clone(),
                Self::local_declaring_scopes(&entry_point.function, &block_scopes),
            );
            named_expr_declaring_scopes.insert(
                fref,
                Self::named_expression_declaring_scopes(&entry_point.function, &block_scopes),
            );
        }

        let mut evaluator = Evaluator {
            global_values: global_values
                .iter()
                .map(|(k, v)| {
                    (
                        module
                            .global_variables
                            .fetch_if(|h| h.binding.unwrap() == *k)
                            .unwrap(),
                        v.clone(),
                    )
                })
                .collect(),
            module,
            entry_point_inputs,
            stack: vec![StackFrame::Function(Box::new(FunctionFrame {
                function_ref: FunctionRef::EntryPoint(entry_point_index),
                local_variables: HashMap::new(),
                evaluated_expressions: HashMap::new(),
                evaluated_function_arguments: Vec::new(),
                statements,
                current_statement_index: 0,
                call_result_handle: None,
                control_flow: ControlFlow::None,
            }))],
            function_declaring_scopes,
            named_expr_declaring_scopes,
        };
        evaluator.initialize_local_variables();
        evaluator
    }

    /// Collect all block scopes (the function body + nested blocks) for scope analysis.
    fn collect_all_block_scopes(function: &naga::Function) -> Vec<Span> {
        let mut block_scopes = vec![Span::total_span(function.body.span_iter().map(|(_, s)| *s))];
        block_scopes.extend(Self::collect_block_spans(function.body.span_iter()));
        block_scopes
    }

    /// Find the smallest block scope that fully contains `item_range`.
    fn smallest_enclosing_scope(
        block_scopes: &[Span],
        item_range: &std::ops::Range<usize>,
    ) -> std::ops::Range<usize> {
        block_scopes
            .iter()
            .filter_map(|scope| {
                scope
                    .to_range()
                    .filter(|s| s.start <= item_range.start && item_range.end <= s.end)
            })
            .min_by_key(|s| s.end - s.start)
            .unwrap_or(0..usize::MAX)
    }

    fn local_declaring_scopes(
        function: &naga::Function,
        block_scopes: &[Span],
    ) -> HashMap<Handle<LocalVariable>, std::ops::Range<usize>> {
        function
            .local_variables
            .iter()
            .filter_map(|(handle, _)| {
                let variable_range = function.local_variables.get_span(handle).to_range()?;
                Some((
                    handle,
                    Self::smallest_enclosing_scope(block_scopes, &variable_range),
                ))
            })
            .collect()
    }

    fn named_expression_declaring_scopes(
        function: &naga::Function,
        block_scopes: &[Span],
    ) -> HashMap<Handle<Expression>, std::ops::Range<usize>> {
        // Build a map from expression handle → span of the Emit statement that covers it.
        let mut emit_spans: HashMap<Handle<Expression>, Span> = HashMap::new();
        Self::collect_emit_spans(&function.body, &mut emit_spans);

        function
            .named_expressions
            .iter()
            .map(|(handle, _)| {
                let scope = emit_spans
                    .get(handle)
                    .and_then(|sp| sp.to_range())
                    .map(|r| Self::smallest_enclosing_scope(block_scopes, &r))
                    .unwrap_or(0..usize::MAX);
                (*handle, scope)
            })
            .collect()
    }

    /// Walk all blocks recursively and record the span of each `Emit` statement
    /// for every expression handle it covers.
    fn collect_emit_spans(block: &naga::Block, out: &mut HashMap<Handle<Expression>, Span>) {
        for (statement, span) in block.span_iter() {
            match statement {
                Statement::Emit(range) => {
                    for handle in range.clone() {
                        out.insert(handle, *span);
                    }
                }
                Statement::Block(inner) => Self::collect_emit_spans(inner, out),
                Statement::If { accept, reject, .. } => {
                    Self::collect_emit_spans(accept, out);
                    Self::collect_emit_spans(reject, out);
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    Self::collect_emit_spans(body, out);
                    Self::collect_emit_spans(continuing, out);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        Self::collect_emit_spans(&case.body, out);
                    }
                }
                _ => {}
            }
        }
    }

    /// Resolve a [`FunctionRef`] to the actual `naga::Function` in the module.
    pub fn resolve_function(&self, fref: &FunctionRef) -> &naga::Function {
        match fref {
            FunctionRef::EntryPoint(idx) => &self.module.entry_points[*idx].function,
            FunctionRef::Called(handle) => &self.module.functions[*handle],
        }
    }

    /// Resolve the `naga::Function` for the function frame at `function_index`.
    fn resolve_function_at(&self, function_index: usize) -> &naga::Function {
        match &self.stack[function_index] {
            StackFrame::Function(f) => self.resolve_function(&f.function_ref),
            _ => unreachable!(),
        }
    }

    /// Return a reference to the `naga::Function` for the current call frame.
    pub fn current_function(&self) -> Option<&naga::Function> {
        let frame = self.current_function_frame()?;
        Some(self.resolve_function(&frame.function_ref))
    }

    /// Index of the topmost `Function` frame, used to look up expressions and variables.
    pub(crate) fn current_function_frame_index(&self) -> Option<usize> {
        self.stack
            .iter()
            .rposition(|sf| matches!(sf, StackFrame::Function(_)))
    }

    /// Return a reference to the current function frame (the nearest `Function` variant on the
    /// stack), or `None` if the stack is empty.
    pub fn current_function_frame(&self) -> Option<&FunctionFrame> {
        let function_index = self.current_function_frame_index()?;
        match &self.stack[function_index] {
            StackFrame::Function(f) => Some(f),
            StackFrame::Block(_) => None,
        }
    }

    /// Return a mutable reference to the current function frame (the nearest `Function` variant on the
    /// stack), or `None` if the stack is empty.
    pub fn current_function_frame_mut(&mut self) -> Option<&mut FunctionFrame> {
        let function_index = self.current_function_frame_index()?;
        match &mut self.stack[function_index] {
            StackFrame::Function(f) => Some(f),
            StackFrame::Block(_) => None,
        }
    }

    /// Return the statements and current index of the top-of-stack frame.
    /// Unlike `current_function_frame`, this reflects the innermost active block,
    /// which may be a nested `if`/`loop`/`switch` block rather than the function body.
    pub fn current_active_block(&self) -> Option<(&naga::Block, usize)> {
        let top = self.stack.last()?;
        Some((top.statements(), top.current_statement_index()))
    }

    pub fn current_line(&self, source: &str) -> Option<u32> {
        let (block, idx) = self.current_active_block()?;
        let spans: Vec<_> = block.span_iter().map(|(_, span)| span).collect();
        Some(spans.get(idx)?.location(source).line_number)
    }

    /// Return all global variables with their names and current values.
    pub fn global_variable_values(&self) -> Vec<(Option<String>, Value)> {
        self.global_values
            .iter()
            .map(|(handle, value)| {
                let name = self.module.global_variables[*handle].name.clone();
                (name, value.clone())
            })
            .collect()
    }

    /// Evaluate the current function arguments in declaration order.
    pub fn current_function_argument_values(&self) -> Vec<(Option<String>, Value)> {
        let func_idx = match self.current_function_frame_index() {
            Some(i) => i,
            None => return vec![],
        };
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return vec![];
        };

        let function = self.resolve_function(&frame.function_ref);
        function
            .arguments
            .iter()
            .enumerate()
            .map(|(index, argument)| {
                (
                    argument.name.clone(),
                    self.evaluate_function_argument(index, func_idx),
                )
            })
            .collect()
    }

    /// Index of the `Function` frame below `function_index` — the caller's frame, used for `CallResult`.
    fn parent_function_frame_index(&self, function_index: usize) -> Option<usize> {
        self.stack[..function_index]
            .iter()
            .rposition(|sf| matches!(sf, StackFrame::Function(_)))
    }

    /// Returns local variable handles that are lexically in scope at the
    /// current execution point: declared before the current position and
    /// owned by a block that contains the current scope.
    pub fn local_variables_in_current_scope(&self) -> HashSet<Handle<LocalVariable>> {
        let func_idx = match self.current_function_frame_index() {
            Some(i) => i,
            None => return HashSet::new(),
        };

        let function = self.resolve_function_at(func_idx);
        let frame = self.current_function_frame().unwrap();
        let declaring_scopes = self
            .function_declaring_scopes
            .get(&frame.function_ref)
            .unwrap();

        let current_block = self.stack.last();

        // The span of the current (innermost) execution scope.
        let current_scope = current_block
            .and_then(|frame| frame.source_span().to_range())
            .unwrap_or(0..usize::MAX);

        // Current execution position for the "declared before" check.
        let current_pos = current_block.and_then(|frame| {
            let idx = frame.current_statement_index();
            frame
                .statements()
                .span_iter()
                .nth(idx)
                .and_then(|(_, sp)| sp.to_range())
                .map(|r| r.start)
        });

        function
            .local_variables
            .iter()
            .filter_map(|(handle, _)| {
                let var_range = function.local_variables.get_span(handle).to_range()?;
                let var_end = var_range.end;

                // Skip variables whose declaration we haven't stepped past.
                if current_pos.is_some_and(|pos| pos < var_end) {
                    return None;
                }

                let declaring_scope = declaring_scopes.get(&handle)?;

                // Variable is visible iff the current scope is nested inside the declaring scope.
                (declaring_scope.contains(&current_scope.start)
                    && current_scope.end <= declaring_scope.end)
                    .then_some(handle)
            })
            .collect()
    }

    /// Recursively collect spans of all inner blocks in the block tree.
    fn collect_block_spans<'a>(
        statements: impl Iterator<Item = (&'a Statement, &'a Span)>,
    ) -> Vec<Span> {
        let mut spans = Vec::new();
        for (statement, statement_span) in statements {
            match statement {
                Statement::Block(inner) => {
                    spans.push(Span::total_span(inner.span_iter().map(|(_, s)| *s)));
                    spans.extend(Self::collect_block_spans(inner.span_iter()));
                }
                Statement::If { accept, reject, .. } => {
                    spans.push(Span::total_span(accept.span_iter().map(|(_, s)| *s)));
                    spans.extend(Self::collect_block_spans(accept.span_iter()));
                    spans.push(Span::total_span(reject.span_iter().map(|(_, s)| *s)));
                    spans.extend(Self::collect_block_spans(reject.span_iter()));
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    spans.push(*statement_span);
                    spans.extend(Self::collect_block_spans(body.span_iter()));
                    spans.extend(Self::collect_block_spans(continuing.span_iter()));
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        spans.push(Span::total_span(case.body.span_iter().map(|(_, s)| *s)));
                        spans.extend(Self::collect_block_spans(case.body.span_iter()));
                    }
                }
                _ => {}
            }
        }
        spans
    }

    /// Evaluate all named expressions (WGSL `let` bindings) in the current function frame
    /// that are in scope at the current execution point.
    /// Returns `(name, value)` pairs in source order.
    pub fn named_expression_values(&self) -> Vec<(String, Value)> {
        let func_idx = match self.current_function_frame_index() {
            Some(i) => i,
            None => return vec![],
        };
        let StackFrame::Function(ref frame) = self.stack[func_idx] else {
            return vec![];
        };
        let function = self.resolve_function(&frame.function_ref);
        let declaring_scopes = match self.named_expr_declaring_scopes.get(&frame.function_ref) {
            Some(s) => s,
            None => return vec![],
        };

        let current_scope = self
            .stack
            .last()
            .and_then(|frame| frame.source_span().to_range())
            .unwrap_or(0..usize::MAX);

        let mut emitted = HashSet::new();
        let last_frame = self.stack.len() - 1;
        for frame_idx in func_idx..self.stack.len() {
            let frame = &self.stack[frame_idx];
            let limit = frame.current_statement_index();
            for (i, (stmt, _)) in frame.statements().span_iter().enumerate() {
                if i > limit {
                    break;
                }
                if let Statement::Emit(range) = stmt {
                    emitted.extend(range.clone());
                }
            }
        }

        function
            .named_expressions
            .iter()
            .filter(|(handle, _)| {
                if !emitted.contains(handle) {
                    return false;
                }
                let Some(declaring_scope) = declaring_scopes.get(handle) else {
                    return false;
                };
                declaring_scope.start <= current_scope.start
                    && current_scope.end <= declaring_scope.end
            })
            .map(|(handle, name)| (name.clone(), self.evaluate_expression(*handle)))
            .collect()
    }
}

// Core execution loop
impl Evaluator {
    /// Advance past any pending control-flow signals and exhausted frames until
    /// a live statement is ready to execute (or the stack is empty).
    fn advance_to_live_statement(&mut self) -> bool {
        loop {
            if self.stack.is_empty() {
                return false;
            }

            // Phase 1: Resolve pending control-flow signals.
            if self.resolve_control_flow_signal() {
                continue;
            }

            // Phase 2: Pop exhausted frames.
            if self.pop_if_exhausted() {
                continue;
            }

            return true;
        }
    }

    /// If the current function frame has a pending control-flow signal, apply it
    /// and return `true`. Otherwise return `false`.
    fn resolve_control_flow_signal(&mut self) -> bool {
        let function_index = match self.current_function_frame_index() {
            Some(i) => i,
            None => return false,
        };
        let StackFrame::Function(ref mut frame) = self.stack[function_index] else {
            unreachable!("current_function_frame_index always returns a Function frame");
        };
        let signal = std::mem::take(&mut frame.control_flow);
        match signal {
            ControlFlow::None => false,
            ControlFlow::Break => {
                self.apply_break(function_index);
                true
            }
            ControlFlow::Continue => {
                self.apply_continue(function_index);
                true
            }
            ControlFlow::Return(return_val) => {
                self.apply_return(function_index, return_val);
                true
            }
        }
    }

    /// If the top frame is exhausted, handle it and return `true`.
    fn pop_if_exhausted(&mut self) -> bool {
        let top = self.stack.len() - 1;
        if self.stack[top].is_exhausted() {
            self.handle_exhausted_frame(top);
            true
        } else {
            false
        }
    }

    /// Execute the current statement and advance the program counter.
    /// Returns the *upcoming* statement that will execute on the next call,
    /// or `None` if execution has finished.
    pub fn step(&mut self) -> Option<NextStatement> {
        if !self.advance_to_live_statement() {
            return None;
        }

        let caller_frame_index = self.stack.len() - 1;

        {
            let current_function_index = self.current_function_frame_index()?;
            let current_statement_index = self.stack.last()?.current_statement_index();
            let block = self.stack.last()?.statements();
            let (current_statement, statement_span) = block
                .span_iter()
                .nth(current_statement_index)
                .map(|(s, sp)| (s.clone(), *sp))?;

            self.handle_statement(current_statement, current_function_index, statement_span);
        }

        self.stack[caller_frame_index].increment_statement_index();

        // Resolve any signals/exhaustion produced by the statement we just ran,
        // then return whatever is live next (or None if execution finished).
        self.advance_to_live_statement();

        self.peek_next_statement()
    }

    /// Step through consecutive Emit statements so that variables are tracked
    /// before the debugger stops.
    pub fn skip_emits(&mut self) {
        while let Some(next) = self.peek_next_statement() {
            if matches!(next.statement, Statement::Emit(_)) {
                self.step();
            } else {
                break;
            }
        }
    }

    fn peek_next_statement(&self) -> Option<NextStatement> {
        let frame = self.current_function_frame()?;

        let current_block = self.stack.last()?;
        let current_statement_index = current_block.current_statement_index();
        let current_statement = current_block
            .statements()
            .get(current_statement_index)?
            .clone();

        Some(NextStatement {
            function_ref: frame.function_ref.clone(),
            statement: current_statement,
            statement_index: current_statement_index,
        })
    }
}

// Control-flow signal handlers
impl Evaluator {
    /// Pop block frames above `function_index` until (and including) the nearest `Loop` or `Switch` frame.
    fn apply_break(&mut self, function_index: usize) {
        while self.stack.len() > function_index + 1 {
            let is_target = matches!(
                self.stack.last(),
                Some(StackFrame::Block(BlockFrame {
                    kind: BlockKind::Loop { .. } | BlockKind::Switch,
                    ..
                }))
            );
            self.stack.pop();
            if is_target {
                break;
            }
        }
    }

    /// Pop block frames above `function_index` until the nearest `Loop` frame (keep it), then switch
    /// it to its `continuing` block.
    fn apply_continue(&mut self, function_index: usize) {
        while self.stack.len() > function_index + 1 {
            if matches!(
                self.stack.last(),
                Some(StackFrame::Block(BlockFrame {
                    kind: BlockKind::Loop { .. },
                    ..
                }))
            ) {
                break;
            }
            self.stack.pop();
        }

        // Switch the loop frame to its continuing block.
        let top = self.stack.len() - 1;
        if top > function_index {
            if let StackFrame::Block(ref mut block_frame) = self.stack[top] {
                block_frame.switch_to_continuing();
            }
        }
    }

    /// Store the return value in the parent function frame, then truncate the stack
    /// to remove the returning function and everything above it.
    fn apply_return(&mut self, function_index: usize, value: Option<Value>) {
        // Read the result handle from the callee before truncating the stack.
        let call_result_handle = match &self.stack[function_index] {
            StackFrame::Function(frame) => frame.call_result_handle,
            StackFrame::Block(_) => None,
        };
        // Store the return value in the parent frame's expression cache, keyed
        // by the CallResult expression handle so that each call's result is
        // independently retrievable even when multiple calls appear in sequence.
        if let (Some(handle), Some(return_val)) = (call_result_handle, value)
            && let Some(parent_function_index) = self.parent_function_frame_index(function_index)
            && let StackFrame::Function(ref mut parent_frame) = self.stack[parent_function_index]
        {
            parent_frame
                .evaluated_expressions
                .insert(handle, return_val);
        }
        self.stack.truncate(function_index);
    }
}

// Exhausted-frame handler
impl Evaluator {
    fn handle_exhausted_frame(&mut self, top: usize) {
        match &self.stack[top] {
            StackFrame::Function(_) => {
                self.stack.pop();
            }
            StackFrame::Block(block_frame) => match &block_frame.kind {
                BlockKind::Plain | BlockKind::Switch => {
                    self.stack.pop();
                }
                BlockKind::Loop {
                    in_continuing: false,
                    ..
                } => {
                    if let StackFrame::Block(ref mut bf) = self.stack[top] {
                        bf.switch_to_continuing();
                    }
                }
                BlockKind::Loop {
                    in_continuing: true,
                    break_if,
                    ..
                } => {
                    let should_break = if let Some(expr) = break_if {
                        self.evaluate_expression(*expr).is_truthy()
                    } else {
                        false
                    };

                    if should_break {
                        self.stack.pop();
                    } else if let StackFrame::Block(ref mut bf) = self.stack[top] {
                        bf.restart_body();
                    }
                }
            },
        }
    }
}

// Statement dispatch
impl Evaluator {
    fn handle_statement(
        &mut self,
        statement: Statement,
        function_index: usize,
        statement_span: naga::Span,
    ) {
        match statement {
            Statement::Emit(_) => {}
            Statement::Call {
                function: function_handle,
                arguments,
                result,
            } => {
                self.handle_call(function_handle, arguments, result);
            }
            Statement::Store { pointer, value } => {
                self.handle_store(pointer, value);
            }
            Statement::Return { value } => {
                let return_value = value.map(|v| self.evaluate_expression(v));
                if let StackFrame::Function(ref mut frame) = self.stack[function_index] {
                    frame.control_flow = ControlFlow::Return(return_value);
                }
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                self.handle_if(condition, accept, reject, statement_span);
            }
            Statement::Block(block) => {
                self.stack.push(StackFrame::Block(BlockFrame {
                    statements: block,
                    current_statement_index: 0,
                    kind: BlockKind::Plain,
                    source_span: statement_span,
                }));
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                self.stack.push(StackFrame::Block(BlockFrame {
                    statements: body,
                    current_statement_index: 0,
                    kind: BlockKind::Loop {
                        other_block: continuing,
                        break_if,
                        in_continuing: false,
                    },
                    source_span: statement_span,
                }));
            }
            Statement::Switch { selector, cases } => {
                self.handle_switch(selector, cases, statement_span);
            }
            Statement::Break => {
                if let StackFrame::Function(ref mut frame) = self.stack[function_index] {
                    frame.control_flow = ControlFlow::Break;
                }
            }
            Statement::Continue => {
                if let StackFrame::Function(ref mut frame) = self.stack[function_index] {
                    frame.control_flow = ControlFlow::Continue;
                }
            }
            Statement::Kill
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_)
            | Statement::ImageStore { .. }
            | Statement::Atomic { .. }
            | Statement::ImageAtomic { .. }
            | Statement::RayQuery { .. }
            | Statement::WorkGroupUniformLoad { .. }
            | Statement::SubgroupBallot { .. }
            | Statement::SubgroupGather { .. }
            | Statement::SubgroupCollectiveOperation { .. } => {}
        }
    }

    fn handle_call(
        &mut self,
        function_handle: naga::Handle<naga::Function>,
        arguments: Vec<Handle<Expression>>,
        call_result_handle: Option<Handle<Expression>>,
    ) {
        let evaluated_function_arguments = arguments
            .iter()
            .map(|&arg| self.evaluate_expression(arg))
            .collect();

        let statements = self.module.functions[function_handle].body.clone();

        self.stack
            .push(StackFrame::Function(Box::new(FunctionFrame {
                function_ref: FunctionRef::Called(function_handle),
                local_variables: HashMap::new(),
                evaluated_expressions: HashMap::new(),
                evaluated_function_arguments,
                statements,
                current_statement_index: 0,
                call_result_handle,
                control_flow: ControlFlow::None,
            })));

        self.initialize_local_variables();
    }

    fn initialize_local_variables(&mut self) {
        let function_index = match self.current_function_frame_index() {
            Some(i) => i,
            None => return,
        };

        let function = self.resolve_function_at(function_index);
        let local_vars: Vec<_> = function.local_variables.iter().collect();

        let mut insert_variables: Vec<(Handle<LocalVariable>, Value)> = Vec::new();
        for (handle, local_var) in local_vars {
            let value = match &local_var.init {
                Some(init_expr) => self.evaluate_expression(*init_expr),
                None => {
                    let ty = &self.module.types[local_var.ty];
                    Value::from(&ty.inner)
                }
            };
            insert_variables.push((handle, value));
        }

        let StackFrame::Function(ref mut frame) = self.stack[function_index] else {
            unreachable!();
        };
        for (handle, value) in insert_variables {
            frame
                .local_variables
                .insert(handle, Value::Pointer(Rc::new(RefCell::new(value))));
        }
    }

    fn handle_if(
        &mut self,
        condition: Handle<Expression>,
        accept: naga::Block,
        reject: naga::Block,
        statement_span: naga::Span,
    ) {
        let condition_result = self.evaluate_expression(condition);

        let branch = if condition_result.is_truthy() {
            accept
        } else {
            reject
        };

        if !branch.is_empty() {
            self.stack.push(StackFrame::Block(BlockFrame {
                statements: branch,
                current_statement_index: 0,
                kind: BlockKind::Plain,
                source_span: statement_span,
            }));
        }
    }

    fn handle_switch(
        &mut self,
        selector: Handle<Expression>,
        cases: Vec<naga::SwitchCase>,
        statement_span: naga::Span,
    ) {
        let selector_val = self.evaluate_expression(selector);

        let selector_i32 = match selector_val {
            Value::Primitive(Primitive::I32(v)) => v,
            Value::Primitive(Primitive::U32(v)) => v as i32,
            _ => return,
        };

        // Find the matching case, fall back to Default.
        let body = cases
            .iter()
            .find(|c| matches!(&c.value, naga::SwitchValue::I32(v) if *v == selector_i32))
            .or_else(|| {
                cases.iter().find(
                    |c| matches!(&c.value, naga::SwitchValue::U32(v) if *v == selector_i32 as u32),
                )
            })
            .or_else(|| {
                cases
                    .iter()
                    .find(|c| matches!(&c.value, naga::SwitchValue::Default))
            })
            .map(|c| c.body.clone());

        if let Some(body) = body
            && !body.is_empty()
        {
            self.stack.push(StackFrame::Block(BlockFrame {
                statements: body,
                current_statement_index: 0,
                kind: BlockKind::Switch,
                source_span: statement_span,
            }));
        }
    }

    fn handle_store(&mut self, pointer: Handle<Expression>, value: Handle<Expression>) {
        let evaluated_value = self.evaluate_expression(value);
        let evaluated_pointer = self.evaluate_expression(pointer);

        match evaluated_pointer {
            Value::Pointer(inner) => {
                *inner.borrow_mut() = evaluated_value;
            }
            _ => {
                eprintln!("Store to non-pointer value: {:?}", evaluated_pointer);
            }
        }
    }
}
