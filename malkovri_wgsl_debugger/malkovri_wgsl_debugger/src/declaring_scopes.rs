use std::collections::HashMap;

use naga::{Expression, Handle, LocalVariable, Module, Span, Statement};

use crate::function_state::FunctionRef;

/// Pre-computed declaring-scope maps for every function in a module.
pub struct ModuleScopes {
    functions: HashMap<FunctionRef, FunctionScopes>,
}

impl ModuleScopes {
    pub fn new(module: &Module) -> Self {
        let mut functions = HashMap::new();

        for (handle, function) in module.functions.iter() {
            functions.insert(FunctionRef::Called(handle), FunctionScopes::new(function));
        }
        for (i, entry_point) in module.entry_points.iter().enumerate() {
            functions.insert(
                FunctionRef::EntryPoint(i),
                FunctionScopes::new(&entry_point.function),
            );
        }

        Self { functions }
    }

    pub fn local_scopes(
        &self,
        fref: &FunctionRef,
    ) -> Option<&HashMap<Handle<LocalVariable>, std::ops::Range<usize>>> {
        self.functions.get(fref).map(|s| &s.locals)
    }

    pub fn named_expr_scopes(
        &self,
        fref: &FunctionRef,
    ) -> Option<&HashMap<Handle<Expression>, std::ops::Range<usize>>> {
        self.functions.get(fref).map(|s| &s.named_exprs)
    }
}

/// Pre-computed scope maps for a single function: maps each local variable and
/// each named expression to the source range of the block that declares it.
struct FunctionScopes {
    locals: HashMap<Handle<LocalVariable>, std::ops::Range<usize>>,
    named_exprs: HashMap<Handle<Expression>, std::ops::Range<usize>>,
}

impl FunctionScopes {
    fn new(function: &naga::Function) -> Self {
        let block_scopes = collect_all_block_scopes(function);
        Self {
            locals: local_declaring_scopes(function, &block_scopes),
            named_exprs: named_expression_declaring_scopes(function, &block_scopes),
        }
    }
}

/// Collect all block scopes (the function body + nested blocks) for scope analysis.
fn collect_all_block_scopes(function: &naga::Function) -> Vec<Span> {
    let mut block_scopes = vec![Span::total_span(function.body.span_iter().map(|(_, s)| *s))];
    block_scopes.extend(collect_block_spans(function.body.span_iter()));
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
            Some((handle, smallest_enclosing_scope(block_scopes, &variable_range)))
        })
        .collect()
}

fn named_expression_declaring_scopes(
    function: &naga::Function,
    block_scopes: &[Span],
) -> HashMap<Handle<Expression>, std::ops::Range<usize>> {
    // Build a map from expression handle → span of the Emit statement that covers it.
    let mut emit_spans: HashMap<Handle<Expression>, Span> = HashMap::new();
    collect_emit_spans(&function.body, &mut emit_spans);

    function
        .named_expressions
        .iter()
        .map(|(handle, _)| {
            let scope = emit_spans
                .get(handle)
                .and_then(|sp| sp.to_range())
                .map(|r| smallest_enclosing_scope(block_scopes, &r))
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
            Statement::Block(inner) => collect_emit_spans(inner, out),
            Statement::If { accept, reject, .. } => {
                collect_emit_spans(accept, out);
                collect_emit_spans(reject, out);
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                collect_emit_spans(body, out);
                collect_emit_spans(continuing, out);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_emit_spans(&case.body, out);
                }
            }
            _ => {}
        }
    }
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
                spans.extend(collect_block_spans(inner.span_iter()));
            }
            Statement::If { accept, reject, .. } => {
                spans.push(Span::total_span(accept.span_iter().map(|(_, s)| *s)));
                spans.extend(collect_block_spans(accept.span_iter()));
                spans.push(Span::total_span(reject.span_iter().map(|(_, s)| *s)));
                spans.extend(collect_block_spans(reject.span_iter()));
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                spans.push(*statement_span);
                spans.extend(collect_block_spans(body.span_iter()));
                spans.extend(collect_block_spans(continuing.span_iter()));
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    spans.push(Span::total_span(case.body.span_iter().map(|(_, s)| *s)));
                    spans.extend(collect_block_spans(case.body.span_iter()));
                }
            }
            _ => {}
        }
    }
    spans
}
