pub(crate) mod interface;
mod parse;
mod scopes;

pub use parse::WgslToModuleError;
use parse::wgsl_to_module;
use scopes::ModuleScopes;

use std::{collections::HashMap, sync::Arc};

use naga::{Expression, Function, Handle, Module, ResourceBinding, Span, Statement, SwitchValue};

use crate::{
    debugger::{Debugger, DebuggerError, ExecutionConfig},
    invocation::inputs::GlobalConstants,
    value::Value,
};

/// Parsed shader code and derived metadata, shared by all invocations and sessions.
/// Execution never mutates the program. One session selects one of its entry points.
pub struct ShaderProgram {
    source: String,
    module: Module,
    scopes: ModuleScopes,
    blocks: Vec<ProgramBlock>,
    function_bodies: HashMap<FunctionId, BlockId>,
}

/// An entry point available for an independent compute, vertex, or fragment run.
#[derive(Clone, Copy, Debug)]
pub struct EntryPointInfo<'a> {
    pub index: usize,
    pub name: &'a str,
    pub stage: naga::ShaderStage,
}

/// Identifies a function in the module without owning/cloning it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FunctionId {
    /// An entry-point function, looked up via `module.entry_points[index].function`.
    EntryPoint(usize),
    /// A regular function, looked up via `module.functions[handle]`.
    Called(Handle<Function>),
}

/// Stable within one immutable program; never an address into an invocation stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BlockId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StatementId {
    block: BlockId,
    index: usize,
}

/// Structured control flow uses block IDs. All other operations retain Naga's
/// representation and are borrowed during execution, including call arguments.
pub(crate) enum Instruction {
    Block(BlockId),
    If {
        condition: Handle<Expression>,
        accept: BlockId,
        reject: BlockId,
    },
    Loop {
        body: BlockId,
        continuing: BlockId,
        break_if: Option<Handle<Expression>>,
    },
    Switch {
        selector: Handle<Expression>,
        cases: Vec<SwitchCase>,
    },
    Leaf(Statement),
}

impl Instruction {
    pub fn leaf(&self) -> Option<&Statement> {
        match self {
            Self::Leaf(statement) => Some(statement),
            _ => None,
        }
    }

    pub fn is_emit(&self) -> bool {
        matches!(self, Self::Leaf(Statement::Emit(_)))
    }
}

pub(crate) struct SwitchCase {
    value: SwitchValue,
    body: BlockId,
    fall_through: bool,
}

#[derive(Default)]
pub(crate) struct ProgramBlock {
    instructions: Vec<(Instruction, Span)>,
}

impl ProgramBlock {
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn span_iter(&self) -> impl Iterator<Item = (&Instruction, &Span)> {
        self.instructions
            .iter()
            .map(|(instruction, span)| (instruction, span))
    }

    pub fn get(&self, index: usize) -> Option<&Instruction> {
        self.instructions
            .get(index)
            .map(|(instruction, _)| instruction)
    }
}

impl ShaderProgram {
    /// Parse and validate a shader once, retaining immutable data for any sessions
    /// created from the returned shared handle.
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use malkovri_wgsl_debugger::{GlobalConstants, ShaderProgram, WorkgroupConfig};
    ///
    /// let program = ShaderProgram::new("@compute @workgroup_size(1) fn main() {}")?;
    /// let mut debugger = program.create_debugger(
    ///     0, WorkgroupConfig::default(), GlobalConstants::default(), HashMap::new(),
    /// )?;
    /// let result = debugger.run_to_breakpoint(1, false, &[], None)?;
    /// assert_eq!(result, malkovri_wgsl_debugger::RunResult::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(source: &str) -> Result<Arc<Self>, WgslToModuleError> {
        let module: Module = wgsl_to_module(source)?;
        let scopes = ModuleScopes::new(&module);
        let mut blocks = Vec::new();
        let mut function_bodies = HashMap::new();
        for (handle, function) in module.functions.iter() {
            function_bodies.insert(
                FunctionId::Called(handle),
                index_block(&function.body, &mut blocks),
            );
        }
        for (index, entry) in module.entry_points.iter().enumerate() {
            function_bodies.insert(
                FunctionId::EntryPoint(index),
                index_block(&entry.function.body, &mut blocks),
            );
        }
        Ok(Arc::new(Self {
            source: source.to_owned(),
            module,
            scopes,
            blocks,
            function_bodies,
        }))
    }

    /// Create one independently mutable execution of the selected entry point.
    /// Each debugger shares this program through Arc and owns its execution state.
    /// Pass [`crate::WorkgroupConfig`] for compute, [`crate::DrawConfig`] for vertex,
    /// or [`ExecutionConfig::Fragment`] for a single fragment with default inputs.
    /// The configuration must match the selected entry point's stage.
    pub fn create_debugger(
        self: &Arc<Self>,
        entry_point_index: usize,
        config: impl Into<ExecutionConfig>,
        constants: GlobalConstants,
        bindings: HashMap<ResourceBinding, Value>,
    ) -> Result<Debugger, DebuggerError> {
        Debugger::new(
            self.clone(),
            entry_point_index,
            config.into(),
            constants,
            bindings,
        )
    }

    pub(crate) fn scopes(&self) -> &ModuleScopes {
        &self.scopes
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn entry_points(&self) -> impl Iterator<Item = EntryPointInfo<'_>> {
        self.module
            .entry_points
            .iter()
            .enumerate()
            .map(|(index, entry)| EntryPointInfo {
                index,
                name: &entry.name,
                stage: entry.stage,
            })
    }

    pub(crate) fn module(&self) -> &Module {
        &self.module
    }

    pub(crate) fn function(&self, function: FunctionId) -> &naga::Function {
        match function {
            FunctionId::EntryPoint(index) => &self.module.entry_points[index].function,
            FunctionId::Called(handle) => &self.module.functions[handle],
        }
    }

    pub(crate) fn function_body(&self, function: FunctionId) -> BlockId {
        self.function_bodies[&function]
    }

    pub(crate) fn block(&self, id: BlockId) -> &ProgramBlock {
        &self.blocks[id.0]
    }

    pub(crate) fn instruction(&self, id: StatementId) -> &Instruction {
        &self.blocks[id.block.0].instructions[id.index].0
    }
}

fn index_block(block: &naga::Block, blocks: &mut Vec<ProgramBlock>) -> BlockId {
    let id = BlockId(blocks.len());
    blocks.push(ProgramBlock::default());
    let instructions = block
        .span_iter()
        .map(|(statement, span)| {
            let instruction = match statement {
                Statement::Block(block) => Instruction::Block(index_block(block, blocks)),
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => Instruction::If {
                    condition: *condition,
                    accept: index_block(accept, blocks),
                    reject: index_block(reject, blocks),
                },
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => Instruction::Loop {
                    body: index_block(body, blocks),
                    continuing: index_block(continuing, blocks),
                    break_if: *break_if,
                },
                Statement::Switch { selector, cases } => Instruction::Switch {
                    selector: *selector,
                    cases: cases
                        .iter()
                        .map(|case| SwitchCase {
                            value: case.value,
                            body: index_block(&case.body, blocks),
                            fall_through: case.fall_through,
                        })
                        .collect(),
                },
                // Leaf statements contain handles and operands, never nested blocks.
                leaf => Instruction::Leaf(leaf.clone()),
            };
            (instruction, *span)
        })
        .collect();
    blocks[id.0] = ProgramBlock { instructions };
    id
}

impl StatementId {
    pub fn new(block: BlockId, index: usize) -> Self {
        Self { block, index }
    }
    pub fn block(self) -> BlockId {
        self.block
    }
    pub fn index(self) -> usize {
        self.index
    }
}

impl SwitchCase {
    pub fn value(&self) -> SwitchValue {
        self.value
    }
    pub fn body(&self) -> BlockId {
        self.body
    }
    pub fn falls_through(&self) -> bool {
        self.fall_through
    }
}
