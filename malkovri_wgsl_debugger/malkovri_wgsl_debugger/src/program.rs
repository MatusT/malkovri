use std::{collections::HashMap, sync::Arc};

use naga::{Expression, Function, Handle, Module, Span, Statement, SwitchValue};

use crate::{
    debugger::{Debugger, DebuggerError, ResourceBinding, WorkgroupConfig},
    declaring_scopes::ModuleScopes,
    entry_point_inputs::GlobalConstants,
    value::Value,
    wgsl::{WgslToModuleError, wgsl_to_module},
};

/// Parsed shader code and derived metadata, shared by all invocations and sessions.
/// Execution never mutates the program. One session selects one of its entry points.
#[derive(Clone)]
pub struct ShaderProgram {
    data: Arc<ProgramData>,
}

struct ProgramData {
    source: String,
    module: Module,
    scopes: ModuleScopes,
    blocks: Vec<ProgramBlock>,
    function_bodies: HashMap<FunctionId, BlockId>,
}

/// An entry point available for an independent compute, vertex, or fragment run.
#[derive(Clone, Copy, Debug)]
pub struct EntryPointInfo<'a> {
    index: usize,
    name: &'a str,
    stage: naga::ShaderStage,
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
    pub fn new(source: &str) -> Result<Self, WgslToModuleError> {
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
        Ok(Self {
            data: Arc::new(ProgramData {
                source: source.to_owned(),
                module,
                scopes,
                blocks,
                function_bodies,
            }),
        })
    }

    /// Create one independently mutable execution of the selected entry point.
    /// Cloning a program or starting another debugger reuses its immutable data.
    pub fn create_debugger(
        &self,
        entry_point_index: usize,
        config: WorkgroupConfig,
        constants: GlobalConstants,
        bindings: HashMap<ResourceBinding, Value>,
    ) -> Result<Debugger, DebuggerError> {
        Debugger::new(self.clone(), entry_point_index, config, constants, bindings)
    }

    pub(crate) fn scopes(&self) -> &ModuleScopes {
        &self.data.scopes
    }

    pub fn source(&self) -> &str {
        &self.data.source
    }

    pub fn entry_points(&self) -> impl Iterator<Item = EntryPointInfo<'_>> {
        self.data
            .module
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
        &self.data.module
    }

    pub(crate) fn function(&self, function: FunctionId) -> &naga::Function {
        match function {
            FunctionId::EntryPoint(index) => &self.data.module.entry_points[index].function,
            FunctionId::Called(handle) => &self.data.module.functions[handle],
        }
    }

    pub(crate) fn function_body(&self, function: FunctionId) -> BlockId {
        self.data.function_bodies[&function]
    }

    pub(crate) fn block(&self, id: BlockId) -> &ProgramBlock {
        &self.data.blocks[id.0]
    }

    pub(crate) fn instruction(&self, id: StatementId) -> &Instruction {
        &self.data.blocks[id.block.0].instructions[id.index].0
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

impl<'a> EntryPointInfo<'a> {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn name(&self) -> &'a str {
        self.name
    }
    pub fn stage(&self) -> naga::ShaderStage {
        self.stage
    }
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
