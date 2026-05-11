use std::collections::HashMap;

use naga::{Expression, Handle, LocalVariable, Statement};

use crate::{
    error::EvaluatorError,
    function_state::{BlockFrame, BlockKind, ControlFlow, FunctionFrame, FunctionRef, StackFrame},
    place::{ArgumentValue, EvaluatedExpression, Place, PlaceRoot},
    primitive::Primitive,
    value::Value,
};

use super::Evaluator;

impl Evaluator {
    // Statement dispatch
    pub(super) fn handle_statement(&mut self, statement: Statement) -> Result<(), EvaluatorError> {
        match statement {
            Statement::Emit(range) => {
                self.initialize_local_variables_for_emit(range)?;
            }
            Statement::Call {
                function: function_handle,
                arguments,
                result,
            } => {
                self.handle_call(function_handle, arguments, result)?;
            }
            Statement::Store { pointer, value } => {
                self.handle_store(pointer, value)?;
            }
            Statement::Return { value } => {
                let return_value = value.map(|v| self.evaluate_expression(v));
                self.current_function_frame_mut()?.control_flow = ControlFlow::Return(return_value);
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                self.handle_if(condition, accept, reject);
            }
            Statement::Block(block) => {
                self.stack.push(StackFrame::Block(BlockFrame {
                    statements: block,
                    current_statement_index: 0,
                    kind: BlockKind::Plain,
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
                }));
            }
            Statement::Switch { selector, cases } => {
                self.handle_switch(selector, cases)?;
            }
            Statement::Break => {
                self.current_function_frame_mut()?.control_flow = ControlFlow::Break;
            }
            Statement::Continue => {
                self.current_function_frame_mut()?.control_flow = ControlFlow::Continue;
            }
            Statement::ControlBarrier(_) | Statement::MemoryBarrier(_) => {}
            Statement::Kill => {
                self.current_function_frame_mut()?.control_flow = ControlFlow::Return(None);
            }
            Statement::ImageStore { .. } => {
                return Err(EvaluatorError::UnsupportedStatement("imageStore".into()));
            }
            Statement::Atomic { .. } => {
                return Err(EvaluatorError::UnsupportedStatement("atomic".into()));
            }
            Statement::ImageAtomic { .. } => {
                return Err(EvaluatorError::UnsupportedStatement("imageAtomic".into()));
            }
            Statement::RayQuery { .. } => {
                return Err(EvaluatorError::UnsupportedStatement("rayQuery".into()));
            }
            Statement::WorkGroupUniformLoad { .. } => {
                return Err(EvaluatorError::UnsupportedStatement(
                    "workGroupUniformLoad".into(),
                ));
            }
            Statement::SubgroupBallot { .. }
            | Statement::SubgroupGather { .. }
            | Statement::SubgroupCollectiveOperation { .. } => {
                return Err(EvaluatorError::UnsupportedStatement(
                    "subgroup operation".into(),
                ));
            }
        }
        Ok(())
    }

    fn handle_call(
        &mut self,
        function_handle: naga::Handle<naga::Function>,
        arguments: Vec<Handle<Expression>>,
        call_result_handle: Option<Handle<Expression>>,
    ) -> Result<(), EvaluatorError> {
        let evaluated_function_arguments = arguments
            .iter()
            .map(|&arg| self.evaluate_argument(arg))
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

        Ok(())
    }

    fn initialize_local_variables_for_emit(
        &mut self,
        range: naga::Range<Expression>,
    ) -> Result<(), EvaluatorError> {
        let mut insert_variables: Vec<(Handle<LocalVariable>, Value)> = Vec::new();
        {
            let function = self.current_function()?;
            let local_vars: Vec<_> = function.local_variables.iter().collect();
            for (handle, local_var) in local_vars {
                let Some(init_expr) = local_var.init else {
                    continue;
                };
                if !range.clone().any(|emitted| emitted == init_expr) {
                    continue;
                };
                let value = self.evaluate_expression(init_expr);
                insert_variables.push((handle, value));
            }
        }

        let frame = self.current_function_frame_mut()?;
        for (handle, value) in insert_variables {
            frame.local_variables.insert(handle, value);
        }
        Ok(())
    }

    fn handle_if(
        &mut self,
        condition: Handle<Expression>,
        accept: naga::Block,
        reject: naga::Block,
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
            }));
        }
    }

    fn handle_switch(
        &mut self,
        selector: Handle<Expression>,
        cases: Vec<naga::SwitchCase>,
    ) -> Result<(), EvaluatorError> {
        let selector_val = self.evaluate_expression(selector);

        let selector_i32 = match selector_val {
            Value::Primitive(Primitive::I32(v)) => v,
            Value::Primitive(Primitive::U32(v)) => v as i32,
            _ => {
                return Err(EvaluatorError::InternalError(format!(
                    "switch selector is not an integer: {:?}",
                    selector_val
                )));
            }
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
            }));
        }
        Ok(())
    }

    fn handle_store(
        &mut self,
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        let evaluated_value = self.evaluate_expression(value);
        self.assign_store_pointer(pointer, evaluated_value)
    }

    fn assign_store_pointer(
        &mut self,
        pointer: Handle<Expression>,
        value: Value,
    ) -> Result<(), EvaluatorError> {
        let place = self.resolve_pointer_place(pointer)?;
        self.write_place(&place, value)
    }

    pub(crate) fn resolve_pointer_place(
        &mut self,
        pointer: Handle<Expression>,
    ) -> Result<Place, EvaluatorError> {
        let func_idx = self.current_function_frame_index()?;
        let expression = {
            let frame = self.current_function_frame()?;
            let function = self.resolve_function(&frame.function_ref);
            function.expressions[pointer].clone()
        };

        match expression {
            Expression::LocalVariable(handle) => {
                self.ensure_local_variable_value(handle, func_idx)?;
                Ok(Place::new(PlaceRoot::Local {
                    function_frame_index: func_idx,
                    handle,
                }))
            }
            Expression::GlobalVariable(handle) => {
                self.ensure_global_variable_value(handle)?;
                Ok(Place::new(PlaceRoot::Global { handle }))
            }
            Expression::FunctionArgument(index) => {
                match self.evaluate_argument_at(index as usize, func_idx) {
                    ArgumentValue::Place(place) => Ok(place),
                    ArgumentValue::Value(_) => Err(EvaluatorError::StoreToNonPointer),
                }
            }
            Expression::AccessIndex { base, index } => {
                let place = self.resolve_pointer_place(base)?;
                Ok(place.with_index(index as usize))
            }
            Expression::Access { base, index } => {
                let index = self.evaluate_expression(index);
                let index = match index {
                    Value::Primitive(Primitive::U32(value)) => value as usize,
                    Value::Primitive(Primitive::I32(value)) if value >= 0 => value as usize,
                    other => return Err(EvaluatorError::IndexNotU32(format!("{other:?}"))),
                };
                let place = self.resolve_pointer_place(base)?;
                Ok(place.with_index(index))
            }
            _ => match self.eval_expr(pointer, func_idx) {
                EvaluatedExpression::Place(place) => Ok(place),
                EvaluatedExpression::Value(_) => Err(EvaluatorError::StoreToNonPointer),
            },
        }
    }
}
