use super::{InvocationState, frame::StackFrame};
use crate::{EvaluatorError, Value, program::BlockId};
use naga::{DerivativeAxis, DerivativeControl, Expression, Handle};

#[derive(Clone, Debug)]
pub(crate) struct QuadWait {
    pub key: Vec<(BlockId, usize, u64)>,
    pub result: Handle<Expression>,
    pub axis: DerivativeAxis,
    pub control: DerivativeControl,
    pub operand: Value,
}

impl InvocationState {
    pub(crate) fn quad_wait(&self) -> Option<&QuadWait> {
        self.quad_wait.as_ref()
    }

    pub(super) fn begin_derivative(
        &mut self,
        result: Handle<Expression>,
        axis: DerivativeAxis,
        control: DerivativeControl,
        operand: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        if self.fragment_info().is_none() {
            return Err(EvaluatorError::UnsupportedStatement(
                "derivatives require generated fragment quads".into(),
            ));
        }
        let key = self
            .stack
            .iter()
            .map(|frame| {
                (
                    frame.block(),
                    frame.current_statement_index(),
                    match frame {
                        StackFrame::Block(block) => block.iteration(),
                        _ => 0,
                    },
                )
            })
            .collect();
        self.quad_wait = Some(QuadWait {
            key,
            result,
            axis,
            control,
            operand: self.evaluate_expression(operand),
        });
        Ok(())
    }

    pub(crate) fn complete_derivative(&mut self, value: Value) -> Result<(), EvaluatorError> {
        let wait = self
            .quad_wait
            .take()
            .ok_or_else(|| EvaluatorError::InternalError("missing quad operation".into()))?;
        self.set_current_expression_value(wait.result, value)?;
        self.emit_index = Some(self.emit_index.unwrap_or(0) + 1);
        Ok(())
    }
}
