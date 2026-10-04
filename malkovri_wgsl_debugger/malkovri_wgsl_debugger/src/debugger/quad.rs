use super::{Debugger, InvocationId, ThreadStatus};
use crate::{EvaluatorError, Primitive};
use naga::{DerivativeAxis, DerivativeControl};

impl Debugger {
    pub(super) fn release_quad(
        &mut self,
        members: Vec<InvocationId>,
    ) -> Result<(), EvaluatorError> {
        let error = |message: &str| {
            EvaluatorError::SynchronizationError(format!("quad derivative: {message}"))
        };
        if members.len() != 4 {
            return Err(error("requires four lanes"));
        }
        let waits = members
            .iter()
            .map(|id| {
                self.group
                    .get(*id)
                    .state()
                    .quad_wait()
                    .cloned()
                    .ok_or_else(|| error("missing operand"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let first = &waits[0];
        if waits.iter().any(|w| {
            w.key != first.key
                || w.result != first.result
                || w.axis != first.axis
                || w.control != first.control
        }) {
            return Err(error(
                "divergent dynamic operation (call, loop, or expression site)",
            ));
        }
        let operands = waits
            .iter()
            .map(|w| {
                w.operand
                    .as_primitive()
                    .and_then(Primitive::as_f32_slice)
                    .ok_or_else(|| error("operand must be a supported float scalar/vector"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if operands.iter().any(|v| v.len() != operands[0].len()) {
            return Err(error("operand shape mismatch"));
        }
        for (lane, id) in members.iter().enumerate() {
            let row = if first.control == DerivativeControl::Coarse {
                0
            } else {
                lane / 2 * 2
            };
            let col = if first.control == DerivativeControl::Coarse {
                0
            } else {
                lane % 2
            };
            let values = (0..operands[0].len())
                .map(|component| {
                    let dx = operands[row + 1][component] - operands[row][component];
                    let dy = operands[col + 2][component] - operands[col][component];
                    match first.axis {
                        DerivativeAxis::X => dx,
                        DerivativeAxis::Y => dy,
                        DerivativeAxis::Width => dx.abs() + dy.abs(),
                    }
                })
                .collect::<Vec<_>>();
            self.group
                .get_mut(*id)
                .state_mut()
                .complete_derivative(Primitive::from(&values).into())?;
            self.group.get_mut(*id).set_status(ThreadStatus::Running);
        }
        Ok(())
    }
}
