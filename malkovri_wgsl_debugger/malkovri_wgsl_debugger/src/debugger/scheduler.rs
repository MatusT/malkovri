use naga::Statement;

use crate::error::EvaluatorError;

use super::{DebugThreadId, Debugger, ParkReason, StepResult, ThreadStatus};

impl Debugger {
    pub fn step(&mut self) -> Result<StepResult, EvaluatorError> {
        self.step_gid(self.focused_thread)
    }

    pub fn step_thread(&mut self, thread_id: DebugThreadId) -> Result<StepResult, EvaluatorError> {
        self.focus_thread(thread_id)?;
        self.step_gid(self.focused_thread)
    }

    pub fn step_all(&mut self) -> Result<StepResult, EvaluatorError> {
        let focused_thread = self.focused_thread;
        let thread_order = self.thread_order.clone();
        for gid in thread_order {
            self.focused_thread = gid;
            self.step_gid(gid)?;
        }
        self.focused_thread = focused_thread;
        self.release_ready_parked_threads()?;
        self.detect_deadlock()?;
        Ok(self.session_step_result())
    }

    fn step_gid(&mut self, gid: [u32; 3]) -> Result<StepResult, EvaluatorError> {
        if matches!(self.thread_status.get(&gid), Some(ThreadStatus::Finished)) {
            return Ok(self.session_step_result());
        }
        if matches!(self.thread_status.get(&gid), Some(ThreadStatus::Parked(_))) {
            self.release_ready_parked_threads()?;
            self.detect_deadlock()?;
            return Ok(self.session_step_result());
        }

        self.focused_thread = gid;

        loop {
            let Some(next) = self.evaluator_mut().current_statement()? else {
                self.thread_status.insert(gid, ThreadStatus::Finished);
                self.release_ready_parked_threads()?;
                self.detect_deadlock()?;
                return Ok(self.session_step_result());
            };

            if let Some(reason) = Self::park_reason_for_statement(&next.statement) {
                self.thread_status.insert(gid, ThreadStatus::Parked(reason));
                self.release_ready_parked_threads()?;
                self.detect_deadlock()?;
                return Ok(self.session_step_result());
            }

            match self.evaluator_mut().step()? {
                None => {
                    self.thread_status.insert(gid, ThreadStatus::Finished);
                    self.release_ready_parked_threads()?;
                    self.detect_deadlock()?;
                    return Ok(self.session_step_result());
                }
                Some(next) if matches!(next.statement, Statement::Emit(_)) => continue,
                Some(_) => return Ok(StepResult::Continue),
            }
        }
    }

    fn session_step_result(&self) -> StepResult {
        if self
            .thread_order
            .iter()
            .all(|gid| matches!(self.thread_status.get(gid), Some(ThreadStatus::Finished)))
        {
            StepResult::Finished
        } else {
            StepResult::Continue
        }
    }

    fn park_reason_for_statement(statement: &Statement) -> Option<ParkReason> {
        match statement {
            Statement::ControlBarrier(barrier) | Statement::MemoryBarrier(barrier) => {
                Some(ParkReason::Barrier(*barrier))
            }
            Statement::WorkGroupUniformLoad { result, .. } => {
                Some(ParkReason::WorkGroupUniformLoad { result: *result })
            }
            Statement::SubgroupBallot { result, .. } => {
                Some(ParkReason::SubgroupBallot { result: *result })
            }
            Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                result,
                ..
            } => Some(ParkReason::SubgroupCollective {
                op: *op,
                collective_op: *collective_op,
                result: *result,
            }),
            Statement::SubgroupGather { mode, result, .. } => Some(ParkReason::SubgroupGather {
                mode: *mode,
                result: *result,
            }),
            _ => None,
        }
    }
}
