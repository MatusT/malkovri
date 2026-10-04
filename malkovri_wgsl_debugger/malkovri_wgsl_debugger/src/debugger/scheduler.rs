use naga::Statement;

use crate::error::EvaluatorError;

use super::{DebugThreadId, Debugger, InvocationId, ParkReason, StepResult, ThreadStatus};

impl Debugger {
    pub fn step(&mut self) -> Result<StepResult, EvaluatorError> {
        if self.is_empty() {
            return Ok(StepResult::Finished);
        }
        self.step_thread(self.focused_thread_id())
    }

    pub fn step_thread(&mut self, thread_id: DebugThreadId) -> Result<StepResult, EvaluatorError> {
        if self.is_empty() {
            return Ok(StepResult::Finished);
        }
        self.focus_thread(thread_id)?;
        let focused = self.focused_thread;
        let quad = self
            .group
            .get(focused)
            .state()
            .fragment_info()
            .map(|info| info.quad_index);
        if let Some(quad) = quad {
            let members = self
                .group
                .ids()
                .filter(|id| {
                    self.group
                        .get(*id)
                        .state()
                        .fragment_info()
                        .is_some_and(|info| info.quad_index == quad)
                })
                .collect::<Vec<_>>();
            for id in members {
                let result = self.step_gid(id);
                self.focused_thread = focused;
                result?;
            }
            Ok(self.session_step_result())
        } else {
            self.step_gid(focused)
        }
    }

    pub fn step_all(&mut self) -> Result<StepResult, EvaluatorError> {
        let focused_thread = self.focused_thread;
        let runnable_threads = self
            .group
            .ids()
            .filter(|&id| matches!(self.group.get(id).status(), ThreadStatus::Running))
            .collect::<Vec<_>>();

        for gid in runnable_threads {
            if !matches!(self.group.get(gid).status(), ThreadStatus::Running) {
                continue;
            }
            self.focused_thread = gid;
            let result = self.step_gid(gid);
            self.focused_thread = focused_thread;
            result?;
        }
        self.focused_thread = focused_thread;
        self.release_ready_parked_threads()?;
        self.detect_deadlock()?;
        Ok(self.session_step_result())
    }

    fn step_gid(&mut self, gid: InvocationId) -> Result<StepResult, EvaluatorError> {
        if matches!(self.group.get(gid).status(), ThreadStatus::Finished) {
            return Ok(self.session_step_result());
        }
        if matches!(self.group.get(gid).status(), ThreadStatus::Parked(_)) {
            self.release_ready_parked_threads()?;
            self.detect_deadlock()?;
            return Ok(self.session_step_result());
        }

        self.focused_thread = gid;

        loop {
            let Some(next) = self.invocation_mut().current_statement()? else {
                self.group.get_mut(gid).set_status(ThreadStatus::Finished);
                self.release_ready_parked_threads()?;
                self.detect_deadlock()?;
                return Ok(self.session_step_result());
            };

            if let Some(reason) = self
                .program
                .instruction(next)
                .leaf()
                .and_then(Self::park_reason_for_statement)
            {
                if self.invocation().fragment_info().is_some() {
                    return Err(EvaluatorError::UnsupportedStatement(
                        "fragment subgroup/barrier operations are not supported".into(),
                    ));
                }
                self.group
                    .get_mut(gid)
                    .set_status(ThreadStatus::Parked(reason));
                self.release_ready_parked_threads()?;
                self.detect_deadlock()?;
                return Ok(self.session_step_result());
            }

            let next = self.invocation_mut().step()?;
            if self.invocation().quad_wait().is_some() {
                self.group
                    .get_mut(gid)
                    .set_status(ThreadStatus::Parked(ParkReason::Quad));
                self.release_ready_parked_threads()?;
                self.detect_deadlock()?;
                return Ok(StepResult::Continue);
            }
            match next {
                None => {
                    self.group.get_mut(gid).set_status(ThreadStatus::Finished);
                    self.release_ready_parked_threads()?;
                    self.detect_deadlock()?;
                    return Ok(self.session_step_result());
                }
                Some(next) if self.program.instruction(next).is_emit() => continue,
                Some(_) => return Ok(StepResult::Continue),
            }
        }
    }

    fn session_step_result(&self) -> StepResult {
        if self
            .group
            .ids()
            .all(|id| matches!(self.group.get(id).status(), ThreadStatus::Finished))
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
