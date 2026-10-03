use naga::Barrier;

use crate::error::EvaluatorError;

use super::{Debugger, InvocationId, ParkReason, ParkScope, ThreadStatus};

struct ReadyGroup {
    reason: ParkReason,
    members: Vec<InvocationId>,
}

impl Debugger {
    pub(super) fn release_ready_parked_threads(&mut self) -> Result<(), EvaluatorError> {
        loop {
            let Some(ReadyGroup { reason, members }) = self.find_ready_parked_group()? else {
                return Ok(());
            };
            self.release_parked_group(reason, members)?;
        }
    }

    fn find_ready_parked_group(&self) -> Result<Option<ReadyGroup>, EvaluatorError> {
        for gid in self.group.ids() {
            let ThreadStatus::Parked(reason) = self.group.get(gid).status() else {
                continue;
            };
            let members = self.live_members_for_reason(gid, reason);
            if members.is_empty() {
                continue;
            }

            let all_parked = members
                .iter()
                .all(|member| matches!(self.group.get(*member).status(), ThreadStatus::Parked(_)));
            if !all_parked {
                continue;
            }

            let all_compatible = members.iter().all(|member| {
                matches!(
                    self.group.get(*member).status(),
                    ThreadStatus::Parked(member_reason) if member_reason == reason
                )
            });

            if all_compatible {
                return Ok(Some(ReadyGroup {
                    reason: reason.clone(),
                    members,
                }));
            }

            return Err(self.synchronization_error("divergent synchronization point", &members));
        }

        Ok(None)
    }

    pub(super) fn detect_deadlock(&self) -> Result<(), EvaluatorError> {
        let live: Vec<_> = self
            .group
            .ids()
            .filter(|gid| !matches!(self.group.get(*gid).status(), ThreadStatus::Finished))
            .collect();

        if live.is_empty() {
            return Ok(());
        }

        if live
            .iter()
            .all(|gid| matches!(self.group.get(*gid).status(), ThreadStatus::Parked(_)))
        {
            return Err(self.synchronization_error("deadlocked synchronization", &live));
        }

        Ok(())
    }

    fn synchronization_error(&self, label: &str, gids: &[InvocationId]) -> EvaluatorError {
        let threads = gids
            .iter()
            .map(|gid| {
                let thread_id = gid.thread_id();
                let status = self.group.get(*gid).status();
                format!("{thread_id}:{gid:?}={status:?}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        EvaluatorError::SynchronizationError(format!("{label}: {threads}"))
    }

    fn live_members_for_reason(&self, gid: InvocationId, reason: &ParkReason) -> Vec<InvocationId> {
        self.members_for_scope(self.scope_for_reason(gid, reason))
            .into_iter()
            .filter(|member| !matches!(self.group.get(*member).status(), ThreadStatus::Finished))
            .collect()
    }

    fn scope_for_reason(&self, gid: InvocationId, reason: &ParkReason) -> ParkScope {
        match reason {
            ParkReason::Barrier(barrier)
                if barrier.contains(Barrier::SUB_GROUP)
                    && !barrier
                        .intersects(Barrier::WORK_GROUP | Barrier::STORAGE | Barrier::TEXTURE) =>
            {
                ParkScope::Subgroup(self.subgroup_id(gid))
            }
            ParkReason::SubgroupBallot { .. }
            | ParkReason::SubgroupCollective { .. }
            | ParkReason::SubgroupGather { .. } => ParkScope::Subgroup(self.subgroup_id(gid)),
            ParkReason::Barrier(_) | ParkReason::WorkGroupUniformLoad { .. } => {
                ParkScope::Workgroup
            }
        }
    }

    fn members_for_scope(&self, scope: ParkScope) -> Vec<InvocationId> {
        match scope {
            ParkScope::Workgroup => self.group.ids().collect(),
            ParkScope::Subgroup(subgroup_id) => self
                .group
                .ids()
                .filter(|gid| self.subgroup_id(*gid) == subgroup_id)
                .collect(),
        }
    }

    fn subgroup_id(&self, id: InvocationId) -> u32 {
        self.group.get(id).state().subgroup_id()
    }

    pub(super) fn subgroup_lane(&self, id: InvocationId) -> u32 {
        self.group.get(id).state().subgroup_lane()
    }

    fn release_parked_group(
        &mut self,
        reason: ParkReason,
        members: Vec<InvocationId>,
    ) -> Result<(), EvaluatorError> {
        match reason {
            ParkReason::Barrier(_) => self.release_barrier(members),
            ParkReason::WorkGroupUniformLoad { result } => {
                self.release_workgroup_uniform_load(members, result)
            }
            ParkReason::SubgroupBallot { result } => self.release_subgroup_ballot(members, result),
            ParkReason::SubgroupCollective {
                op,
                collective_op,
                result,
            } => self.release_subgroup_collective(members, op, collective_op, result),
            ParkReason::SubgroupGather { mode, result } => {
                self.release_subgroup_gather(members, mode, result)
            }
        }
    }

    fn release_barrier(&mut self, members: Vec<InvocationId>) -> Result<(), EvaluatorError> {
        for gid in members {
            let next = {
                let evaluator = self.group.get_mut(gid).state_mut();
                evaluator.consume_current_statement_and_skip_emits()?
            };
            self.group.get_mut(gid).set_status(if next.is_some() {
                ThreadStatus::Running
            } else {
                ThreadStatus::Finished
            });
        }
        Ok(())
    }
}
