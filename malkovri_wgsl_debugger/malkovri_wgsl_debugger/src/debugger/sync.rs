use naga::Barrier;

use crate::error::EvaluatorError;

use super::{Debugger, ParkReason, ParkScope, ThreadStatus};

impl Debugger {
    pub(super) fn release_ready_parked_threads(&mut self) -> Result<(), EvaluatorError> {
        loop {
            let Some((reason, members)) = self.find_ready_parked_group()? else {
                return Ok(());
            };
            self.release_parked_group(reason, members)?;
        }
    }

    fn find_ready_parked_group(
        &self,
    ) -> Result<Option<(ParkReason, Vec<[u32; 3]>)>, EvaluatorError> {
        for gid in &self.thread_order {
            let Some(ThreadStatus::Parked(reason)) = self.thread_status.get(gid) else {
                continue;
            };
            let members = self.live_members_for_reason(*gid, reason);
            if members.is_empty() {
                continue;
            }

            let all_parked = members.iter().all(|member| {
                matches!(
                    self.thread_status.get(member),
                    Some(ThreadStatus::Parked(_))
                )
            });
            if !all_parked {
                continue;
            }

            let all_compatible = members.iter().all(|member| {
                matches!(
                    self.thread_status.get(member),
                    Some(ThreadStatus::Parked(member_reason)) if member_reason == reason
                )
            });

            if all_compatible {
                return Ok(Some((reason.clone(), members)));
            }

            return Err(self.synchronization_error("divergent synchronization point", &members));
        }

        Ok(None)
    }

    pub(super) fn detect_deadlock(&self) -> Result<(), EvaluatorError> {
        let live: Vec<_> = self
            .thread_order
            .iter()
            .copied()
            .filter(|gid| !matches!(self.thread_status.get(gid), Some(ThreadStatus::Finished)))
            .collect();

        if live.is_empty() {
            return Ok(());
        }

        if live
            .iter()
            .all(|gid| matches!(self.thread_status.get(gid), Some(ThreadStatus::Parked(_))))
        {
            return Err(self.synchronization_error("deadlocked synchronization", &live));
        }

        Ok(())
    }

    fn synchronization_error(&self, label: &str, gids: &[[u32; 3]]) -> EvaluatorError {
        let threads = gids
            .iter()
            .map(|gid| {
                let thread_id = self.thread_id_for_gid(*gid);
                let status = self.thread_status.get(gid);
                format!("{thread_id}:{gid:?}={status:?}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        EvaluatorError::SynchronizationError(format!("{label}: {threads}"))
    }

    pub(super) fn thread_id_for_gid(&self, gid: [u32; 3]) -> u64 {
        self.thread_order
            .iter()
            .position(|candidate| *candidate == gid)
            .map(|index| index as u64 + 1)
            .unwrap_or(1)
    }

    fn live_members_for_reason(&self, gid: [u32; 3], reason: &ParkReason) -> Vec<[u32; 3]> {
        self.members_for_scope(self.scope_for_reason(gid, reason))
            .into_iter()
            .filter(|member| {
                !matches!(self.thread_status.get(member), Some(ThreadStatus::Finished))
            })
            .collect()
    }

    fn scope_for_reason(&self, gid: [u32; 3], reason: &ParkReason) -> ParkScope {
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

    fn members_for_scope(&self, scope: ParkScope) -> Vec<[u32; 3]> {
        match scope {
            ParkScope::Workgroup => self.thread_order.clone(),
            ParkScope::Subgroup(subgroup_id) => self
                .thread_order
                .iter()
                .copied()
                .filter(|gid| self.subgroup_id(*gid) == subgroup_id)
                .collect(),
        }
    }

    fn subgroup_id(&self, gid: [u32; 3]) -> u32 {
        self.evaluators[&gid]
            .active_thread()
            .compute_inputs
            .subgroup_id
    }

    pub(super) fn subgroup_lane(&self, gid: [u32; 3]) -> u32 {
        self.evaluators[&gid]
            .active_thread()
            .compute_inputs
            .subgroup_invocation_id
    }

    fn release_parked_group(
        &mut self,
        reason: ParkReason,
        members: Vec<[u32; 3]>,
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

    fn release_barrier(&mut self, members: Vec<[u32; 3]>) -> Result<(), EvaluatorError> {
        for gid in members {
            let next = {
                let evaluator = self.evaluators.get_mut(&gid).ok_or_else(|| {
                    EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
                })?;
                evaluator.consume_current_statement_without_running()?
            };
            self.thread_status.insert(
                gid,
                if next.is_some() {
                    ThreadStatus::Running
                } else {
                    ThreadStatus::Finished
                },
            );
        }
        Ok(())
    }
}
