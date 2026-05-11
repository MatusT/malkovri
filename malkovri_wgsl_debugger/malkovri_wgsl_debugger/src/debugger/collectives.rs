use std::collections::HashMap;

use naga::{CollectiveOperation, Expression, GatherMode, Handle, Statement, SubgroupOperation};

use crate::{error::EvaluatorError, evaluator::Evaluator, primitive::Primitive, value::Value};

use super::{Debugger, ThreadStatus};

impl Debugger {
    pub(super) fn release_workgroup_uniform_load(
        &mut self,
        members: Vec<[u32; 3]>,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        let mut agreed_place = None;
        let mut loaded_value = None;

        for gid in &members {
            let evaluator = self.evaluators.get_mut(gid).ok_or_else(|| {
                EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
            })?;
            let Some(next) = evaluator.current_statement()? else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} finished while parked at workGroupUniformLoad",
                    self.thread_id_for_gid(*gid)
                )));
            };
            let Statement::WorkGroupUniformLoad {
                pointer,
                result: found,
            } = next.statement
            else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} is not parked at workGroupUniformLoad",
                    self.thread_id_for_gid(*gid)
                )));
            };
            if found != result {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} reached a different workGroupUniformLoad site",
                    self.thread_id_for_gid(*gid)
                )));
            }

            let place = evaluator.resolve_pointer_place(pointer)?;
            if let Some(expected) = &agreed_place {
                if expected != &place {
                    return Err(EvaluatorError::SynchronizationError(format!(
                        "workGroupUniformLoad pointer mismatch at thread {}",
                        self.thread_id_for_gid(*gid)
                    )));
                }
            } else {
                loaded_value = Some(evaluator.read_place(&place));
                agreed_place = Some(place);
            }
        }

        let value = loaded_value.unwrap_or(Value::Uninitialized);
        self.inject_statement_result_and_consume(members, result, value)
    }

    pub(super) fn release_subgroup_ballot(
        &mut self,
        members: Vec<[u32; 3]>,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        let mut ballot = [0u32; 4];
        for gid in &members {
            let lane = self.subgroup_lane(*gid);
            let predicate = {
                let evaluator = self.evaluators.get_mut(gid).ok_or_else(|| {
                    EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
                })?;
                let Some(next) = evaluator.current_statement()? else {
                    return Err(EvaluatorError::SynchronizationError(format!(
                        "thread {} finished while parked at subgroupBallot",
                        self.thread_id_for_gid(*gid)
                    )));
                };
                let Statement::SubgroupBallot {
                    result: found,
                    predicate,
                } = next.statement
                else {
                    return Err(EvaluatorError::SynchronizationError(format!(
                        "thread {} is not parked at subgroupBallot",
                        self.thread_id_for_gid(*gid)
                    )));
                };
                if found != result {
                    return Err(EvaluatorError::SynchronizationError(format!(
                        "thread {} reached a different subgroupBallot site",
                        self.thread_id_for_gid(*gid)
                    )));
                }
                predicate
                    .map(|expr| evaluator.evaluate_expression(expr).is_truthy())
                    .unwrap_or(true)
            };
            if predicate {
                ballot[(lane / 32) as usize] |= 1u32 << (lane % 32);
            }
        }

        self.inject_statement_result_and_consume(
            members,
            result,
            Value::Primitive(Primitive::U32x4(ballot)),
        )
    }

    pub(super) fn release_subgroup_collective(
        &mut self,
        mut members: Vec<[u32; 3]>,
        op: SubgroupOperation,
        collective_op: CollectiveOperation,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        members.sort_by_key(|gid| self.subgroup_lane(*gid));

        let mut lane_values = Vec::new();
        for gid in &members {
            let evaluator = self.evaluators.get_mut(gid).ok_or_else(|| {
                EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
            })?;
            let Some(next) = evaluator.current_statement()? else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} finished while parked at subgroup collective",
                    self.thread_id_for_gid(*gid)
                )));
            };
            let Statement::SubgroupCollectiveOperation {
                op: found_op,
                collective_op: found_collective_op,
                argument,
                result: found_result,
            } = next.statement
            else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} is not parked at subgroup collective",
                    self.thread_id_for_gid(*gid)
                )));
            };
            if found_op != op || found_collective_op != collective_op || found_result != result {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} reached a different subgroup collective site",
                    self.thread_id_for_gid(*gid)
                )));
            }
            lane_values.push((*gid, evaluator.evaluate_expression(argument)));
        }

        let results = match collective_op {
            CollectiveOperation::Reduce => {
                let value = Self::reduce_subgroup_values(
                    op,
                    lane_values.iter().map(|(_, value)| value.clone()).collect(),
                )?;
                lane_values
                    .iter()
                    .map(|(gid, _)| (*gid, value.clone()))
                    .collect()
            }
            CollectiveOperation::InclusiveScan => {
                Self::scan_subgroup_values(op, &lane_values, true)?
            }
            CollectiveOperation::ExclusiveScan => {
                Self::scan_subgroup_values(op, &lane_values, false)?
            }
        };

        self.inject_many_statement_results_and_consume(results, result)
    }

    pub(super) fn release_subgroup_gather(
        &mut self,
        mut members: Vec<[u32; 3]>,
        mode: GatherMode,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        members.sort_by_key(|gid| self.subgroup_lane(*gid));

        let mut lane_values = HashMap::new();
        let mut target_lanes = HashMap::new();
        for gid in &members {
            let lane = self.subgroup_lane(*gid);
            let evaluator = self.evaluators.get_mut(gid).ok_or_else(|| {
                EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
            })?;
            let Some(next) = evaluator.current_statement()? else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} finished while parked at subgroup gather",
                    self.thread_id_for_gid(*gid)
                )));
            };
            let Statement::SubgroupGather {
                mode: found_mode,
                argument,
                result: found_result,
            } = next.statement
            else {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} is not parked at subgroup gather",
                    self.thread_id_for_gid(*gid)
                )));
            };
            if found_mode != mode || found_result != result {
                return Err(EvaluatorError::SynchronizationError(format!(
                    "thread {} reached a different subgroup gather site",
                    self.thread_id_for_gid(*gid)
                )));
            }
            lane_values.insert(lane, evaluator.evaluate_expression(argument));
            target_lanes.insert(lane, Self::gather_target_lane(evaluator, lane, mode));
        }

        let first_lane = lane_values.keys().min().copied().unwrap_or(0);
        let results = members
            .iter()
            .map(|gid| {
                let lane = self.subgroup_lane(*gid);
                let target = match mode {
                    GatherMode::BroadcastFirst => first_lane,
                    _ => target_lanes.get(&lane).copied().unwrap_or(lane),
                };
                let value = lane_values
                    .get(&target)
                    .cloned()
                    .unwrap_or(Value::Uninitialized);
                (*gid, value)
            })
            .collect();

        self.inject_many_statement_results_and_consume(results, result)
    }

    fn inject_statement_result_and_consume(
        &mut self,
        members: Vec<[u32; 3]>,
        result: Handle<Expression>,
        value: Value,
    ) -> Result<(), EvaluatorError> {
        let results = members
            .into_iter()
            .map(|gid| (gid, value.clone()))
            .collect();
        self.inject_many_statement_results_and_consume(results, result)
    }

    fn inject_many_statement_results_and_consume(
        &mut self,
        results: Vec<([u32; 3], Value)>,
        result: Handle<Expression>,
    ) -> Result<(), EvaluatorError> {
        for (gid, value) in results {
            let next = {
                let evaluator = self.evaluators.get_mut(&gid).ok_or_else(|| {
                    EvaluatorError::InternalError(format!("missing evaluator for {gid:?}"))
                })?;
                evaluator.set_current_expression_value(result, value)?;
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

    fn gather_target_lane(evaluator: &Evaluator, lane: u32, mode: GatherMode) -> u32 {
        match mode {
            GatherMode::BroadcastFirst => lane,
            GatherMode::Broadcast(expr) | GatherMode::Shuffle(expr) => {
                Self::evaluate_u32(evaluator, expr).unwrap_or(lane)
            }
            GatherMode::ShuffleDown(expr) => {
                lane.saturating_add(Self::evaluate_u32(evaluator, expr).unwrap_or(0))
            }
            GatherMode::ShuffleUp(expr) => {
                lane.saturating_sub(Self::evaluate_u32(evaluator, expr).unwrap_or(0))
            }
            GatherMode::ShuffleXor(expr) => lane ^ Self::evaluate_u32(evaluator, expr).unwrap_or(0),
            GatherMode::QuadBroadcast(expr) => {
                let index = Self::evaluate_u32(evaluator, expr).unwrap_or(0) % 4;
                (lane / 4) * 4 + index
            }
            GatherMode::QuadSwap(direction) => {
                lane ^ match direction {
                    naga::Direction::X => 1,
                    naga::Direction::Y => 2,
                    naga::Direction::Diagonal => 3,
                }
            }
        }
    }

    fn evaluate_u32(evaluator: &Evaluator, expr: Handle<Expression>) -> Option<u32> {
        match evaluator.evaluate_expression(expr) {
            Value::Primitive(Primitive::U32(value)) => Some(value),
            Value::Primitive(Primitive::I32(value)) if value >= 0 => Some(value as u32),
            _ => None,
        }
    }

    fn reduce_subgroup_values(
        op: SubgroupOperation,
        values: Vec<Value>,
    ) -> Result<Value, EvaluatorError> {
        if values.is_empty() {
            return Ok(Value::Uninitialized);
        }

        match op {
            SubgroupOperation::All => Ok(Value::Primitive(Primitive::U32(u32::from(
                values.iter().all(Value::is_truthy),
            )))),
            SubgroupOperation::Any => Ok(Value::Primitive(Primitive::U32(u32::from(
                values.iter().any(Value::is_truthy),
            )))),
            SubgroupOperation::Add
            | SubgroupOperation::Mul
            | SubgroupOperation::Min
            | SubgroupOperation::Max
            | SubgroupOperation::And
            | SubgroupOperation::Or
            | SubgroupOperation::Xor => {
                let mut iter = values.into_iter();
                let first = iter.next().unwrap();
                iter.try_fold(first, |acc, value| {
                    Self::combine_subgroup_values(op, acc, value)
                })
            }
        }
    }

    fn scan_subgroup_values(
        op: SubgroupOperation,
        lane_values: &[([u32; 3], Value)],
        inclusive: bool,
    ) -> Result<Vec<([u32; 3], Value)>, EvaluatorError> {
        let Some((_, first_value)) = lane_values.first() else {
            return Ok(Vec::new());
        };
        let mut acc = Self::identity_subgroup_value(op, first_value)?;
        let mut results = Vec::new();
        for (gid, value) in lane_values {
            if inclusive {
                acc = Self::combine_subgroup_values(op, acc, value.clone())?;
                results.push((*gid, acc.clone()));
            } else {
                results.push((*gid, acc.clone()));
                acc = Self::combine_subgroup_values(op, acc, value.clone())?;
            }
        }
        Ok(results)
    }

    fn combine_subgroup_values(
        op: SubgroupOperation,
        left: Value,
        right: Value,
    ) -> Result<Value, EvaluatorError> {
        let (Value::Primitive(left), Value::Primitive(right)) = (left, right) else {
            return Ok(Value::Uninitialized);
        };

        let primitive = match op {
            SubgroupOperation::Add => left + right,
            SubgroupOperation::Mul => left * right,
            SubgroupOperation::Min => Self::primitive_min(left, right)?,
            SubgroupOperation::Max => Self::primitive_max(left, right)?,
            SubgroupOperation::And => left & right,
            SubgroupOperation::Or => left | right,
            SubgroupOperation::Xor => left ^ right,
            SubgroupOperation::All | SubgroupOperation::Any => {
                return Ok(Value::Primitive(Primitive::U32(u32::from(match op {
                    SubgroupOperation::All => {
                        Value::Primitive(left).is_truthy() && Value::Primitive(right).is_truthy()
                    }
                    SubgroupOperation::Any => {
                        Value::Primitive(left).is_truthy() || Value::Primitive(right).is_truthy()
                    }
                    _ => unreachable!(),
                }))));
            }
        };

        Ok(Value::Primitive(primitive))
    }

    fn identity_subgroup_value(
        op: SubgroupOperation,
        sample: &Value,
    ) -> Result<Value, EvaluatorError> {
        let Value::Primitive(sample) = sample else {
            return Ok(Value::Uninitialized);
        };
        let primitive = match op {
            SubgroupOperation::Add => {
                Self::map_primitive_components(*sample, |_| 0.0, |_| 0, |_| 0)
            }
            SubgroupOperation::Mul => {
                Self::map_primitive_components(*sample, |_| 1.0, |_| 1, |_| 1)
            }
            other => {
                return Err(EvaluatorError::UnsupportedStatement(format!(
                    "subgroup {:?} scan",
                    other
                )));
            }
        };
        Ok(Value::Primitive(primitive))
    }

    fn primitive_min(left: Primitive, right: Primitive) -> Result<Primitive, EvaluatorError> {
        left.zip_map_numeric(right, f32::min, i32::min, u32::min)
            .or_else(|| match (left, right) {
                (Primitive::F64(a), Primitive::F64(b)) => Some(Primitive::F64(a.min(b))),
                (Primitive::I64(a), Primitive::I64(b)) => Some(Primitive::I64(a.min(b))),
                (Primitive::U64(a), Primitive::U64(b)) => Some(Primitive::U64(a.min(b))),
                _ => None,
            })
            .ok_or_else(|| EvaluatorError::UnsupportedStatement("subgroup min type".to_string()))
    }

    fn primitive_max(left: Primitive, right: Primitive) -> Result<Primitive, EvaluatorError> {
        left.zip_map_numeric(right, f32::max, i32::max, u32::max)
            .or_else(|| match (left, right) {
                (Primitive::F64(a), Primitive::F64(b)) => Some(Primitive::F64(a.max(b))),
                (Primitive::I64(a), Primitive::I64(b)) => Some(Primitive::I64(a.max(b))),
                (Primitive::U64(a), Primitive::U64(b)) => Some(Primitive::U64(a.max(b))),
                _ => None,
            })
            .ok_or_else(|| EvaluatorError::UnsupportedStatement("subgroup max type".to_string()))
    }

    fn map_primitive_components(
        primitive: Primitive,
        ff32: impl Fn(f32) -> f32 + Copy,
        fi32: impl Fn(i32) -> i32 + Copy,
        fu32: impl Fn(u32) -> u32 + Copy,
    ) -> Primitive {
        primitive
            .map_numeric(ff32, fi32, fu32)
            .unwrap_or(match primitive {
                Primitive::F64(_) => Primitive::F64(ff32(0.0) as f64),
                Primitive::I64(_) => Primitive::I64(fi32(0) as i64),
                Primitive::U64(_) => Primitive::U64(fu32(0) as u64),
                other => other,
            })
    }
}
