var<private> sum_value: u32;
var<private> inclusive_value: u32;
var<private> from_lane_two_value: u32;
var<private> ballot_value: vec4<u32>;
var<private> stop_value: u32;

@compute @workgroup_size(4, 1, 1)
fn main(@builtin(subgroup_invocation_id) lane: u32) {
    let value = lane + 1u;
    let sum = subgroupAdd(value);
    let inclusive = subgroupInclusiveAdd(value);
    let from_lane_two = subgroupBroadcast(lane + 10u, 2u);
    let ballot = subgroupBallot(lane < 2u);
    sum_value = sum;
    inclusive_value = inclusive;
    from_lane_two_value = from_lane_two;
    ballot_value = ballot;
    stop_value = sum + inclusive + from_lane_two + ballot.x;
}
